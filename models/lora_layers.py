"""Thread-safe Switchable Self-Teacher LoRA & Slider-Routed Mixture-of-Experts Dual-LoRA / DoRA.

Supports:
1. SwitchableLoRALinear: On/off LoRA switch on frozen Vision Foundation Models
   (C-RADIO / DINOv3 / SigLIP2) for zero-extra-VRAM Clean Self-Teacher alignment.
2. ActionDualLoRALinear & ActionDualLoRAConv2d (V5 Universal Slider-Routed UIR-LoRA + DoRA):
   - Eliminates cross-task gradient conflict (e.g., low-pass `denoise` vs. high-pass `deblur_motion`)
     by routing each task family to its own named LoRA specialist in `nn.ModuleDict` driven directly
     by the user's spatial/global sliders (`zero blind estimator`).
   - Supports directional `1x7 + 7x1` Strip-Depthwise Conv-LoRA (`motion_deconv`), `3x3 + 5x5`
     Multi-Scale Optical-Disc Conv-LoRA (`focus_optical`), and Magnitude-Direction `DoRA` scaling.
   - Plug-and-play expandable: adding new knobs to `Config.LORA_EXPERT_SPECS` later preserves
     checkpoint compatibility for all existing named experts.
   - Turning `lambda_pix=0, lambda_sem=0` bypasses all adapters cleanly, exposing the exact frozen
     base UNet for zero-extra-VRAM Stage-3 Classifier Score Distillation (`CSD` / `TSD`).
"""

from contextlib import contextmanager
import math
import types
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def infer_module_dtype(mod: nn.Module, fallback: torch.dtype = torch.float32) -> torch.dtype:
    """DataParallel-safe dtype lookup (works on replicas where _parameters become plain Tensor attributes)."""
    conv_in = getattr(mod, "conv_in", None)
    if conv_in is not None and getattr(conv_in, "weight", None) is not None:
        return conv_in.weight.dtype
    w = getattr(mod, "weight", None)
    if isinstance(w, torch.Tensor):
        return w.dtype
    bufs = getattr(mod, "_buffers", None)
    if bufs and bufs.get("_dp_anchor") is not None:
        return bufs["_dp_anchor"].dtype
    try:
        return next(mod.parameters()).dtype
    except StopIteration:
        pass
    try:
        return next(mod.buffers()).dtype
    except StopIteration:
        return fallback


def infer_module_device(mod: nn.Module) -> torch.device:
    """DataParallel-safe device lookup (works on replicas where _parameters become plain Tensor attributes)."""
    conv_in = getattr(mod, "conv_in", None)
    if conv_in is not None and getattr(conv_in, "weight", None) is not None:
        return conv_in.weight.device
    w = getattr(mod, "weight", None)
    if isinstance(w, torch.Tensor):
        return w.device
    bufs = getattr(mod, "_buffers", None)
    if bufs and bufs.get("_dp_anchor") is not None:
        return bufs["_dp_anchor"].device
    try:
        return next(mod.parameters()).device
    except StopIteration:
        pass
    try:
        return next(mod.buffers()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_module_dataparallel_safe(root: nn.Module) -> nn.Module:
    """Make Hugging Face / remote-code modules safe for multi-GPU torch.nn.DataParallel replication."""
    try:
        first_p = next(root.parameters())
        anchor_dev, anchor_dtype = first_p.device, first_p.dtype
    except StopIteration:
        anchor_dev, anchor_dtype = torch.device("cpu"), torch.float32

    if "_dp_anchor" not in root._buffers:
        root.register_buffer("_dp_anchor", torch.zeros(1, device=anchor_dev, dtype=anchor_dtype), persistent=False)

    for mod in root.modules():
        bound_methods = {
            k: v.__func__
            for k, v in list(mod.__dict__.items())
            if isinstance(v, types.MethodType) and v.__self__ is mod
        }
        has_dtype_prop = isinstance(getattr(type(mod), "dtype", None), property)
        has_device_prop = isinstance(getattr(type(mod), "device", None), property)
        needs_props = (has_dtype_prop or has_device_prop) and not getattr(type(mod), "_is_dp_safe_cls", False)

        if needs_props and "_dp_anchor" not in mod._buffers:
            mod.register_buffer("_dp_anchor", torch.zeros(1, device=anchor_dev, dtype=anchor_dtype), persistent=False)

        if bound_methods or needs_props:
            cls_dict = dict(bound_methods)
            cls_dict["_is_dp_safe_cls"] = True
            if needs_props:
                if has_dtype_prop:
                    cls_dict["dtype"] = property(infer_module_dtype)
                if has_device_prop:
                    cls_dict["device"] = property(infer_module_device)
            try:
                sub_cls = type(f"{type(mod).__name__}_DPSafe", (type(mod),), cls_dict)
                mod.__class__ = sub_cls
                for k in bound_methods:
                    del mod.__dict__[k]
            except TypeError:
                pass

    return root


_LoRAStateTuple = Tuple[Optional[torch.Tensor], float, float, Optional[torch.Tensor]]

_DEFAULT_VISION_LORA_ENABLED: bool = True
_VISION_LORA_BY_DEVICE: Dict[str, bool] = {}

_DEFAULT_LORA_STATE: _LoRAStateTuple = (None, 1.0, 0.75, None)
_LORA_STATE_BY_DEVICE: Dict[str, _LoRAStateTuple] = {}


def _resolve_device_key(
    device: Optional[torch.device] = None,
    tensor: Optional[torch.Tensor] = None,
) -> Optional[str]:
    """Normalize a torch.device (e.g. 'cuda' -> 'cuda:0') so forward and C++ autograd threads match."""
    dev = device if device is not None else (tensor.device if tensor is not None else None)
    if dev is None:
        return None
    if isinstance(dev, str):
        dev = torch.device(dev)
    if dev.type == "cuda":
        idx = torch.cuda.current_device() if dev.index is None else dev.index
        return f"cuda:{idx}"
    return dev.type


def _get_vision_lora_enabled(device: Optional[torch.device] = None) -> bool:
    if device is not None:
        key = _resolve_device_key(device=device)
        if key is not None and key in _VISION_LORA_BY_DEVICE:
            return _VISION_LORA_BY_DEVICE[key]
    return _DEFAULT_VISION_LORA_ENABLED


def _get_generator_lora_state(device: Optional[torch.device] = None) -> _LoRAStateTuple:
    if device is not None:
        key = _resolve_device_key(device=device)
        if key is not None and key in _LORA_STATE_BY_DEVICE:
            return _LORA_STATE_BY_DEVICE[key]
    return _DEFAULT_LORA_STATE


@contextmanager
def vision_lora_context(enabled: bool, device: Optional[torch.device] = None):
    """Toggle Vision Encoder LoRA detours on (student) or off (clean self-teacher)."""
    global _DEFAULT_VISION_LORA_ENABLED
    flag = bool(enabled)
    key = _resolve_device_key(device=device)
    if key is not None:
        had_key = key in _VISION_LORA_BY_DEVICE
        prev_val = _VISION_LORA_BY_DEVICE.get(key, True)
        _VISION_LORA_BY_DEVICE[key] = flag
        try:
            yield
        finally:
            if had_key:
                _VISION_LORA_BY_DEVICE[key] = prev_val
            else:
                _VISION_LORA_BY_DEVICE.pop(key, None)
    else:
        prev_default = _DEFAULT_VISION_LORA_ENABLED
        prev_map = dict(_VISION_LORA_BY_DEVICE)
        _VISION_LORA_BY_DEVICE.clear()
        _DEFAULT_VISION_LORA_ENABLED = flag
        try:
            yield
        finally:
            _VISION_LORA_BY_DEVICE.clear()
            _VISION_LORA_BY_DEVICE.update(prev_map)
            _DEFAULT_VISION_LORA_ENABLED = prev_default


def set_generator_lora_state(
    action_cond: Optional[torch.Tensor],
    lambda_pix: float = 1.0,
    lambda_sem: float = 0.75,
    device: Optional[torch.device] = None,
    spatial_action_map: Optional[torch.Tensor] = None,
) -> None:
    """Persist device-scoped action conditioning & spatial slider map across forward + C++ autograd."""
    global _DEFAULT_LORA_STATE
    ref_tensor = action_cond if action_cond is not None else spatial_action_map
    state: _LoRAStateTuple = (action_cond, float(lambda_pix), float(lambda_sem), spatial_action_map)
    key = _resolve_device_key(device=device, tensor=ref_tensor)
    if key is not None:
        _LORA_STATE_BY_DEVICE[key] = state
    else:
        _DEFAULT_LORA_STATE = state


def clear_generator_lora_state(device: Optional[torch.device] = None) -> None:
    """Release device-scoped action conditioning tensor references after backward completes."""
    global _DEFAULT_LORA_STATE
    if device is not None:
        key = _resolve_device_key(device=device)
        if key is not None:
            _LORA_STATE_BY_DEVICE.pop(key, None)
            return
    _LORA_STATE_BY_DEVICE.clear()
    _DEFAULT_LORA_STATE = (None, 1.0, 0.75, None)


@contextmanager
def generator_lora_context(
    action_cond: Optional[torch.Tensor],
    lambda_pix: float = 1.0,
    lambda_sem: float = 0.75,
    device: Optional[torch.device] = None,
    spatial_action_map: Optional[torch.Tensor] = None,
):
    """Provide device-scoped action conditioning, spatial slider routing & Dual-LoRA scales."""
    global _DEFAULT_LORA_STATE
    ref_tensor = action_cond if action_cond is not None else spatial_action_map
    state: _LoRAStateTuple = (action_cond, float(lambda_pix), float(lambda_sem), spatial_action_map)
    key = _resolve_device_key(device=device, tensor=ref_tensor)
    if key is not None:
        had_key = key in _LORA_STATE_BY_DEVICE
        prev_val = _LORA_STATE_BY_DEVICE.get(key)
        _LORA_STATE_BY_DEVICE[key] = state
        try:
            yield
        finally:
            if had_key and prev_val is not None:
                _LORA_STATE_BY_DEVICE[key] = prev_val
            else:
                _LORA_STATE_BY_DEVICE.pop(key, None)
    else:
        prev_default = _DEFAULT_LORA_STATE
        prev_map = dict(_LORA_STATE_BY_DEVICE)
        _LORA_STATE_BY_DEVICE.clear()
        _DEFAULT_LORA_STATE = state
        try:
            yield
        finally:
            _LORA_STATE_BY_DEVICE.clear()
            _LORA_STATE_BY_DEVICE.update(prev_map)
            _DEFAULT_LORA_STATE = prev_default


class SwitchableLoRALinear(nn.Module):
    """Frozen nn.Linear wrapped with an on/off low-rank detour for Self-Teacher alignment."""

    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.rank = rank
        self.scaling = float(alpha) / max(1, rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        device = base_linear.weight.device
        self.lora_A = nn.Linear(in_features, rank, bias=False, device=device, dtype=torch.float32)
        self.lora_B = nn.Linear(rank, out_features, bias=False, device=device, dtype=torch.float32)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if not _get_vision_lora_enabled(x.device):
            return base_out
        x_in = x if torch.is_autocast_enabled() else x.to(self.lora_A.weight.dtype)
        delta = self.lora_B(self.lora_A(self.dropout(x_in))) * self.scaling
        return base_out + delta.to(base_out.dtype)


def resolve_expert_specs(
    expert_specs: Optional[Sequence[Dict]],
    degradation_types: Optional[Sequence[str]],
    signed_types: Optional[Sequence[str]] = ("exposure", "warmth"),
    default_rank: int = 24,
    rank_scale: float = 1.0,
) -> List[Dict]:
    """Resolve declarative `LORA_EXPERT_SPECS` into concrete `(name, slider_indices, rank, conv_mode)` specs.

    Automatically creates an `"extra_universal"` expert if the user appends new structural knobs to
    `DEGRADATION_TYPES` in the future without yet adding a custom entry to `LORA_EXPERT_SPECS`.
    """
    deg_list = list(degradation_types or [])
    signed_set = set(signed_types or ())
    name_to_idx = {n: i for i, n in enumerate(deg_list)}

    resolved: List[Dict] = []
    covered_indices = set()

    if expert_specs and deg_list:
        for raw in expert_specs:
            exp_name = str(raw.get("name", f"expert_{len(resolved)}"))
            sliders = list(raw.get("sliders", []))
            indices = [name_to_idx[s] for s in sliders if s in name_to_idx]
            if not indices:
                continue
            covered_indices.update(indices)
            r_base = int(raw.get("rank", default_rank))
            eff_rank = max(4, int(round(r_base * float(rank_scale))))
            conv_mode = str(raw.get("conv_mode", "standard_3x3"))
            resolved.append(
                {
                    "name": exp_name,
                    "indices": indices,
                    "rank": eff_rank,
                    "conv_mode": conv_mode,
                }
            )

        # Future-proof catch-all: any new structural slider added to DEGRADATION_TYPES gets routed automatically!
        unassigned = [
            i for i, n in enumerate(deg_list) if (n not in signed_set) and (i not in covered_indices)
        ]
        if unassigned:
            resolved.append(
                {
                    "name": "extra_universal",
                    "indices": unassigned,
                    "rank": max(8, int(round(default_rank * float(rank_scale)))),
                    "conv_mode": "multiscale_5x5",
                }
            )

    if not resolved:
        all_struct = [i for i, n in enumerate(deg_list) if n not in signed_set]
        resolved.append(
            {
                "name": "universal_pix",
                "indices": all_struct,
                "rank": max(4, int(round(default_rank * float(rank_scale)))),
                "conv_mode": "standard_3x3",
            }
        )
    return resolved


def _broadcast_rank_gate(gate: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """Broadcast [B, r] gate onto [B, ..., r] (linear) or [B, r, H, W] (conv)."""
    bsz = gate.shape[0]
    if hidden.shape[0] != bsz:
        if hidden.shape[0] % bsz == 0:
            gate = gate.repeat_interleave(hidden.shape[0] // bsz, dim=0)
        else:
            return torch.ones_like(hidden)

    if hidden.ndim == 2:
        return gate
    if hidden.ndim == 3:
        return gate.unsqueeze(1)
    if hidden.ndim == 4:
        return gate.unsqueeze(-1).unsqueeze(-1)
    shape = [gate.shape[0]] + [1] * (hidden.ndim - 2) + [gate.shape[1]]
    return gate.view(*shape)


def _compute_conv_expert_route(
    spatial_action_map: Optional[torch.Tensor],
    indices: Sequence[int],
    target_hw: Tuple[int, int],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute [B, 1, H_f, W_f] continuous slider routing weight (exact 0.0 when expert sliders == 0)."""
    if spatial_action_map is None or not indices:
        return torch.ones((batch_size, 1, 1, 1), device=device, dtype=torch.float32)

    a_map = spatial_action_map.float()
    if a_map.device != device:
        a_map = a_map.to(device)
    if a_map.shape[0] != batch_size:
        if batch_size % max(1, a_map.shape[0]) == 0:
            a_map = a_map.repeat_interleave(batch_size // a_map.shape[0], dim=0)
        else:
            return torch.ones((batch_size, 1, 1, 1), device=device, dtype=torch.float32)

    if a_map.shape[-2:] != target_hw:
        a_local = F.interpolate(a_map, size=target_hw, mode="area")
    else:
        a_local = a_map

    idx_t = torch.as_tensor(indices, dtype=torch.long, device=device)
    return a_local.index_select(1, idx_t).abs().amax(dim=1, keepdim=True).clamp(0.0, 1.0)


def _compute_linear_expert_route(
    spatial_action_map: Optional[torch.Tensor],
    indices: Sequence[int],
    x_shape: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Compute spatial token [B, N, 1] or batch [B, 1, 1] slider routing weight for Linear layers."""
    bsz = x_shape[0]
    ndim = len(x_shape)
    ones_shape = (bsz, 1) if ndim == 2 else (bsz, 1, 1)
    if spatial_action_map is None or not indices:
        return torch.ones(ones_shape, device=device, dtype=torch.float32)

    a_map = spatial_action_map.float()
    if a_map.device != device:
        a_map = a_map.to(device)
    if a_map.shape[0] != bsz:
        if bsz % max(1, a_map.shape[0]) == 0:
            a_map = a_map.repeat_interleave(bsz // a_map.shape[0], dim=0)
        else:
            return torch.ones(ones_shape, device=device, dtype=torch.float32)

    idx_t = torch.as_tensor(indices, dtype=torch.long, device=device)
    act_ch = a_map.index_select(1, idx_t).abs().amax(dim=1, keepdim=True).clamp(0.0, 1.0)

    # Spatial self/cross-attention query/output projections have token count L == H_tok * W_tok
    if ndim == 3:
        seq_len = x_shape[1]
        h_img, w_img = act_ch.shape[-2], act_ch.shape[-1]
        if seq_len > 1 and (h_img * w_img) % seq_len == 0:
            ratio = int(round(math.sqrt((h_img * w_img) / float(seq_len))))
            if ratio >= 1 and (h_img // ratio) * (w_img // ratio) == seq_len:
                h_tok, w_tok = h_img // ratio, w_img // ratio
                tok_w = F.interpolate(act_ch, size=(h_tok, w_tok), mode="area").flatten(2).transpose(1, 2)
                return tok_w

    # Prompt / global linear projections: peak-aware global activation
    peak = act_ch.amax(dim=(2, 3))
    mean = act_ch.mean(dim=(2, 3))
    glob_w = (0.70 * peak + 0.30 * mean).clamp(0.0, 1.0)
    return glob_w if ndim == 2 else glob_w.unsqueeze(1)


class _ExpertLoRABranch(nn.Module):
    """Task-specialized low-rank branch supporting standard, 1x7+7x1 strip, and 3x3+5x5 optical convs."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        action_dim: int,
        is_conv: bool = False,
        conv_mode: str = "standard_3x3",
        kernel_size=1,
        padding=0,
        stride=1,
        dilation=1,
        device=None,
    ):
        super().__init__()
        self.is_conv = is_conv
        self.rank = rank
        ks_tuple = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        st_tuple = (stride, stride) if isinstance(stride, int) else tuple(stride)
        dil_tuple = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        if isinstance(padding, int):
            pad_tuple = (padding, padding)
        elif isinstance(padding, tuple):
            pad_tuple = padding
        else:
            pad_tuple = None

        is_spatial_conv = is_conv and (ks_tuple != (1, 1))
        self.conv_mode = conv_mode if is_spatial_conv else "pointwise"
        self.scaling = 1.0 / float(max(1, rank))

        # Match `base_conv` geometry whenever stride != 1 or asymmetric padding is used
        # (e.g. diffusers AutoencoderKL Downsample2D pads 64->65 then runs k=3, s=2, p=0 -> 32x32!)
        same_hw_as_1x1 = (
            pad_tuple is not None
            and st_tuple == (1, 1)
            and (2 * pad_tuple[0] == dil_tuple[0] * (ks_tuple[0] - 1))
            and (2 * pad_tuple[1] == dil_tuple[1] * (ks_tuple[1] - 1))
        )
        down_ks = 1 if same_hw_as_1x1 else kernel_size
        down_st = 1 if same_hw_as_1x1 else stride
        down_pad = 0 if same_hw_as_1x1 else padding
        down_dil = 1 if same_hw_as_1x1 else dilation

        if not is_conv:
            self.down = nn.Linear(in_dim, rank, bias=False, device=device, dtype=torch.float32)
            self.up = nn.Linear(rank, out_dim, bias=False, device=device, dtype=torch.float32)
        elif self.conv_mode == "strip_1x7_7x1":
            # Oriented horizontal + vertical strip depthwise bottleneck for wide camera-shake streaks!
            self.down = nn.Conv2d(
                in_dim,
                rank,
                kernel_size=down_ks,
                stride=down_st,
                padding=down_pad,
                dilation=down_dil,
                bias=False,
                device=device,
                dtype=torch.float32,
            )
            self.dw_h = nn.Conv2d(
                rank, rank, kernel_size=(1, 7), padding=(0, 3), groups=rank, bias=False, device=device, dtype=torch.float32
            )
            self.dw_v = nn.Conv2d(
                rank, rank, kernel_size=(7, 1), padding=(3, 0), groups=rank, bias=False, device=device, dtype=torch.float32
            )
            self.dw_local = nn.Conv2d(
                rank, rank, kernel_size=3, padding=1, groups=rank, bias=False, device=device, dtype=torch.float32
            )
            self.up = nn.Conv2d(
                rank, out_dim, kernel_size=1, stride=1, padding=0, bias=False, device=device, dtype=torch.float32
            )
            for dw in (self.dw_h, self.dw_v, self.dw_local):
                nn.init.kaiming_uniform_(dw.weight, a=math.sqrt(5))
                dw.weight.data.mul_(0.35)
        elif self.conv_mode == "multiscale_5x5":
            # Multi-scale 3x3 + 5x5 depthwise bottleneck for optical lens defocus / bokeh disc inversion!
            self.down = nn.Conv2d(
                in_dim,
                rank,
                kernel_size=down_ks,
                stride=down_st,
                padding=down_pad,
                dilation=down_dil,
                bias=False,
                device=device,
                dtype=torch.float32,
            )
            self.dw_3 = nn.Conv2d(
                rank, rank, kernel_size=3, padding=1, groups=rank, bias=False, device=device, dtype=torch.float32
            )
            self.dw_5 = nn.Conv2d(
                rank, rank, kernel_size=5, padding=2, groups=rank, bias=False, device=device, dtype=torch.float32
            )
            self.up = nn.Conv2d(
                rank, out_dim, kernel_size=1, stride=1, padding=0, bias=False, device=device, dtype=torch.float32
            )
            for dw in (self.dw_3, self.dw_5):
                nn.init.kaiming_uniform_(dw.weight, a=math.sqrt(5))
                dw.weight.data.mul_(0.35)
        else:
            self.down = nn.Conv2d(
                in_dim,
                rank,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
                device=device,
                dtype=torch.float32,
            )
            self.up = nn.Conv2d(
                rank, out_dim, kernel_size=1, stride=1, padding=0, bias=False, device=device, dtype=torch.float32
            )

        self.gate_proj = nn.Linear(action_dim, rank, bias=True, device=device, dtype=torch.float32)

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, x: torch.Tensor, action_cond: Optional[torch.Tensor]) -> torch.Tensor:
        x_in = x if torch.is_autocast_enabled() else x.to(self.down.weight.dtype)
        h = self.down(x_in)
        if self.conv_mode == "strip_1x7_7x1":
            h = h + self.dw_h(h) + self.dw_v(h) + self.dw_local(h)
        elif self.conv_mode == "multiscale_5x5":
            h = h + self.dw_3(h) + self.dw_5(h)

        if action_cond is not None:
            cond_in = action_cond if torch.is_autocast_enabled() else action_cond.to(self.gate_proj.weight.dtype)
            gate = 1.0 + 2.0 * torch.tanh(self.gate_proj(cond_in))
            h = h * _broadcast_rank_gate(gate.to(h.dtype), h)
        return self.up(h) * self.scaling


class ActionDualLoRALinear(nn.Module):
    """Frozen nn.Linear with Slider-Routed UIR-LoRA Pixel Experts, DoRA Magnitude & Semantic-Detail LoRA."""

    def __init__(
        self,
        base_linear: nn.Linear,
        action_dim: int,
        rank: int = 24,
        alpha: float = 24.0,
        expert_specs: Optional[Sequence[Dict]] = None,
        sem_rank: int = 16,
        sem_alpha: float = 16.0,
        use_dora: bool = True,
    ):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.alpha_scale = float(alpha)
        self.sem_alpha_scale = float(sem_alpha)
        self.use_dora = bool(use_dora)
        device = base_linear.weight.device
        in_f, out_f = base_linear.in_features, base_linear.out_features

        specs = list(expert_specs) if expert_specs else [{"name": "universal_pix", "indices": [], "rank": rank}]
        self.expert_indices: Dict[str, List[int]] = {}
        self.pix_experts = nn.ModuleDict()
        for sp in specs:
            ename = str(sp["name"])
            erank = max(4, int(sp.get("rank", rank)))
            self.expert_indices[ename] = list(sp.get("indices", []))
            self.pix_experts[ename] = _ExpertLoRABranch(
                in_f, out_f, erank, action_dim, is_conv=False, device=device
            )

        if self.use_dora:
            self.dora_mag = nn.Parameter(torch.zeros(out_f, device=device, dtype=torch.float32))
        else:
            self.register_parameter("dora_mag", None)

        eff_sem_rank = max(4, int(sem_rank))
        self.sem_branch = _ExpertLoRABranch(
            in_f, out_f, eff_sem_rank, action_dim, is_conv=False, device=device
        )

    def pixel_parameters(self) -> List[nn.Parameter]:
        params = list(self.pix_experts.parameters())
        if self.dora_mag is not None:
            params.append(self.dora_mag)
        return params

    def semantic_parameters(self) -> List[nn.Parameter]:
        return list(self.sem_branch.parameters())

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        action_cond, lam_pix, lam_sem, spatial_map = _get_generator_lora_state(x.device)
        if abs(lam_pix) < 1e-6 and abs(lam_sem) < 1e-6:
            return out

        if action_cond is not None and action_cond.device != x.device:
            action_cond = action_cond.to(x.device)

        delta = None
        if abs(lam_pix) >= 1e-6:
            pix_sum = None
            max_route = None
            for ename, expert in self.pix_experts.items():
                route_w = _compute_linear_expert_route(
                    spatial_map, self.expert_indices[ename], tuple(x.shape), x.device
                )
                exp_out = expert(x, action_cond)
                term = exp_out * route_w.to(exp_out.dtype)
                pix_sum = term if pix_sum is None else (pix_sum + term)
                max_route = route_w if max_route is None else torch.maximum(max_route, route_w)

            delta = (lam_pix * self.alpha_scale) * pix_sum
            if self.dora_mag is not None and max_route is not None:
                # Magnitude-Direction DoRA: decoupled output channel gain gated strictly by active sliders
                mag_gain = (0.35 * torch.tanh(self.dora_mag)).view(*([1] * (out.ndim - 1)), -1)
                active_env = (min(1.0, abs(lam_pix)) * max_route).to(torch.float32)
                dora_term = (out.float() + delta.float()) * (active_env * mag_gain)
                delta = delta.float() + dora_term

        if abs(lam_sem) >= 1e-6:
            sem_d = (lam_sem * self.sem_alpha_scale) * self.sem_branch(x, action_cond)
            delta = sem_d if delta is None else (delta + sem_d)

        return out if delta is None else (out + delta.to(out.dtype))


class ActionDualLoRAConv2d(nn.Module):
    """Frozen nn.Conv2d with Spatially Slider-Routed UIR-LoRA Experts (Strip & Multi-Scale) + Semantic LoRA."""

    def __init__(
        self,
        base_conv: nn.Conv2d,
        action_dim: int,
        rank: int = 16,
        alpha: float = 16.0,
        expert_specs: Optional[Sequence[Dict]] = None,
        sem_rank: int = 12,
        sem_alpha: float = 12.0,
    ):
        super().__init__()
        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad = False

        self.alpha_scale = float(alpha)
        self.sem_alpha_scale = float(sem_alpha)
        device = base_conv.weight.device

        specs = list(expert_specs) if expert_specs else [
            {"name": "universal_pix", "indices": [], "rank": rank, "conv_mode": "standard_3x3"}
        ]
        self.expert_indices: Dict[str, List[int]] = {}
        self.pix_experts = nn.ModuleDict()
        for sp in specs:
            ename = str(sp["name"])
            raw_r = int(sp.get("rank", rank))
            conv_r = max(6, (raw_r * 3) // 4)
            cmode = str(sp.get("conv_mode", "standard_3x3"))
            self.expert_indices[ename] = list(sp.get("indices", []))
            self.pix_experts[ename] = _ExpertLoRABranch(
                base_conv.in_channels,
                base_conv.out_channels,
                conv_r,
                action_dim,
                is_conv=True,
                conv_mode=cmode,
                kernel_size=base_conv.kernel_size,
                padding=base_conv.padding,
                stride=base_conv.stride,
                dilation=base_conv.dilation,
                device=device,
            )

        eff_sem_r = max(6, (int(sem_rank) * 3) // 4)
        self.sem_branch = _ExpertLoRABranch(
            base_conv.in_channels,
            base_conv.out_channels,
            eff_sem_r,
            action_dim,
            is_conv=True,
            conv_mode="standard_3x3",
            kernel_size=base_conv.kernel_size,
            padding=base_conv.padding,
            stride=base_conv.stride,
            dilation=base_conv.dilation,
            device=device,
        )

    def pixel_parameters(self) -> List[nn.Parameter]:
        return list(self.pix_experts.parameters())

    def semantic_parameters(self) -> List[nn.Parameter]:
        return list(self.sem_branch.parameters())

    @property
    def in_channels(self) -> int:
        return self.base.in_channels

    @property
    def out_channels(self) -> int:
        return self.base.out_channels

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        action_cond, lam_pix, lam_sem, spatial_map = _get_generator_lora_state(x.device)
        if abs(lam_pix) < 1e-6 and abs(lam_sem) < 1e-6:
            return out

        if action_cond is not None and action_cond.device != x.device:
            action_cond = action_cond.to(x.device)

        target_hw = (out.shape[-2], out.shape[-1])
        bsz = out.shape[0]

        delta = None
        if abs(lam_pix) >= 1e-6:
            pix_sum = None
            for ename, expert in self.pix_experts.items():
                route_2d = _compute_conv_expert_route(
                    spatial_map, self.expert_indices[ename], target_hw, bsz, out.device
                )
                exp_out = expert(x, action_cond)
                if exp_out.shape[-2:] != target_hw:
                    exp_out = F.interpolate(exp_out, size=target_hw, mode="bilinear", align_corners=False)
                term = exp_out * route_2d.to(exp_out.dtype)
                pix_sum = term if pix_sum is None else (pix_sum + term)
            delta = (lam_pix * self.alpha_scale) * pix_sum

        if abs(lam_sem) >= 1e-6:
            sem_out = self.sem_branch(x, action_cond)
            if sem_out.shape[-2:] != target_hw:
                sem_out = F.interpolate(sem_out, size=target_hw, mode="bilinear", align_corners=False)
            sem_d = (lam_sem * self.sem_alpha_scale) * sem_out
            delta = sem_d if delta is None else (delta + sem_d)

        return out if delta is None else (out + delta.to(out.dtype))


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module):
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def inject_vision_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_keywords: Iterable[str] = (
        "qkv",
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "proj",
        "query",
        "key",
        "value",
        "fc1",
        "fc2",
    ),
) -> List[nn.Parameter]:
    """Freeze `model` and replace matching nn.Linear layers with SwitchableLoRALinear."""
    for p in model.parameters():
        p.requires_grad = False

    exclude_keywords = ("patch_embed", "patch_generator", "conditioner", "adaptor")
    targets = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not any(ex in name.lower() for ex in exclude_keywords):
            leaf = name.split(".")[-1]
            if any(kw in leaf for kw in target_keywords):
                targets.append((name, mod))

    lora_params: List[nn.Parameter] = []
    for name, mod in targets:
        wrapped = SwitchableLoRALinear(mod, rank=rank, alpha=alpha, dropout=dropout)
        _set_submodule(model, name, wrapped)
        lora_params.extend([p for p in wrapped.parameters() if p.requires_grad])
    return lora_params


def inject_generator_dual_lora(
    model: nn.Module,
    action_dim: int,
    rank: int = 24,
    alpha: float = 24.0,
    include_conv: bool = True,
    expert_specs: Optional[Sequence[Dict]] = None,
    sem_rank: int = 16,
    sem_alpha: float = 16.0,
    use_dora: bool = True,
    unfreeze_norms: bool = False,
    linear_keywords: Iterable[str] = ("to_q", "to_k", "to_v", "to_out", "proj_in", "proj_out", "ff.net"),
    conv_keywords: Iterable[str] = (
        "conv_in",
        "conv_out",
        "conv_shortcut",
        "conv1",
        "conv2",
        "downsamplers",
        "upsamplers",
        "proj_in",
        "proj_out",
    ),
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Freeze `model` and inject Slider-Routed UIR-LoRA Experts + DoRA + Semantic-Detail LoRA.

    Returns:
        (pix_lora_params, sem_lora_params) for decoupled staged or joint optimization.
    """
    for p in model.parameters():
        p.requires_grad = False

    linear_targets = []
    conv_targets = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(kw in name for kw in linear_keywords):
            linear_targets.append((name, mod))
        elif (
            include_conv
            and isinstance(mod, nn.Conv2d)
            and mod.groups == 1
            and any(kw in name for kw in conv_keywords)
        ):
            conv_targets.append((name, mod))

    pix_params: List[nn.Parameter] = []
    sem_params: List[nn.Parameter] = []

    for name, mod in linear_targets:
        wrapped = ActionDualLoRALinear(
            mod,
            action_dim=action_dim,
            rank=rank,
            alpha=alpha,
            expert_specs=expert_specs,
            sem_rank=sem_rank,
            sem_alpha=sem_alpha,
            use_dora=use_dora,
        )
        _set_submodule(model, name, wrapped)
        pix_params.extend(wrapped.pixel_parameters())
        sem_params.extend(wrapped.semantic_parameters())

    for name, mod in conv_targets:
        wrapped = ActionDualLoRAConv2d(
            mod,
            action_dim=action_dim,
            rank=rank,
            alpha=alpha,
            expert_specs=expert_specs,
            sem_rank=sem_rank,
            sem_alpha=sem_alpha,
        )
        _set_submodule(model, name, wrapped)
        pix_params.extend(wrapped.pixel_parameters())
        sem_params.extend(wrapped.semantic_parameters())

    if unfreeze_norms:
        for mod in model.modules():
            if isinstance(mod, (nn.GroupNorm, nn.LayerNorm)):
                for p in mod.parameters(recurse=False):
                    p.requires_grad = True
                    pix_params.append(p)

    return pix_params, sem_params
