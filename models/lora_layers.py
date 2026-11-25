"""Thread-safe Switchable Shared+Specialist LoRA-MoE for Generator UNet and VAE Encoder.

Provides:
1. Device-scoped `generator_lora_context(lambda_pix, lambda_sem, spatial_action_map=...)`
   with O(1) cached spatial routing maps. Turning `lambda_pix=0, lambda_sem=0` exposes
   the exact frozen base UNet/VAE for target encoding and Stage-2 CSD.
2. Clean `DualLoRALinear` & `DualLoRAConv2d` combining:
   - An always-active shared base branch (`shared_branch`, e.g. `r=20`)
   - Additive slider-routed standard specialist branches (`pix_experts`, e.g. `r=8` each)
   - An optional semantic-detail branch (`sem_branch`, Stage 2)
"""

from contextlib import contextmanager
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def infer_module_dtype(mod: nn.Module, fallback: torch.dtype = torch.float32) -> torch.dtype:
    """Safe dtype lookup across wrapped UNet / VAE modules."""
    conv_in = getattr(mod, "conv_in", None)
    if conv_in is not None and getattr(conv_in, "weight", None) is not None:
        return conv_in.weight.dtype
    w = getattr(mod, "weight", None)
    if isinstance(w, torch.Tensor):
        return w.dtype
    try:
        return next(mod.parameters()).dtype
    except StopIteration:
        pass
    try:
        return next(mod.buffers()).dtype
    except StopIteration:
        return fallback


def infer_module_device(mod: nn.Module) -> torch.device:
    """Safe device lookup across wrapped UNet / VAE modules."""
    conv_in = getattr(mod, "conv_in", None)
    if conv_in is not None and getattr(conv_in, "weight", None) is not None:
        return conv_in.weight.device
    w = getattr(mod, "weight", None)
    if isinstance(w, torch.Tensor):
        return w.device
    try:
        return next(mod.parameters()).device
    except StopIteration:
        pass
    try:
        return next(mod.buffers()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_LoRAStateTuple = Tuple[float, float, Optional[torch.Tensor]]
_DEFAULT_LORA_STATE: _LoRAStateTuple = (1.0, 0.65, None)
_LORA_STATE_BY_DEVICE: Dict[str, _LoRAStateTuple] = {}
_ROUTE_PYRAMID_CACHE_BY_DEVICE: Dict[str, Dict[Tuple, torch.Tensor]] = {}


def _resolve_device_key(
    device: Optional[torch.device] = None,
    tensor: Optional[torch.Tensor] = None,
) -> Optional[str]:
    """Normalize a torch.device (e.g. 'cuda' -> 'cuda:0') so forward and autograd threads match."""
    dev = device if device is not None else (tensor.device if tensor is not None else None)
    if dev is None:
        return None
    if isinstance(dev, str):
        dev = torch.device(dev)
    if dev.type == "cuda":
        idx = torch.cuda.current_device() if dev.index is None else dev.index
        return f"cuda:{idx}"
    return dev.type


def _get_generator_lora_state(device: Optional[torch.device] = None) -> _LoRAStateTuple:
    if device is not None:
        key = _resolve_device_key(device=device)
        if key is not None and key in _LORA_STATE_BY_DEVICE:
            return _LORA_STATE_BY_DEVICE[key]
    return _DEFAULT_LORA_STATE


def set_generator_lora_state(
    lambda_pix: float = 1.0,
    lambda_sem: float = 0.65,
    device: Optional[torch.device] = None,
    spatial_action_map: Optional[torch.Tensor] = None,
) -> None:
    """Persist device-scoped Dual-LoRA scales & spatial slider map across forward + C++ autograd."""
    global _DEFAULT_LORA_STATE
    state: _LoRAStateTuple = (float(lambda_pix), float(lambda_sem), spatial_action_map)
    key = _resolve_device_key(device=device, tensor=spatial_action_map)
    if key is not None:
        _LORA_STATE_BY_DEVICE[key] = state
        _ROUTE_PYRAMID_CACHE_BY_DEVICE.pop(key, None)
    else:
        _DEFAULT_LORA_STATE = state
        _ROUTE_PYRAMID_CACHE_BY_DEVICE.clear()


def clear_generator_lora_state(device: Optional[torch.device] = None) -> None:
    """Release device-scoped Dual-LoRA state and route caches after backward completes."""
    global _DEFAULT_LORA_STATE
    if device is not None:
        key = _resolve_device_key(device=device)
        if key is not None:
            _LORA_STATE_BY_DEVICE.pop(key, None)
            _ROUTE_PYRAMID_CACHE_BY_DEVICE.pop(key, None)
            return
    _LORA_STATE_BY_DEVICE.clear()
    _ROUTE_PYRAMID_CACHE_BY_DEVICE.clear()
    _DEFAULT_LORA_STATE = (1.0, 0.65, None)


@contextmanager
def generator_lora_context(
    lambda_pix: float = 1.0,
    lambda_sem: float = 0.65,
    device: Optional[torch.device] = None,
    spatial_action_map: Optional[torch.Tensor] = None,
):
    """Scoped context manager setting device-scoped Dual-LoRA scales and spatial routing map."""
    global _DEFAULT_LORA_STATE
    state: _LoRAStateTuple = (float(lambda_pix), float(lambda_sem), spatial_action_map)
    key = _resolve_device_key(device=device, tensor=spatial_action_map)
    if key is not None:
        had_key = key in _LORA_STATE_BY_DEVICE
        prev_val = _LORA_STATE_BY_DEVICE.get(key)
        prev_cache = _ROUTE_PYRAMID_CACHE_BY_DEVICE.pop(key, None)
        _LORA_STATE_BY_DEVICE[key] = state
        try:
            yield
        finally:
            _ROUTE_PYRAMID_CACHE_BY_DEVICE.pop(key, None)
            if prev_cache is not None:
                _ROUTE_PYRAMID_CACHE_BY_DEVICE[key] = prev_cache
            if had_key and prev_val is not None:
                _LORA_STATE_BY_DEVICE[key] = prev_val
            else:
                _LORA_STATE_BY_DEVICE.pop(key, None)
    else:
        prev_default = _DEFAULT_LORA_STATE
        prev_map = dict(_LORA_STATE_BY_DEVICE)
        prev_all_cache = dict(_ROUTE_PYRAMID_CACHE_BY_DEVICE)
        _LORA_STATE_BY_DEVICE.clear()
        _ROUTE_PYRAMID_CACHE_BY_DEVICE.clear()
        _DEFAULT_LORA_STATE = state
        try:
            yield
        finally:
            _LORA_STATE_BY_DEVICE.clear()
            _LORA_STATE_BY_DEVICE.update(prev_map)
            _ROUTE_PYRAMID_CACHE_BY_DEVICE.clear()
            _ROUTE_PYRAMID_CACHE_BY_DEVICE.update(prev_all_cache)
            _DEFAULT_LORA_STATE = prev_default


def resolve_expert_specs(
    expert_specs: Optional[Sequence[Dict]],
    degradation_types: Optional[Sequence[str]],
    signed_types: Optional[Sequence[str]] = ("exposure", "warmth"),
    default_rank: int = 8,
) -> List[Dict]:
    """Resolve `LORA_EXPERT_SPECS` into concrete `[{"name", "indices", "rank"}]` specs."""
    if not expert_specs or not degradation_types:
        return []
    deg_list = list(degradation_types)
    signed_set = set(signed_types or ())
    name_to_idx = {n: i for i, n in enumerate(deg_list)}

    resolved: List[Dict] = []
    covered_indices = set()
    for raw in expert_specs:
        exp_name = str(raw.get("name", f"expert_{len(resolved)}"))
        sliders = list(raw.get("sliders", []))
        indices = [name_to_idx[s] for s in sliders if s in name_to_idx]
        if not indices:
            continue
        covered_indices.update(indices)
        erank = max(2, int(raw.get("rank", default_rank)))
        resolved.append({"name": exp_name, "indices": indices, "rank": erank})

    unassigned = [
        i for i, n in enumerate(deg_list) if (n not in signed_set) and (i not in covered_indices)
    ]
    if unassigned:
        resolved.append({"name": "extra_specialist", "indices": unassigned, "rank": max(2, int(default_rank))})
    return resolved


def _get_cached_resized_action_map(
    spatial_action_map: torch.Tensor,
    target_hw: Tuple[int, int],
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """O(1) cached `(B, K, H_f, W_f)` absolute slider pyramid lookup."""
    dev_key = _resolve_device_key(device=device) or "default"
    dev_cache = _ROUTE_PYRAMID_CACHE_BY_DEVICE.setdefault(dev_key, {})
    cache_key = ("2d", int(spatial_action_map.data_ptr()), batch_size, target_hw[0], target_hw[1])
    cached = dev_cache.get(cache_key)
    if cached is not None:
        return cached

    a_map = spatial_action_map.detach().float()
    if a_map.device != device:
        a_map = a_map.to(device)
    if a_map.shape[0] != batch_size:
        if batch_size % max(1, a_map.shape[0]) == 0:
            a_map = a_map.repeat_interleave(batch_size // a_map.shape[0], dim=0)
        else:
            return None

    a_abs = a_map.abs().clamp_(0.0, 1.0)
    if a_abs.shape[-2:] != target_hw:
        a_local = F.interpolate(a_abs, size=target_hw, mode="area")
    else:
        a_local = a_abs
    dev_cache[cache_key] = a_local
    return a_local


def _compute_conv_expert_route(
    spatial_action_map: Optional[torch.Tensor],
    indices: Tuple[int, ...],
    target_hw: Tuple[int, int],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute [B, 1, H_f, W_f] continuous slider routing weight (exact 0.0 when expert sliders == 0)."""
    if spatial_action_map is None or not indices:
        return torch.ones((batch_size, 1, 1, 1), device=device, dtype=torch.float32)

    dev_key = _resolve_device_key(device=device) or "default"
    dev_cache = _ROUTE_PYRAMID_CACHE_BY_DEVICE.setdefault(dev_key, {})
    exp_key = ("conv_route", int(spatial_action_map.data_ptr()), batch_size, target_hw[0], target_hw[1], indices)
    cached_r = dev_cache.get(exp_key)
    if cached_r is not None:
        return cached_r

    a_local = _get_cached_resized_action_map(spatial_action_map, target_hw, batch_size, device)
    if a_local is None:
        return torch.ones((batch_size, 1, 1, 1), device=device, dtype=torch.float32)

    if len(indices) == 1:
        i0 = indices[0]
        route = a_local[:, i0 : i0 + 1]
    else:
        idx_t = torch.as_tensor(indices, dtype=torch.long, device=device)
        route = a_local.index_select(1, idx_t).amax(dim=1, keepdim=True)
    dev_cache[exp_key] = route
    return route


def _compute_linear_expert_route(
    spatial_action_map: Optional[torch.Tensor],
    indices: Tuple[int, ...],
    x_shape: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Compute spatial token [B, N, 1] or batch [B, 1, 1] slider routing weight for Linear layers."""
    bsz = x_shape[0]
    ndim = len(x_shape)
    ones_shape = (bsz, 1) if ndim == 2 else (bsz, 1, 1)
    if spatial_action_map is None or not indices:
        return torch.ones(ones_shape, device=device, dtype=torch.float32)

    seq_len = x_shape[1] if ndim == 3 else 0
    dev_key = _resolve_device_key(device=device) or "default"
    dev_cache = _ROUTE_PYRAMID_CACHE_BY_DEVICE.setdefault(dev_key, {})
    lin_key = ("lin_route", int(spatial_action_map.data_ptr()), bsz, ndim, seq_len, indices)
    cached_l = dev_cache.get(lin_key)
    if cached_l is not None:
        return cached_l

    h_img, w_img = spatial_action_map.shape[-2], spatial_action_map.shape[-1]
    if ndim == 3 and seq_len > 1 and (h_img * w_img) % seq_len == 0:
        ratio = int(round(math.sqrt((h_img * w_img) / float(seq_len))))
        if ratio >= 1 and (h_img // ratio) * (w_img // ratio) == seq_len:
            h_tok, w_tok = h_img // ratio, w_img // ratio
            r_2d = _compute_conv_expert_route(spatial_action_map, indices, (h_tok, w_tok), bsz, device)
            tok_w = r_2d.flatten(2).transpose(1, 2).contiguous()
            dev_cache[lin_key] = tok_w
            return tok_w

    a_full = _get_cached_resized_action_map(spatial_action_map, (h_img, w_img), bsz, device)
    if a_full is None:
        return torch.ones(ones_shape, device=device, dtype=torch.float32)
    if len(indices) == 1:
        i0 = indices[0]
        act_ch = a_full[:, i0 : i0 + 1]
    else:
        idx_t = torch.as_tensor(indices, dtype=torch.long, device=device)
        act_ch = a_full.index_select(1, idx_t).amax(dim=1, keepdim=True)
    peak = act_ch.amax(dim=(2, 3))
    mean = act_ch.mean(dim=(2, 3))
    glob_w = (0.70 * peak + 0.30 * mean).clamp(0.0, 1.0)
    res = glob_w if ndim == 2 else glob_w.unsqueeze(1)
    dev_cache[lin_key] = res
    return res


class _LoRABranch(nn.Module):
    """Standard low-rank LoRA branch supporting both Linear and Conv2d layers."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        is_conv: bool = False,
        kernel_size=1,
        padding=0,
        stride=1,
        dilation=1,
        device=None,
    ):
        super().__init__()
        self.rank = max(1, int(rank))
        self.scaling = float(alpha) / float(self.rank)
        if is_conv:
            self.down = nn.Conv2d(
                in_dim,
                self.rank,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
                device=device,
                dtype=torch.float32,
            )
            self.up = nn.Conv2d(
                self.rank,
                out_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                device=device,
                dtype=torch.float32,
            )
        else:
            self.down = nn.Linear(in_dim, self.rank, bias=False, device=device, dtype=torch.float32)
            self.up = nn.Linear(self.rank, out_dim, bias=False, device=device, dtype=torch.float32)

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x if torch.is_autocast_enabled() else x.to(self.down.weight.dtype)
        return self.up(self.down(x_in)) * self.scaling


class DualLoRALinear(nn.Module):
    """Frozen nn.Linear wrapped with Shared Base LoRA + optional Specialists + Task-Routed Semantic LoRA."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        expert_specs: Optional[Sequence[Dict]] = None,
        sem_rank: int = 0,
        sem_alpha: float = 0.0,
        sem_indices: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        device = base_linear.weight.device
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.shared_branch = _LoRABranch(in_f, out_f, rank=rank, alpha=alpha, is_conv=False, device=device)

        self.expert_indices: Dict[str, Tuple[int, ...]] = {}
        self.pix_experts = nn.ModuleDict()
        for sp in expert_specs or ():
            ename = str(sp["name"])
            erank = max(2, int(sp.get("rank", 8)))
            self.expert_indices[ename] = tuple(int(i) for i in sp.get("indices", []))
            self.pix_experts[ename] = _LoRABranch(
                in_f, out_f, rank=erank, alpha=float(erank), is_conv=False, device=device
            )

        self.sem_indices: Tuple[int, ...] = tuple(int(i) for i in sem_indices) if sem_indices else ()
        if int(sem_rank) > 0:
            self.sem_branch: Optional[_LoRABranch] = _LoRABranch(
                in_f, out_f, rank=int(sem_rank), alpha=float(sem_alpha), is_conv=False, device=device
            )
        else:
            self.sem_branch = None

    def pixel_parameters(self) -> List[nn.Parameter]:
        return list(self.shared_branch.parameters()) + list(self.pix_experts.parameters())

    def semantic_parameters(self) -> List[nn.Parameter]:
        return list(self.sem_branch.parameters()) if self.sem_branch is not None else []

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
        lam_pix, lam_sem, spatial_map = _get_generator_lora_state(x.device)
        if abs(lam_pix) < 1e-6 and abs(lam_sem) < 1e-6:
            return out

        skip_zero_branches = not (self.training and torch.is_grad_enabled())
        delta = None
        if abs(lam_pix) >= 1e-6:
            pix_sum = self.shared_branch(x)
            for ename, expert in self.pix_experts.items():
                route_w = _compute_linear_expert_route(
                    spatial_map, self.expert_indices[ename], tuple(x.shape), x.device
                )
                if skip_zero_branches and float(route_w.max().item()) < 1e-6:
                    continue
                pix_sum = pix_sum + expert(x) * route_w.to(pix_sum.dtype)
            delta = float(lam_pix) * pix_sum

        if self.sem_branch is not None and abs(lam_sem) >= 1e-6:
            if self.sem_indices and spatial_map is not None:
                sem_route = _compute_linear_expert_route(
                    spatial_map, self.sem_indices, tuple(x.shape), x.device
                )
                if not (skip_zero_branches and float(sem_route.max().item()) < 1e-6):
                    sem_out = self.sem_branch(x)
                    sem_d = float(lam_sem) * sem_out * sem_route.to(sem_out.dtype)
                    delta = sem_d if delta is None else (delta + sem_d)
            else:
                sem_d = float(lam_sem) * self.sem_branch(x)
                delta = sem_d if delta is None else (delta + sem_d)
        return out if delta is None else (out + delta.to(out.dtype))


class DualLoRAConv2d(nn.Module):
    """Frozen nn.Conv2d wrapped with Shared Base LoRA + optional Specialists + Task-Routed Semantic LoRA."""

    def __init__(
        self,
        base_conv: nn.Conv2d,
        rank: int = 8,
        alpha: float = 8.0,
        expert_specs: Optional[Sequence[Dict]] = None,
        sem_rank: int = 0,
        sem_alpha: float = 0.0,
        sem_indices: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad = False

        device = base_conv.weight.device
        conv_kwargs = dict(
            is_conv=True,
            kernel_size=base_conv.kernel_size,
            padding=base_conv.padding,
            stride=base_conv.stride,
            dilation=base_conv.dilation,
            device=device,
        )
        self.shared_branch = _LoRABranch(
            base_conv.in_channels, base_conv.out_channels, rank=rank, alpha=alpha, **conv_kwargs
        )

        self.expert_indices: Dict[str, Tuple[int, ...]] = {}
        self.pix_experts = nn.ModuleDict()
        for sp in expert_specs or ():
            ename = str(sp["name"])
            erank = max(2, int(sp.get("rank", 8)))
            self.expert_indices[ename] = tuple(int(i) for i in sp.get("indices", []))
            self.pix_experts[ename] = _LoRABranch(
                base_conv.in_channels,
                base_conv.out_channels,
                rank=erank,
                alpha=float(erank),
                **conv_kwargs,
            )

        self.sem_indices: Tuple[int, ...] = tuple(int(i) for i in sem_indices) if sem_indices else ()
        if int(sem_rank) > 0:
            self.sem_branch: Optional[_LoRABranch] = _LoRABranch(
                base_conv.in_channels,
                base_conv.out_channels,
                rank=int(sem_rank),
                alpha=float(sem_alpha),
                **conv_kwargs,
            )
        else:
            self.sem_branch = None

    def pixel_parameters(self) -> List[nn.Parameter]:
        return list(self.shared_branch.parameters()) + list(self.pix_experts.parameters())

    def semantic_parameters(self) -> List[nn.Parameter]:
        return list(self.sem_branch.parameters()) if self.sem_branch is not None else []

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
        lam_pix, lam_sem, spatial_map = _get_generator_lora_state(x.device)
        if abs(lam_pix) < 1e-6 and abs(lam_sem) < 1e-6:
            return out

        skip_zero_branches = not (self.training and torch.is_grad_enabled())
        target_hw = (out.shape[-2], out.shape[-1])
        bsz = out.shape[0]
        delta = None
        if abs(lam_pix) >= 1e-6:
            pix_sum = self.shared_branch(x)
            for ename, expert in self.pix_experts.items():
                route_2d = _compute_conv_expert_route(
                    spatial_map, self.expert_indices[ename], target_hw, bsz, out.device
                )
                if skip_zero_branches and float(route_2d.max().item()) < 1e-6:
                    continue
                exp_out = expert(x)
                if exp_out.shape[-2:] != target_hw:
                    exp_out = F.interpolate(exp_out, size=target_hw, mode="bilinear", align_corners=False)
                pix_sum = pix_sum + exp_out * route_2d.to(pix_sum.dtype)
            delta = float(lam_pix) * pix_sum

        if self.sem_branch is not None and abs(lam_sem) >= 1e-6:
            if self.sem_indices and spatial_map is not None:
                sem_route = _compute_conv_expert_route(
                    spatial_map, self.sem_indices, target_hw, bsz, out.device
                )
                if not (skip_zero_branches and float(sem_route.max().item()) < 1e-6):
                    sem_out = self.sem_branch(x)
                    if sem_out.shape[-2:] != target_hw:
                        sem_out = F.interpolate(sem_out, size=target_hw, mode="bilinear", align_corners=False)
                    sem_d = float(lam_sem) * sem_out * sem_route.to(sem_out.dtype)
                    delta = sem_d if delta is None else (delta + sem_d)
            else:
                sem_d = float(lam_sem) * self.sem_branch(x)
                delta = sem_d if delta is None else (delta + sem_d)
        return out if delta is None else (out + delta.to(out.dtype))


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module):
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def inject_generator_dual_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    conv_rank: Optional[int] = None,
    conv_alpha: Optional[float] = None,
    expert_specs: Optional[Sequence[Dict]] = None,
    sem_rank: int = 0,
    sem_alpha: float = 0.0,
    sem_conv_rank: Optional[int] = None,
    sem_conv_alpha: Optional[float] = None,
    sem_indices: Optional[Sequence[int]] = None,
    include_conv: bool = True,
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
    """Freeze `model` and inject DualLoRALinear / DualLoRAConv2d layers.

    Supports decoupled linear vs. conv ranks and task-routed `sem_branch` gating.

    Returns:
        (pix_lora_params, sem_lora_params) for decoupled 2-stage optimization.
    """
    for p in model.parameters():
        p.requires_grad = False

    c_rank = int(rank) if conv_rank is None else int(conv_rank)
    c_alpha = float(alpha) if conv_alpha is None else float(conv_alpha)
    sc_rank = int(sem_rank) if sem_conv_rank is None else int(sem_conv_rank)
    sc_alpha = float(sem_alpha) if sem_conv_alpha is None else float(sem_conv_alpha)

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
        wrapped = DualLoRALinear(
            mod,
            rank=rank,
            alpha=alpha,
            expert_specs=expert_specs,
            sem_rank=sem_rank,
            sem_alpha=sem_alpha,
            sem_indices=sem_indices,
        )
        _set_submodule(model, name, wrapped)
        pix_params.extend(wrapped.pixel_parameters())
        sem_params.extend(wrapped.semantic_parameters())

    for name, mod in conv_targets:
        wrapped = DualLoRAConv2d(
            mod,
            rank=c_rank,
            alpha=c_alpha,
            expert_specs=expert_specs,
            sem_rank=sc_rank,
            sem_alpha=sc_alpha,
            sem_indices=sem_indices,
        )
        _set_submodule(model, name, wrapped)
        pix_params.extend(wrapped.pixel_parameters())
        sem_params.extend(wrapped.semantic_parameters())

    return pix_params, sem_params
