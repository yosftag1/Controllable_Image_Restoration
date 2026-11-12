"""Thread-safe Switchable Self-Teacher LoRA & Action-Modulated Dual-LoRA layers.

Supports:
1. SwitchableLoRALinear: On/off LoRA switch on frozen Vision Foundation Models
   (C-RADIOv2.5 / DINOv3 / SigLIP2) for zero-extra-VRAM Clean Self-Teacher alignment.
2. ActionDualLoRALinear & ActionDualLoRAConv2d: Combines PiSA-SR Dual-LoRA
   (decoupled Pixel-Faithful LoRA vs. Semantic-Detail LoRA) with S3Diff Action Gating.
"""

from contextlib import contextmanager
import math
import threading
import types
from typing import Iterable, List, Optional, Tuple

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
    """Make Hugging Face / remote-code modules safe for multi-GPU torch.nn.DataParallel replication.

    1. Registers a non-persistent `_dp_anchor` buffer so `.dtype` / `.device` / `buffers()` never raise
       `StopIteration` on replica >= 1 (where PyTorch `replicate()` converts Parameters into Tensors).
    2. Promotes instance-bound `types.MethodType` monkey-patches (e.g. C-RADIO `_forward_cpe`) onto a
       dynamic subclass so `DataParallel` replicas bind `self` to each GPU's local replica instead of `cuda:0`.
    """
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


_THREAD_STATE = threading.local()


def _get_vision_lora_enabled() -> bool:
    return getattr(_THREAD_STATE, "vision_lora_enabled", True)


def _get_generator_lora_state() -> Tuple[Optional[torch.Tensor], float, float]:
    return (
        getattr(_THREAD_STATE, "gen_action_cond", None),
        getattr(_THREAD_STATE, "gen_lambda_pix", 1.0),
        getattr(_THREAD_STATE, "gen_lambda_sem", 0.75),
    )


@contextmanager
def vision_lora_context(enabled: bool):
    """Toggle Vision Encoder LoRA detours on (student) or off (clean self-teacher)."""
    prev = getattr(_THREAD_STATE, "vision_lora_enabled", True)
    _THREAD_STATE.vision_lora_enabled = bool(enabled)
    try:
        yield
    finally:
        _THREAD_STATE.vision_lora_enabled = prev


def set_generator_lora_state(
    action_cond: Optional[torch.Tensor],
    lambda_pix: float = 1.0,
    lambda_sem: float = 0.75,
) -> None:
    """Persist thread-local action conditioning across forward + gradient-checkpointed backward."""
    _THREAD_STATE.gen_action_cond = action_cond
    _THREAD_STATE.gen_lambda_pix = float(lambda_pix)
    _THREAD_STATE.gen_lambda_sem = float(lambda_sem)


def clear_generator_lora_state() -> None:
    """Release thread-local action conditioning tensor references after backward completes."""
    _THREAD_STATE.gen_action_cond = None
    _THREAD_STATE.gen_lambda_pix = 1.0
    _THREAD_STATE.gen_lambda_sem = 0.75


@contextmanager
def generator_lora_context(
    action_cond: Optional[torch.Tensor],
    lambda_pix: float = 1.0,
    lambda_sem: float = 0.75,
):
    """Provide thread-local action conditioning & Dual-LoRA scales."""
    prev_cond = getattr(_THREAD_STATE, "gen_action_cond", None)
    prev_pix = getattr(_THREAD_STATE, "gen_lambda_pix", 1.0)
    prev_sem = getattr(_THREAD_STATE, "gen_lambda_sem", 0.75)

    set_generator_lora_state(action_cond, lambda_pix=lambda_pix, lambda_sem=lambda_sem)
    try:
        yield
    finally:
        _THREAD_STATE.gen_action_cond = prev_cond
        _THREAD_STATE.gen_lambda_pix = prev_pix
        _THREAD_STATE.gen_lambda_sem = prev_sem


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
        if not _get_vision_lora_enabled():
            return base_out
        x_in = x if torch.is_autocast_enabled() else x.to(self.lora_A.weight.dtype)
        delta = self.lora_B(self.lora_A(self.dropout(x_in))) * self.scaling
        return base_out + delta.to(base_out.dtype)


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


class _DualLoRABranch(nn.Module):
    """Single low-rank branch (A, B) with an S3Diff-style action rank gate."""

    def __init__(self, in_dim: int, out_dim: int, rank: int, action_dim: int, is_conv: bool = False, kernel_size=1, padding=0, stride=1, device=None):
        super().__init__()
        self.is_conv = is_conv
        self.rank = rank
        if is_conv:
            self.down = nn.Conv2d(in_dim, rank, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, device=device, dtype=torch.float32)
            self.up = nn.Conv2d(rank, out_dim, kernel_size=1, stride=1, padding=0, bias=False, device=device, dtype=torch.float32)
        else:
            self.down = nn.Linear(in_dim, rank, bias=False, device=device, dtype=torch.float32)
            self.up = nn.Linear(rank, out_dim, bias=False, device=device, dtype=torch.float32)

        self.gate_proj = nn.Linear(action_dim, rank, bias=True, device=device, dtype=torch.float32)

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, x: torch.Tensor, action_cond: Optional[torch.Tensor]) -> torch.Tensor:
        x_in = x if torch.is_autocast_enabled() else x.to(self.down.weight.dtype)
        h = self.down(x_in)
        if action_cond is not None:
            cond_in = action_cond if torch.is_autocast_enabled() else action_cond.to(self.gate_proj.weight.dtype)
            # Centered at 1.0 + tanh(...) in [0.0, 2.0] so individual rank channels scale up/down per action
            gate = 1.0 + torch.tanh(self.gate_proj(cond_in))
            h = h * _broadcast_rank_gate(gate.to(h.dtype), h)
        return self.up(h)


class ActionDualLoRALinear(nn.Module):
    """Frozen nn.Linear augmented with Pixel-Faithful LoRA + Semantic-Detail LoRA."""

    def __init__(self, base_linear: nn.Linear, action_dim: int, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.scaling = float(alpha) / max(1, rank)
        device = base_linear.weight.device
        in_f, out_f = base_linear.in_features, base_linear.out_features

        self.pix_branch = _DualLoRABranch(in_f, out_f, rank, action_dim, is_conv=False, device=device)
        self.sem_branch = _DualLoRABranch(in_f, out_f, rank, action_dim, is_conv=False, device=device)

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
        action_cond, lam_pix, lam_sem = _get_generator_lora_state()
        if abs(lam_pix) < 1e-6 and abs(lam_sem) < 1e-6:
            return out

        if action_cond is not None and action_cond.device != x.device:
            action_cond = action_cond.to(x.device)

        delta = None
        if abs(lam_pix) >= 1e-6:
            delta = (lam_pix * self.scaling) * self.pix_branch(x, action_cond)
        if abs(lam_sem) >= 1e-6:
            sem_d = (lam_sem * self.scaling) * self.sem_branch(x, action_cond)
            delta = sem_d if delta is None else (delta + sem_d)

        return out if delta is None else (out + delta.to(out.dtype))


class ActionDualLoRAConv2d(nn.Module):
    """Frozen nn.Conv2d augmented with Pixel-Faithful LoRA + Semantic-Detail LoRA."""

    def __init__(self, base_conv: nn.Conv2d, action_dim: int, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad = False

        self.scaling = float(alpha) / max(1, rank)
        device = base_conv.weight.device

        self.pix_branch = _DualLoRABranch(
            base_conv.in_channels,
            base_conv.out_channels,
            rank,
            action_dim,
            is_conv=True,
            kernel_size=base_conv.kernel_size,
            padding=base_conv.padding,
            stride=base_conv.stride,
            device=device,
        )
        self.sem_branch = _DualLoRABranch(
            base_conv.in_channels,
            base_conv.out_channels,
            rank,
            action_dim,
            is_conv=True,
            kernel_size=base_conv.kernel_size,
            padding=base_conv.padding,
            stride=base_conv.stride,
            device=device,
        )

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
        action_cond, lam_pix, lam_sem = _get_generator_lora_state()
        if abs(lam_pix) < 1e-6 and abs(lam_sem) < 1e-6:
            return out

        if action_cond is not None and action_cond.device != x.device:
            action_cond = action_cond.to(x.device)

        delta = None
        if abs(lam_pix) >= 1e-6:
            delta = (lam_pix * self.scaling) * self.pix_branch(x, action_cond)
        if abs(lam_sem) >= 1e-6:
            sem_d = (lam_sem * self.scaling) * self.sem_branch(x, action_cond)
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
    target_keywords: Iterable[str] = ("qkv", "q_proj", "k_proj", "v_proj", "out_proj", "proj", "query", "key", "value", "fc1", "fc2"),
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
    rank: int = 16,
    alpha: float = 16.0,
    include_conv: bool = True,
    linear_keywords: Iterable[str] = ("to_q", "to_k", "to_v", "to_out", "proj_in", "proj_out", "ff.net"),
    conv_keywords: Iterable[str] = ("conv1", "conv2", "proj_in", "proj_out"),
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Freeze `model` and inject ActionDualLoRA into attention/FFN linears and ResNet convs.

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
        wrapped = ActionDualLoRALinear(mod, action_dim=action_dim, rank=rank, alpha=alpha)
        _set_submodule(model, name, wrapped)
        pix_params.extend(list(wrapped.pix_branch.parameters()))
        sem_params.extend(list(wrapped.sem_branch.parameters()))

    conv_rank = max(4, rank // 2)
    conv_alpha = float(alpha) * (conv_rank / max(1, rank))
    for name, mod in conv_targets:
        wrapped = ActionDualLoRAConv2d(mod, action_dim=action_dim, rank=conv_rank, alpha=conv_alpha)
        _set_submodule(model, name, wrapped)
        pix_params.extend(list(wrapped.pix_branch.parameters()))
        sem_params.extend(list(wrapped.sem_branch.parameters()))

    return pix_params, sem_params
