"""Thread-safe Switchable Dual-Branch LoRA for Generator UNet and VAE Encoder.

Provides:
1. Device-scoped `generator_lora_context(lambda_pix, lambda_sem)` so turning
   `lambda_pix=0.0, lambda_sem=0.0` exposes the exact frozen base UNet/VAE for
   target latent encoding and Stage-2 Target-Anchored CSD with zero extra VRAM.
2. Clean `DualLoRALinear` & `DualLoRAConv2d` wrapping frozen layers with a
   pixel-fidelity branch (`pix_branch`) and an optional semantic-detail branch
   (`sem_branch`).
"""

from contextlib import contextmanager
import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


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


_LoRAStateTuple = Tuple[float, float]
_DEFAULT_LORA_STATE: _LoRAStateTuple = (1.0, 0.50)
_LORA_STATE_BY_DEVICE: Dict[str, _LoRAStateTuple] = {}


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
    lambda_sem: float = 0.50,
    device: Optional[torch.device] = None,
) -> None:
    """Persist device-scoped Dual-LoRA scales across forward + C++ autograd."""
    global _DEFAULT_LORA_STATE
    state: _LoRAStateTuple = (float(lambda_pix), float(lambda_sem))
    key = _resolve_device_key(device=device)
    if key is not None:
        _LORA_STATE_BY_DEVICE[key] = state
    else:
        _DEFAULT_LORA_STATE = state


def clear_generator_lora_state(device: Optional[torch.device] = None) -> None:
    """Release device-scoped Dual-LoRA state after backward completes."""
    global _DEFAULT_LORA_STATE
    if device is not None:
        key = _resolve_device_key(device=device)
        if key is not None:
            _LORA_STATE_BY_DEVICE.pop(key, None)
            return
    _LORA_STATE_BY_DEVICE.clear()
    _DEFAULT_LORA_STATE = (1.0, 0.50)


@contextmanager
def generator_lora_context(
    lambda_pix: float = 1.0,
    lambda_sem: float = 0.50,
    device: Optional[torch.device] = None,
):
    """Scoped context manager setting device-scoped Dual-LoRA scales."""
    global _DEFAULT_LORA_STATE
    state: _LoRAStateTuple = (float(lambda_pix), float(lambda_sem))
    key = _resolve_device_key(device=device)
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
    """Frozen nn.Linear wrapped with unified Pixel-Fidelity and optional Semantic-Detail LoRA."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 32,
        alpha: float = 32.0,
        sem_rank: int = 0,
        sem_alpha: float = 0.0,
    ):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        device = base_linear.weight.device
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.pix_branch = _LoRABranch(in_f, out_f, rank=rank, alpha=alpha, is_conv=False, device=device)
        if int(sem_rank) > 0:
            self.sem_branch: Optional[_LoRABranch] = _LoRABranch(
                in_f, out_f, rank=int(sem_rank), alpha=float(sem_alpha), is_conv=False, device=device
            )
        else:
            self.sem_branch = None

    def pixel_parameters(self) -> List[nn.Parameter]:
        return list(self.pix_branch.parameters())

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
        lam_pix, lam_sem = _get_generator_lora_state(x.device)
        if abs(lam_pix) < 1e-6 and abs(lam_sem) < 1e-6:
            return out

        delta = None
        if abs(lam_pix) >= 1e-6:
            delta = float(lam_pix) * self.pix_branch(x)
        if self.sem_branch is not None and abs(lam_sem) >= 1e-6:
            sem_d = float(lam_sem) * self.sem_branch(x)
            delta = sem_d if delta is None else (delta + sem_d)
        return out if delta is None else (out + delta.to(out.dtype))


class DualLoRAConv2d(nn.Module):
    """Frozen nn.Conv2d wrapped with unified Pixel-Fidelity and optional Semantic-Detail LoRA."""

    def __init__(
        self,
        base_conv: nn.Conv2d,
        rank: int = 32,
        alpha: float = 32.0,
        sem_rank: int = 0,
        sem_alpha: float = 0.0,
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
        self.pix_branch = _LoRABranch(
            base_conv.in_channels, base_conv.out_channels, rank=rank, alpha=alpha, **conv_kwargs
        )
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
        return list(self.pix_branch.parameters())

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
        lam_pix, lam_sem = _get_generator_lora_state(x.device)
        if abs(lam_pix) < 1e-6 and abs(lam_sem) < 1e-6:
            return out

        delta = None
        if abs(lam_pix) >= 1e-6:
            delta = float(lam_pix) * self.pix_branch(x)
        if self.sem_branch is not None and abs(lam_sem) >= 1e-6:
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
    rank: int = 32,
    alpha: float = 32.0,
    sem_rank: int = 0,
    sem_alpha: float = 0.0,
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

    Returns:
        (pix_lora_params, sem_lora_params) for decoupled 2-stage optimization.
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
        wrapped = DualLoRALinear(
            mod,
            rank=rank,
            alpha=alpha,
            sem_rank=sem_rank,
            sem_alpha=sem_alpha,
        )
        _set_submodule(model, name, wrapped)
        pix_params.extend(wrapped.pixel_parameters())
        sem_params.extend(wrapped.semantic_parameters())

    for name, mod in conv_targets:
        wrapped = DualLoRAConv2d(
            mod,
            rank=rank,
            alpha=alpha,
            sem_rank=sem_rank,
            sem_alpha=sem_alpha,
        )
        _set_submodule(model, name, wrapped)
        pix_params.extend(wrapped.pixel_parameters())
        sem_params.extend(wrapped.semantic_parameters())

    return pix_params, sem_params
