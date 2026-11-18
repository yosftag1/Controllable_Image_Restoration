"""End-to-End Controllable Pretrained Restoration Model (V5 Universal Slider-Routed Experts).

Key V5 Architectural Upgrades:
1. Default `sd-2.1-base` Backbone + Zero-Extra-VRAM Target-Anchored CSD (`PiSA-SR` + `TSD-SR`):
   Eliminates `sd-turbo`'s baked-in ADD oversharpening bias and uses the frozen base UNet itself
   (`LoRA OFF`, 0 MB extra model weights) with negative prompts in Stage 3 to suppress fake ripples.
2. Expandable Slider-Routed UIR-LoRA Task Experts + Strip/Multi-Scale Conv-LoRA + DoRA:
   Routes `denoise`/`deartifact`, `deblur_motion` (`1x7+7x1` strips), and `sharpen_focus` (`5x5` discs)
   to dedicated named experts driven 100% by the user's sliders (`zero blind estimator`, ready for
   future knobs like lens distortion or flare).
3. Universal `OSDD`-style `eVAE` Spatial Skip Alignment + Frozen Natural-Manifold VAE Decoder +
   Strictly `1x1` `PointwiseRadiometricHead` + Slider-Driven Adaptive Timestep (`FideDiff`).
4. Isolated Experimental `HYPIR` Mode (`--hypir-mode` / `HYPIRConfig`) kept separate for clean testing.
"""

from contextlib import contextmanager
import gc
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .action_overlay import ActionOverlayEncoder, PointwiseSpatialModulator
from .lora_layers import (
    _get_generator_lora_state,
    _resolve_device_key,
    clear_generator_lora_state,
    generator_lora_context,
    infer_module_device,
    infer_module_dtype,
    inject_generator_dual_lora,
    resolve_expert_specs,
    set_generator_lora_state,
)
from .semantic_encoder import SemanticEncoder, load_hf_public_safe
from .skip_vae import SkipConnectedVAE, _MockTinyVAE


_DEFAULT_SPATIAL_ACTION_MAP: Optional[torch.Tensor] = None
_SPATIAL_MAP_BY_DEVICE: Dict[str, Optional[torch.Tensor]] = {}

_DEFAULT_SPATIAL_SEM_GRID: Optional[torch.Tensor] = None
_SPATIAL_SEM_GRID_BY_DEVICE: Dict[str, Optional[torch.Tensor]] = {}


def set_spatial_action_map(
    action_map: Optional[torch.Tensor],
    sem_grid: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> None:
    """Persist device-scoped spatial action map & dense 2D semantic grid across forward + C++ autograd."""
    global _DEFAULT_SPATIAL_ACTION_MAP, _DEFAULT_SPATIAL_SEM_GRID
    key = _resolve_device_key(device=device, tensor=action_map if action_map is not None else sem_grid)
    if key is not None:
        _SPATIAL_MAP_BY_DEVICE[key] = action_map
        _SPATIAL_SEM_GRID_BY_DEVICE[key] = sem_grid
    else:
        _DEFAULT_SPATIAL_ACTION_MAP = action_map
        _DEFAULT_SPATIAL_SEM_GRID = sem_grid


def _get_spatial_action_map(device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    if device is not None:
        key = _resolve_device_key(device=device)
        if key is not None and key in _SPATIAL_MAP_BY_DEVICE:
            return _SPATIAL_MAP_BY_DEVICE[key]
    return _DEFAULT_SPATIAL_ACTION_MAP


def _get_spatial_sem_grid(device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    if device is not None:
        key = _resolve_device_key(device=device)
        if key is not None and key in _SPATIAL_SEM_GRID_BY_DEVICE:
            return _SPATIAL_SEM_GRID_BY_DEVICE[key]
    return _DEFAULT_SPATIAL_SEM_GRID


def clear_spatial_action_map(device: Optional[torch.device] = None) -> None:
    """Release device-scoped spatial action map & semantic grid references after backward completes."""
    global _DEFAULT_SPATIAL_ACTION_MAP, _DEFAULT_SPATIAL_SEM_GRID
    if device is not None:
        key = _resolve_device_key(device=device)
        if key is not None:
            _SPATIAL_MAP_BY_DEVICE.pop(key, None)
            _SPATIAL_SEM_GRID_BY_DEVICE.pop(key, None)
            return
    _SPATIAL_MAP_BY_DEVICE.clear()
    _SPATIAL_SEM_GRID_BY_DEVICE.clear()
    _DEFAULT_SPATIAL_ACTION_MAP = None
    _DEFAULT_SPATIAL_SEM_GRID = None


@contextmanager
def _spatial_action_context(
    action_map: Optional[torch.Tensor],
    sem_grid: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
):
    global _DEFAULT_SPATIAL_ACTION_MAP, _DEFAULT_SPATIAL_SEM_GRID
    key = _resolve_device_key(device=device, tensor=action_map if action_map is not None else sem_grid)
    if key is not None:
        had_act = key in _SPATIAL_MAP_BY_DEVICE
        prev_act = _SPATIAL_MAP_BY_DEVICE.get(key)
        had_sem = key in _SPATIAL_SEM_GRID_BY_DEVICE
        prev_sem = _SPATIAL_SEM_GRID_BY_DEVICE.get(key)
        _SPATIAL_MAP_BY_DEVICE[key] = action_map
        _SPATIAL_SEM_GRID_BY_DEVICE[key] = sem_grid
        try:
            yield
        finally:
            if had_act:
                _SPATIAL_MAP_BY_DEVICE[key] = prev_act
            else:
                _SPATIAL_MAP_BY_DEVICE.pop(key, None)
            if had_sem:
                _SPATIAL_SEM_GRID_BY_DEVICE[key] = prev_sem
            else:
                _SPATIAL_SEM_GRID_BY_DEVICE.pop(key, None)
    else:
        prev_def_act = _DEFAULT_SPATIAL_ACTION_MAP
        prev_def_sem = _DEFAULT_SPATIAL_SEM_GRID
        prev_act_map = dict(_SPATIAL_MAP_BY_DEVICE)
        prev_sem_map = dict(_SPATIAL_SEM_GRID_BY_DEVICE)
        _SPATIAL_MAP_BY_DEVICE.clear()
        _SPATIAL_SEM_GRID_BY_DEVICE.clear()
        _DEFAULT_SPATIAL_ACTION_MAP = action_map
        _DEFAULT_SPATIAL_SEM_GRID = sem_grid
        try:
            yield
        finally:
            _SPATIAL_MAP_BY_DEVICE.clear()
            _SPATIAL_MAP_BY_DEVICE.update(prev_act_map)
            _SPATIAL_SEM_GRID_BY_DEVICE.clear()
            _SPATIAL_SEM_GRID_BY_DEVICE.update(prev_sem_map)
            _DEFAULT_SPATIAL_ACTION_MAP = prev_def_act
            _DEFAULT_SPATIAL_SEM_GRID = prev_def_sem


def _build_default_alphas_cumprod(
    num_train_timesteps: int = 1000,
    beta_start: float = 0.00085,
    beta_end: float = 0.012,
) -> torch.Tensor:
    betas = torch.linspace(
        math.sqrt(beta_start), math.sqrt(beta_end), num_train_timesteps, dtype=torch.float64
    ) ** 2
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0).float()


def _precompute_openclip_prompt_pair(
    repo_id: str,
    pos_prompt: str,
    neg_prompt: str,
    cross_attn_dim: int,
    prompt_seq_len: int = 77,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute positive & negative CLIP text embeddings on CPU (0 MB text-encoder GPU VRAM!)."""
    is_rank0 = os.environ.get("RANK", "0") == "0"
    try:
        from transformers import AutoTokenizer, CLIPTextModel

        if is_rank0:
            print(f"[PromptAnchor] Computing pos/neg CLIP prompt anchors from '{repo_id}'...", flush=True)
        tokenizer = load_hf_public_safe(AutoTokenizer.from_pretrained, repo_id, subfolder="tokenizer")
        try:
            text_encoder = load_hf_public_safe(
                CLIPTextModel.from_pretrained,
                repo_id,
                subfolder="text_encoder",
                variant="fp16",
                torch_dtype=torch.float32,
            )
        except Exception:
            text_encoder = load_hf_public_safe(CLIPTextModel.from_pretrained, repo_id, subfolder="text_encoder")
        text_encoder.eval()
        max_len = int(getattr(tokenizer, "model_max_length", prompt_seq_len) or prompt_seq_len)
        if max_len <= 0 or max_len > 256:
            max_len = prompt_seq_len
        with torch.no_grad():
            text_inputs = tokenizer(
                [pos_prompt, neg_prompt],
                padding="max_length",
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            )
            out = text_encoder(text_inputs.input_ids)
            embeds = (
                out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
            ).detach().clone().float().cpu()
        del text_encoder, tokenizer, out, text_inputs
        gc.collect()
        if embeds.ndim == 3 and embeds.shape[0] == 2 and embeds.shape[-1] == cross_attn_dim:
            if is_rank0:
                print(
                    f"[PromptAnchor] Precomputed pos/neg CLIP prompts {tuple(embeds[0:1].shape)} on CPU "
                    f"(0 MB text-encoder VRAM)",
                    flush=True,
                )
            return embeds[0:1].contiguous(), embeds[1:2].contiguous()
    except Exception as exc:
        if is_rank0:
            print(f"[PromptAnchor] Fallback zero prompt anchors ({exc})", flush=True)
    zeros = torch.zeros(1, prompt_seq_len, cross_attn_dim, dtype=torch.float32)
    return zeros, zeros.clone()


class _ActionConditionedConvIn(nn.Module):
    """Injects 1x1 spatial `action_map` directly into UNet `conv_in`."""

    def __init__(self, orig_conv_in: nn.Module, num_actions: int, out_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.orig_conv_in = orig_conv_in
        proj_out = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        nn.init.zeros_(proj_out.weight)
        nn.init.zeros_(proj_out.bias)
        self.act_stem = nn.Sequential(
            nn.Conv2d(num_actions, hidden_dim, kernel_size=1),
            nn.SiLU(),
            proj_out,
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.orig_conv_in.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.orig_conv_in.bias

    @property
    def in_channels(self) -> int:
        return self.orig_conv_in.in_channels

    @property
    def out_channels(self) -> int:
        return self.orig_conv_in.out_channels

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules")
            if modules is not None and "orig_conv_in" in modules:
                return getattr(modules["orig_conv_in"], name)
            raise

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        out = self.orig_conv_in(sample)
        action_map = _get_spatial_action_map(out.device)
        if action_map is None:
            return out
        if action_map.device != out.device:
            action_map = action_map.to(out.device)
        target_hw = out.shape[-2:]
        if action_map.shape[-2:] != target_hw:
            a_local = F.interpolate(action_map.float(), size=target_hw, mode="area")
        else:
            a_local = action_map.float()
        delta = self.act_stem(a_local)
        return out + delta.to(out.dtype)


class _ActionModulatedTimeEmbedding(nn.Module):
    """Injects Dual-Branch (Pixel + Semantic) slider FiLM offsets directly into UNet `temb`."""

    def __init__(self, orig_time_emb: nn.Module, action_dim: int, temb_dim: int):
        super().__init__()
        self.orig_time_emb = orig_time_emb
        self.pix_mlp = self._make_zero_mlp(action_dim, temb_dim)
        self.sem_mlp = self._make_zero_mlp(action_dim, temb_dim)

    @staticmethod
    def _make_zero_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
        proj_out = nn.Linear(out_dim, out_dim)
        nn.init.zeros_(proj_out.weight)
        nn.init.zeros_(proj_out.bias)
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            proj_out,
        )

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules")
            if modules is not None and "orig_time_emb" in modules:
                return getattr(modules["orig_time_emb"], name)
            raise

    def forward(self, *args, **kwargs):
        temb = self.orig_time_emb(*args, **kwargs)
        action_cond, lambda_pix, lambda_sem, _spatial = _get_generator_lora_state(temb.device)
        if action_cond is None:
            return temb

        cond_f = action_cond.float()
        if cond_f.device != temb.device:
            cond_f = cond_f.to(temb.device)
        if cond_f.shape[0] != temb.shape[0] and temb.shape[0] % max(1, cond_f.shape[0]) == 0:
            cond_f = cond_f.repeat_interleave(temb.shape[0] // cond_f.shape[0], dim=0)

        delta = temb.new_zeros(temb.shape, dtype=torch.float32)
        if abs(lambda_pix) >= 1e-6:
            delta = delta + float(lambda_pix) * self.pix_mlp(cond_f)
        if abs(lambda_sem) >= 1e-6:
            delta = delta + float(lambda_sem) * self.sem_mlp(cond_f)
        return temb + delta.to(temb.dtype)


class _SpatialModulatedUNetStage(nn.Module):
    """Wraps a UNet block with 1x1 Spatial Action Modulation + optional 2D C-RADIO grid injection."""

    def __init__(
        self,
        orig_block: nn.Module,
        out_channels: int,
        num_actions: int,
        hidden_dim: int = 64,
        sem_dim: Optional[int] = None,
    ):
        super().__init__()
        self.orig_block = orig_block
        self.spatial_mod = PointwiseSpatialModulator(
            num_actions=num_actions,
            channels=out_channels,
            hidden_dim=hidden_dim,
        )
        if sem_dim is not None and sem_dim > 0:
            proj_ch = min(256, max(64, out_channels // 4))
            sem_out_conv = nn.Conv2d(proj_ch, out_channels, kernel_size=1)
            nn.init.zeros_(sem_out_conv.weight)
            nn.init.zeros_(sem_out_conv.bias)
            self.sem_grid_proj = nn.Sequential(
                nn.Conv2d(sem_dim + num_actions, proj_ch, kernel_size=1),
                nn.SiLU(),
                sem_out_conv,
            )
        else:
            self.sem_grid_proj = None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules")
            if modules is not None and "orig_block" in modules:
                return getattr(modules["orig_block"], name)
            raise

    def forward(self, *args, **kwargs):
        out = self.orig_block(*args, **kwargs)
        sample = out[0] if isinstance(out, tuple) else out
        action_map = _get_spatial_action_map(sample.device)
        if action_map is None:
            return out

        if action_map.device != sample.device:
            action_map = action_map.to(sample.device)
        mod_sample = self.spatial_mod(sample, action_map)

        if self.sem_grid_proj is not None:
            sem_grid = _get_spatial_sem_grid(sample.device)
            if sem_grid is not None:
                if sem_grid.device != sample.device:
                    sem_grid = sem_grid.to(sample.device)
                target_hw = mod_sample.shape[-2:]
                g_f = F.normalize(sem_grid.float(), dim=1)
                if g_f.shape[-2:] != target_hw:
                    g_f = F.interpolate(g_f, size=target_hw, mode="bilinear", align_corners=False)
                if action_map.shape[-2:] != target_hw:
                    a_sem = F.interpolate(action_map.float(), size=target_hw, mode="area")
                else:
                    a_sem = action_map.float()
                act_mag = (a_sem.abs().amax(dim=1, keepdim=True) / 0.10).clamp(0.0, 1.0)
                sem_delta = self.sem_grid_proj(torch.cat([g_f, a_sem], dim=1)) * act_mag
                mod_sample = mod_sample + sem_delta.to(mod_sample.dtype)

        return (mod_sample,) + out[1:] if isinstance(out, tuple) else mod_sample


class PointwiseRadiometricHead(nn.Module):
    """Strictly `1x1` (zero spatial receptive field!) Rational-Curve Tone & Warmth Adapter."""

    def __init__(self, num_actions: int, struct_indices: Sequence[int], hidden_dim: int = 32):
        super().__init__()
        self.register_buffer("struct_indices", torch.tensor(list(struct_indices), dtype=torch.long), persistent=False)
        tone_in = 9 + num_actions
        tone_out = nn.Conv2d(hidden_dim, 3, kernel_size=1)
        nn.init.zeros_(tone_out.weight)
        nn.init.zeros_(tone_out.bias)
        self.tone_head = nn.Sequential(
            nn.Conv2d(tone_in, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            tone_out,
        )

    def pixel_stage_parameters(self) -> List[nn.Parameter]:
        return list(self.tone_head.parameters())

    def semantic_stage_parameters(self) -> List[nn.Parameter]:
        return []

    def forward(
        self,
        src_01: torch.Tensor,
        vae_rgb_01: torch.Tensor,
        action_map: torch.Tensor,
        fidelity_scale: float = 1.0,
        detail_scale: float = 0.0,
    ) -> torch.Tensor:
        del detail_scale
        s_f = src_01.float()
        v_f = vae_rgb_01.float()
        a_f = action_map.float()
        if a_f.shape[-2:] != v_f.shape[-2:]:
            a_f = F.interpolate(a_f, size=v_f.shape[-2:], mode="bilinear", align_corners=False)

        if self.struct_indices.numel() > 0:
            s_idx = self.struct_indices.to(a_f.device)
            struct_act = a_f.index_select(1, s_idx).abs().amax(dim=1, keepdim=True)
        else:
            struct_act = a_f.abs().amax(dim=1, keepdim=True)

        struct_blend = (struct_act / 0.08).clamp(0.0, 1.0)
        base = s_f * (1.0 - struct_blend) + v_f * struct_blend

        if abs(fidelity_scale) >= 1e-6:
            quad = base * (1.0 - base)
            sq = base * base
            tone_delta = self.tone_head(torch.cat([base, quad, sq, a_f], dim=1))
            out = (base + float(fidelity_scale) * tone_delta).clamp(0.0, 1.0)
        else:
            out = base

        return out.to(vae_rgb_01.dtype)


class _MockTinyUNetBlock(nn.Module):
    """Lightweight cross-attention + conv block mimicking diffusers UNet2DConditionModel."""

    def __init__(self, in_ch: int, out_ch: int, cross_dim: int, stride: int = 1, upsample: bool = False):
        super().__init__()
        self.has_cross_attention = True
        self.gradient_checkpointing = False
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest") if upsample else None
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(out_ch, out_ch)

        self.norm_attn = nn.LayerNorm(out_ch)
        self.to_q = nn.Linear(out_ch, out_ch)
        self.to_k = nn.Linear(cross_dim, out_ch)
        self.to_v = nn.Linear(cross_dim, out_ch)
        self.to_out = nn.Linear(out_ch, out_ch)

    def _forward_impl(self, x: torch.Tensor, temb: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_proj(F.silu(temb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(h)

        b, c, hh, ww = h.shape
        seq = self.norm_attn(h.flatten(2).transpose(1, 2))
        q = self.to_q(seq)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)
        attn = F.scaled_dot_product_attention(q, k, v)
        seq_out = self.to_out(attn)
        return h + seq_out.transpose(1, 2).reshape(b, c, hh, ww)

    def forward(self, x: torch.Tensor, temb: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return grad_checkpoint(self._forward_impl, x, temb, encoder_hidden_states, use_reentrant=False)
        return self._forward_impl(x, temb, encoder_hidden_states)


class _MockTinyUNet(nn.Module):
    """Offline UNet2DConditionModel-compatible generator for fast local unit tests."""

    class _Cfg:
        cross_attention_dim = 128
        in_channels = 4
        out_channels = 4
        block_out_channels = (64, 128)

    class _Out:
        def __init__(self, sample):
            self.sample = sample

    def __init__(self):
        super().__init__()
        self.config = self._Cfg()
        c0, c1 = self.config.block_out_channels
        cd = self.config.cross_attention_dim

        self.conv_in = nn.Conv2d(4, c0, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(nn.Linear(1, c1), nn.SiLU(), nn.Linear(c1, c1))
        self.t_proj0 = nn.Linear(c1, c0)

        self.down_blocks = nn.ModuleList(
            [
                _MockTinyUNetBlock(c0, c0, cd, stride=1),
                _MockTinyUNetBlock(c0, c1, cd, stride=2),
            ]
        )
        self.mid_block = _MockTinyUNetBlock(c1, c1, cd, stride=1)
        self.up_blocks = nn.ModuleList(
            [
                _MockTinyUNetBlock(c1, c0, cd, upsample=True),
                _MockTinyUNetBlock(c0, c0, cd, stride=1),
            ]
        )
        self.conv_out = nn.Conv2d(c0, 4, kernel_size=3, padding=1)

    def enable_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        for mod in self.modules():
            if isinstance(mod, _MockTinyUNetBlock):
                mod.gradient_checkpointing = True

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
    ):
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.shape[0] == 1 and sample.shape[0] > 1:
            timestep = timestep.expand(sample.shape[0])
        t_norm = (timestep.float() / 1000.0).unsqueeze(-1)
        temb1 = self.time_mlp(t_norm)
        temb0 = self.t_proj0(temb1)

        h = self.conv_in(sample)
        h0 = self.down_blocks[0](h, temb0, encoder_hidden_states)
        h1 = self.down_blocks[1](h0, temb1, encoder_hidden_states)
        hm = self.mid_block(h1, temb1, encoder_hidden_states)
        u0 = self.up_blocks[0](hm + h1, temb0, encoder_hidden_states)
        u1 = self.up_blocks[1](u0 + h0, temb0, encoder_hidden_states)
        out = self.conv_out(F.silu(u1))
        return self._Out(out) if return_dict else (out,)


class ControllableRestorer(nn.Module):
    """V5 Universal Slider-Routed Experts + Universal eVAE + Zero-VRAM Target-Anchored CSD Restorer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        backbone_key = config.GENERATOR_BACKBONE.lower()

        self.semantic_encoder = SemanticEncoder(config)

        alphas_cumprod = _build_default_alphas_cumprod()
        prediction_type = "epsilon"
        repo_id = None
        if backbone_key == "mock_tiny":
            raw_vae = _MockTinyVAE(base_ch=32)
            raw_unet = _MockTinyUNet()
            prediction_type = "sample"
        else:
            from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

            primary_repo = config.GENERATOR_REPO_MAP.get(backbone_key, config.GENERATOR_BACKBONE)
            fallback_map = getattr(config, "GENERATOR_FALLBACK_REPOS", {})
            candidate_repos = list(fallback_map.get(backbone_key, [primary_repo]))
            if primary_repo not in candidate_repos:
                candidate_repos.insert(0, primary_repo)

            is_rank0 = os.environ.get("RANK", "0") == "0"
            raw_vae = None
            raw_unet = None
            last_repo_err = None
            for cand_repo in candidate_repos:
                try:
                    if is_rank0:
                        print(f"[Restorer] Loading VAE & UNet from '{cand_repo}'...", flush=True)
                    try:
                        raw_vae = load_hf_public_safe(
                            AutoencoderKL.from_pretrained,
                            cand_repo,
                            subfolder="vae",
                            variant="fp16",
                            torch_dtype=torch.float32,
                        )
                    except Exception:
                        raw_vae = load_hf_public_safe(AutoencoderKL.from_pretrained, cand_repo, subfolder="vae")

                    try:
                        raw_unet = load_hf_public_safe(
                            UNet2DConditionModel.from_pretrained,
                            cand_repo,
                            subfolder="unet",
                            variant="fp16",
                            torch_dtype=torch.float32,
                        )
                    except Exception:
                        raw_unet = load_hf_public_safe(
                            UNet2DConditionModel.from_pretrained, cand_repo, subfolder="unet"
                        )
                    repo_id = cand_repo
                    if is_rank0:
                        print(f"[Restorer] Loaded generator backbone '{repo_id}'", flush=True)
                    break
                except Exception as repo_err:
                    last_repo_err = repo_err
                    if is_rank0:
                        print(
                            f"[Restorer] Repo '{cand_repo}' unavailable ({repo_err}); trying next mirror...",
                            flush=True,
                        )

            if raw_vae is None or raw_unet is None:
                raise RuntimeError(
                    f"Could not load generator backbone '{backbone_key}' from candidates {candidate_repos}: {last_repo_err}"
                )

            try:
                sched = load_hf_public_safe(DDPMScheduler.from_pretrained, repo_id, subfolder="scheduler")
                if getattr(sched, "alphas_cumprod", None) is not None:
                    alphas_cumprod = sched.alphas_cumprod.detach().clone().float()
                prediction_type = str(getattr(sched.config, "prediction_type", "epsilon")).lower()
            except Exception:
                pass

        self.prediction_type = prediction_type
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)

        deg_names = list(getattr(config, "DEGRADATION_TYPES", []))
        signed_set = set(getattr(config, "SIGNED_ACTION_TYPES", ["exposure", "warmth"]))
        struct_idx = [i for i, n in enumerate(deg_names) if n not in signed_set]
        self.register_buffer("struct_action_indices", torch.tensor(struct_idx, dtype=torch.long), persistent=False)

        # Universal eVAE (Slider-Routed Encoder Experts + Universal Spatial Skip Alignment + Frozen Decoder)
        self.skip_vae = SkipConnectedVAE(raw_vae, config)

        hypir_scale = (
            float(getattr(config, "HYPIR_LORA_RANK_SCALE", 2.0))
            if bool(getattr(config, "HYPIR_MODE", False))
            else 1.0
        )
        unet_expert_specs = resolve_expert_specs(
            getattr(config, "LORA_EXPERT_SPECS", None),
            deg_names,
            signed_types=signed_set,
            default_rank=int(getattr(config, "UNET_LORA_RANK", 28)),
            rank_scale=hypir_scale,
        )
        sem_rank = max(6, int(getattr(config, "SEM_DETAIL_LORA_RANK", 16)))
        sem_alpha = float(getattr(config, "SEM_DETAIL_LORA_ALPHA", float(sem_rank)))

        # Expandable Slider-Routed UIR-LoRA Experts + Strip/Multi-Scale Conv-LoRA + DoRA on UNet
        self.unet = raw_unet
        self.unet_pix_lora_params, self.unet_sem_lora_params = inject_generator_dual_lora(
            self.unet,
            action_dim=config.ACTION_HIDDEN_DIM,
            rank=config.UNET_LORA_RANK,
            alpha=config.UNET_LORA_ALPHA,
            include_conv=True,
            expert_specs=unet_expert_specs,
            sem_rank=sem_rank,
            sem_alpha=sem_alpha,
            use_dora=bool(getattr(config, "USE_DORA", True)),
            unfreeze_norms=bool(getattr(config, "UNFREEZE_UNET_NORMS", False)),
        )

        if getattr(config, "GRADIENT_CHECKPOINTING", True) and hasattr(self.unet, "enable_gradient_checkpointing"):
            try:
                self.unet.enable_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.unet.enable_gradient_checkpointing()

        cross_attn_dim = getattr(self.unet.config, "cross_attention_dim", 768)
        if isinstance(cross_attn_dim, (list, tuple)):
            cross_attn_dim = int(cross_attn_dim[-1])
        else:
            cross_attn_dim = int(cross_attn_dim)

        prompt_seq_len = int(getattr(config, "PROMPT_SEQ_LEN", 77))
        if repo_id is not None:
            pos_prompt = str(
                getattr(
                    config,
                    "DEFAULT_POS_PROMPT",
                    "a high-resolution, sharp, clean, natural photograph, realistic micro-texture, smooth gradients, accurate colors, crisp optical focus",
                )
            )
            neg_prompt = str(
                getattr(
                    config,
                    "DEFAULT_NEG_PROMPT",
                    "blurry, motion blur, out of focus, camera shake, noisy, jpeg artifacts, oversharpened halos, ringing, wavy fake texture, checkerboard, plastic",
                )
            )
            base_prompt_embeds, neg_prompt_embeds = _precompute_openclip_prompt_pair(
                repo_id=repo_id,
                pos_prompt=pos_prompt,
                neg_prompt=neg_prompt,
                cross_attn_dim=cross_attn_dim,
                prompt_seq_len=prompt_seq_len,
            )
            prompt_seq_len = int(base_prompt_embeds.shape[1])
        else:
            base_prompt_embeds = torch.zeros(1, prompt_seq_len, cross_attn_dim, dtype=torch.float32)
            neg_prompt_embeds = torch.zeros(1, prompt_seq_len, cross_attn_dim, dtype=torch.float32)

        self.register_buffer("pos_prompt_embeds", base_prompt_embeds.clone(), persistent=False)
        self.register_buffer("neg_prompt_embeds", neg_prompt_embeds.clone(), persistent=False)

        self.spatial_mod_params: List[nn.Parameter] = []
        self.temb_pix_params: List[nn.Parameter] = []
        self.temb_sem_params: List[nn.Parameter] = []
        self._attach_unet_spatial_overlays(config, cross_attn_dim=cross_attn_dim, prompt_seq_len=prompt_seq_len)

        self.action_overlay = ActionOverlayEncoder(
            num_actions=config.NUM_ACTIONS,
            sem_spatial_dim=self.semantic_encoder.spatial_dim,
            sem_global_dim=self.semantic_encoder.global_dim,
            cross_attn_dim=cross_attn_dim,
            action_hidden_dim=config.ACTION_HIDDEN_DIM,
            prompt_seq_len=prompt_seq_len,
            base_prompt_embeds=base_prompt_embeds,
        )

        tone_hidden = int(getattr(config, "TONE_HEAD_HIDDEN_DIM", 32))
        self.pixel_refiner = PointwiseRadiometricHead(
            num_actions=config.NUM_ACTIONS,
            struct_indices=struct_idx,
            hidden_dim=tone_hidden,
        )

        latent_ch = int(getattr(self.unet.config, "out_channels", 4))
        init_scale = float(getattr(config, "INITIAL_VELOCITY_SCALE", 1.0))
        self.velocity_scale = nn.Parameter(torch.full((1, latent_ch, 1, 1), init_scale))

    def _attach_unet_spatial_overlays(self, config, cross_attn_dim: int, prompt_seq_len: int = 77):
        num_actions = config.NUM_ACTIONS
        mod_hidden = getattr(config, "SPATIAL_MOD_HIDDEN_DIM", 64)
        action_dim = int(getattr(config, "ACTION_HIDDEN_DIM", 128))
        sem_dim = int(self.semantic_encoder.spatial_dim)

        in_ch = int(getattr(self.unet.config, "in_channels", 4))
        unet_dev = infer_module_device(self.unet)
        unet_dtype = infer_module_dtype(self.unet)

        probed_channels: Dict[str, int] = {}
        hooks = []

        def _make_probe_hook(stage_key: str):
            def _hook(_module, _inputs, output):
                sample = output[0] if isinstance(output, tuple) else output
                probed_channels[stage_key] = int(sample.shape[-1] if stage_key == "temb" else sample.shape[1])

            return _hook

        temb_attr = "time_embedding" if hasattr(self.unet, "time_embedding") else (
            "time_mlp" if hasattr(self.unet, "time_mlp") else None
        )
        if temb_attr is not None:
            hooks.append(getattr(self.unet, temb_attr).register_forward_hook(_make_probe_hook("temb")))

        hooks.append(self.unet.conv_in.register_forward_hook(_make_probe_hook("conv_in")))
        for idx, blk in enumerate(self.unet.down_blocks):
            hooks.append(blk.register_forward_hook(_make_probe_hook(f"down_{idx}")))
        hooks.append(self.unet.mid_block.register_forward_hook(_make_probe_hook("mid")))
        for idx, blk in enumerate(self.unet.up_blocks):
            hooks.append(blk.register_forward_hook(_make_probe_hook(f"up_{idx}")))

        try:
            with torch.no_grad():
                dummy_lat = torch.zeros(1, in_ch, 32, 32, device=unet_dev, dtype=unet_dtype)
                dummy_t = torch.tensor(
                    [int(getattr(config, "RESTORATION_MAX_TIMESTEP", 279))], device=unet_dev, dtype=torch.long
                )
                dummy_enc = torch.zeros(1, prompt_seq_len, cross_attn_dim, device=unet_dev, dtype=unet_dtype)
                self.unet(dummy_lat, dummy_t, encoder_hidden_states=dummy_enc)
        finally:
            for handle in hooks:
                handle.remove()

        if "conv_in" in probed_channels:
            wrapped_cin = _ActionConditionedConvIn(
                self.unet.conv_in, num_actions, probed_channels["conv_in"], hidden_dim=mod_hidden
            ).to(device=unet_dev)
            self.unet.conv_in = wrapped_cin
            self.spatial_mod_params.extend(list(wrapped_cin.act_stem.parameters()))

        if temb_attr is not None and "temb" in probed_channels:
            orig_temb = getattr(self.unet, temb_attr)
            wrapped_temb = _ActionModulatedTimeEmbedding(orig_temb, action_dim, probed_channels["temb"]).to(
                device=unet_dev
            )
            setattr(self.unet, temb_attr, wrapped_temb)
            self.temb_pix_params.extend(list(wrapped_temb.pix_mlp.parameters()))
            self.temb_sem_params.extend(list(wrapped_temb.sem_mlp.parameters()))

        for idx in range(len(self.unet.down_blocks)):
            out_ch = probed_channels[f"down_{idx}"]
            wrapped_down = _SpatialModulatedUNetStage(
                self.unet.down_blocks[idx], out_ch, num_actions, mod_hidden, sem_dim=None
            ).to(device=unet_dev)
            self.unet.down_blocks[idx] = wrapped_down
            self.spatial_mod_params.extend(list(wrapped_down.spatial_mod.parameters()))

        mid_ch = probed_channels["mid"]
        wrapped_mid = _SpatialModulatedUNetStage(
            self.unet.mid_block, mid_ch, num_actions, mod_hidden, sem_dim=sem_dim
        ).to(device=unet_dev)
        self.unet.mid_block = wrapped_mid
        self.spatial_mod_params.extend(list(wrapped_mid.spatial_mod.parameters()))
        self.spatial_mod_params.extend(list(wrapped_mid.sem_grid_proj.parameters()))

        num_up = len(self.unet.up_blocks)
        sem_up_idx = min(1, num_up - 1)
        for idx in range(num_up):
            out_ch = probed_channels[f"up_{idx}"]
            use_sem = sem_dim if idx == sem_up_idx else None
            wrapped_up = _SpatialModulatedUNetStage(
                self.unet.up_blocks[idx], out_ch, num_actions, mod_hidden, sem_dim=use_sem
            ).to(device=unet_dev)
            self.unet.up_blocks[idx] = wrapped_up
            self.spatial_mod_params.extend(list(wrapped_up.spatial_mod.parameters()))
            if wrapped_up.sem_grid_proj is not None:
                self.spatial_mod_params.extend(list(wrapped_up.sem_grid_proj.parameters()))

    def get_stage_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        vision_params = list(self.semantic_encoder.vision_lora_params)
        pixel_params = (
            list(self.unet_pix_lora_params)
            + list(self.temb_pix_params)
            + list(self.skip_vae.vae_pix_lora_params)
            + list(self.skip_vae.skip_bridge_parameters())
            + list(self.action_overlay.parameters())
            + list(self.spatial_mod_params)
            + list(self.pixel_refiner.pixel_stage_parameters())
            + [self.velocity_scale]
        )
        semantic_params = (
            list(self.unet_sem_lora_params)
            + list(self.temb_sem_params)
            + list(self.skip_vae.vae_sem_lora_params)
            + list(self.pixel_refiner.semantic_stage_parameters())
        )
        return {
            "vision_lora": vision_params,
            "pixel_stage": pixel_params,
            "semantic_stage": semantic_params,
        }

    def set_training_stage(self, stage: str):
        groups = self.get_stage_parameter_groups()
        for p in self.parameters():
            p.requires_grad = False

        stage = stage.lower()
        if stage == "align_warmup":
            for p in groups["vision_lora"]:
                p.requires_grad = True
        elif stage == "pixel_stage":
            for p in groups["vision_lora"] + groups["pixel_stage"]:
                p.requires_grad = True
        elif stage == "semantic_stage":
            for p in groups["semantic_stage"]:
                p.requires_grad = True
        elif stage == "joint":
            for p in groups["vision_lora"] + groups["pixel_stage"] + groups["semantic_stage"]:
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown training stage: {stage}")

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        groups = self.get_stage_parameter_groups()
        keep_ids = {id(p) for group in groups.values() for p in group}
        out: Dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            if id(param) in keep_ids:
                out[name] = param.detach().cpu()
        return out

    def load_trainable_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = False):
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        groups = self.get_stage_parameter_groups()
        trainable_ids = {id(p) for group in groups.values() for p in group}
        param_map = dict(self.named_parameters())
        buffer_map = dict(self.named_buffers())

        loaded_keys = set()
        unexpected: List[str] = []

        with torch.no_grad():
            for name, tensor in state_dict.items():
                if name in param_map:
                    dst = param_map[name]
                    if tuple(dst.shape) != tuple(tensor.shape):
                        unexpected.append(f"{name} (shape {tuple(tensor.shape)} != {tuple(dst.shape)})")
                        continue
                    dst.copy_(tensor.to(device=dst.device, dtype=dst.dtype))
                    loaded_keys.add(name)
                elif name in buffer_map:
                    dst = buffer_map[name]
                    if tuple(dst.shape) != tuple(tensor.shape):
                        unexpected.append(f"{name} (shape {tuple(tensor.shape)} != {tuple(dst.shape)})")
                        continue
                    dst.copy_(tensor.to(device=dst.device, dtype=dst.dtype))
                    loaded_keys.add(name)
                else:
                    unexpected.append(name)

        missing = [
            name
            for name, param in param_map.items()
            if id(param) in trainable_ids and name not in loaded_keys
        ]
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Error loading controllable adapters (missing={missing}, unexpected={unexpected})"
            )
        return missing, unexpected

    @staticmethod
    def clear_runtime_contexts() -> None:
        clear_generator_lora_state()
        clear_spatial_action_map()

    def _compute_inference_detail_scale(self, action_map: torch.Tensor, detail_scale: float) -> float:
        if abs(detail_scale) < 1e-6 or self.struct_action_indices.numel() == 0:
            return float(detail_scale)
        idx = self.struct_action_indices.to(action_map.device)
        struct_max = float(action_map.float().index_select(1, idx).abs().max().item())
        gate = min(1.0, max(0.0, struct_max / 0.25))
        return float(detail_scale) * gate

    def _resolve_unet_timesteps(self, t_cont: torch.Tensor, action_map: torch.Tensor, bsz: int) -> torch.Tensor:
        """Slider-driven adaptive timestep (`FideDiff` principle without a blind estimator) or fixed HYPIR step."""
        num_sched = int(self.alphas_cumprod.numel())
        if t_cont.ndim == 0:
            t_cont = t_cont.unsqueeze(0)
        if t_cont.shape[0] == 1 and bsz > 1:
            t_cont = t_cont.expand(bsz)

        if bool(getattr(self.config, "HYPIR_MODE", False)):
            fixed_t = int(getattr(self.config, "HYPIR_FIXED_TIMESTEP", 200))
            return torch.full((bsz,), fixed_t, device=action_map.device, dtype=torch.long).clamp(1, num_sched - 1)

        max_t = float(getattr(self.config, "RESTORATION_MAX_TIMESTEP", 279))
        min_t = float(getattr(self.config, "RESTORATION_MIN_TIMESTEP", max_t))
        if bool(getattr(self.config, "ADAPTIVE_TIMESTEP", True)) and self.struct_action_indices.numel() > 0:
            s_idx = self.struct_action_indices.to(action_map.device)
            s_peak = action_map.float().index_select(1, s_idx).abs().amax(dim=(1, 2, 3)).clamp(0.0, 1.0)
            eff_t = min_t + (max_t - min_t) * torch.pow(s_peak, 0.75)
        else:
            eff_t = torch.full((bsz,), max_t, device=action_map.device, dtype=torch.float32)

        return (t_cont.float() * eff_t).round().long().clamp(1, num_sched - 1)

    def _predict_clean_latent(
        self,
        z_t: torch.Tensor,
        t_cont: torch.Tensor,
        cross_tokens: torch.Tensor,
        global_action_cond: torch.Tensor,
        action_map: torch.Tensor,
        sem_grid: Optional[torch.Tensor],
        fidelity_scale: float,
        detail_scale: float,
    ) -> torch.Tensor:
        bsz = z_t.shape[0]
        t_unet = ControllableRestorer._resolve_unet_timesteps(self, t_cont, action_map, bsz)

        abar_table = (
            self.alphas_cumprod
            if self.alphas_cumprod.device == z_t.device
            else self.alphas_cumprod.to(z_t.device)
        )
        alpha_bar = abar_table.index_select(0, t_unet).view(-1, 1, 1, 1).float()
        sqrt_alpha_bar = torch.sqrt(alpha_bar.clamp(min=1e-6))
        sqrt_one_minus = torch.sqrt((1.0 - alpha_bar).clamp(min=0.0))
        sigma_t = sqrt_one_minus / sqrt_alpha_bar

        unet_dtype = infer_module_dtype(self.unet, fallback=z_t.dtype)
        z_scaled = z_t.float() if self.prediction_type == "sample" else (sqrt_alpha_bar * z_t.float())
        z_in = z_scaled.to(unet_dtype)
        if self.training and torch.is_grad_enabled() and not z_in.requires_grad:
            z_in = z_in.detach().requires_grad_(True)

        with generator_lora_context(
            global_action_cond,
            lambda_pix=fidelity_scale,
            lambda_sem=detail_scale,
            device=z_in.device,
            spatial_action_map=action_map,
        ):
            with _spatial_action_context(action_map, sem_grid=sem_grid, device=z_in.device):
                raw_pred = self.unet(
                    z_in,
                    t_unet,
                    encoder_hidden_states=cross_tokens.to(unet_dtype),
                ).sample.float()

        if self.prediction_type == "v_prediction":
            return alpha_bar * z_t.float() - sqrt_one_minus * (self.velocity_scale * raw_pred)
        if self.prediction_type == "sample":
            return z_t.float() + self.velocity_scale * (raw_pred - z_t.float())

        eps_pred = self.velocity_scale * raw_pred
        return z_t.float() - sigma_t * eps_pred

    def compute_target_anchored_csd_loss(
        self,
        hat_z_0: torch.Tensor,
        z_tgt: torch.Tensor,
        action_map: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-extra-VRAM Target-Anchored Classifier Score Distillation (`PiSA-SR` CSD + `TSD-SR` TSD).

        Temporarily disables all LoRAs & spatial overlays (`lambda_pix=0, lambda_sem=0`) on the already-loaded
        frozen base UNet under `torch.no_grad()` to compute a target-tethered + negative-prompt score gradient
        without allocating a second UNet or storing teacher activations!
        """
        if bool(getattr(self.config, "HYPIR_MODE", False)):
            return hat_z_0.new_zeros((), dtype=torch.float32)

        bsz = hat_z_0.shape[0]
        dev = hat_z_0.device
        num_sched = int(self.alphas_cumprod.numel())
        t_low, t_high = getattr(self.config, "CSD_TIMESTEP_RANGE", (40, 320))
        t_low = max(1, min(int(t_low), num_sched - 1))
        t_high = max(t_low + 1, min(int(t_high), num_sched))

        with torch.no_grad():
            t_csd = torch.randint(t_low, t_high, (bsz,), device=dev, dtype=torch.long)
            abar_table = self.alphas_cumprod if self.alphas_cumprod.device == dev else self.alphas_cumprod.to(dev)
            abar = abar_table.index_select(0, t_csd).view(-1, 1, 1, 1).float()
            sqrt_ab = torch.sqrt(abar.clamp(min=1e-6))
            sqrt_om = torch.sqrt((1.0 - abar).clamp(min=1e-6))

            eps = torch.randn_like(hat_z_0.float())
            z_stu_det = hat_z_0.detach().float()
            z_gt_det = z_tgt.detach().float()

            z_t_stu = sqrt_ab * z_stu_det + sqrt_om * eps
            z_t_gt = sqrt_ab * z_gt_det + sqrt_om * eps

            unet_dtype = infer_module_dtype(self.unet, fallback=hat_z_0.dtype)
            pos_emb = self.pos_prompt_embeds.to(device=dev, dtype=unet_dtype).expand(bsz, -1, -1)
            neg_emb = self.neg_prompt_embeds.to(device=dev, dtype=unet_dtype).expand(bsz, -1, -1)

            def _to_eps(raw_out: torch.Tensor, z_noisy: torch.Tensor) -> torch.Tensor:
                if self.prediction_type == "v_prediction":
                    return sqrt_om * z_noisy + sqrt_ab * raw_out
                if self.prediction_type == "sample":
                    return z_noisy - raw_out
                return raw_out

            with generator_lora_context(
                None, lambda_pix=0.0, lambda_sem=0.0, device=dev, spatial_action_map=None
            ), _spatial_action_context(None, sem_grid=None, device=dev):
                cat_z = torch.cat([z_t_stu, z_t_stu], dim=0).to(unet_dtype)
                cat_t = torch.cat([t_csd, t_csd], dim=0)
                cat_emb = torch.cat([pos_emb, neg_emb], dim=0)
                raw_pn = self.unet(cat_z, cat_t, encoder_hidden_states=cat_emb).sample.float()
                raw_pos, raw_neg = raw_pn.chunk(2, dim=0)

                raw_gt = self.unet(
                    z_t_gt.to(unet_dtype), t_csd, encoder_hidden_states=pos_emb
                ).sample.float()

            eps_stu_pos = _to_eps(raw_pos, z_t_stu)
            eps_stu_neg = _to_eps(raw_neg, z_t_stu)
            eps_gt_pos = _to_eps(raw_gt, z_t_gt)

            cfg_scale = float(getattr(self.config, "CSD_CFG_SCALE", 4.5))
            # TSD target anchor `(eps_stu_pos - eps_gt_pos)` + PiSA-SR negative-prompt suppression `(pos - neg)`
            score_dir = (eps_stu_pos - eps_gt_pos) + 0.30 * cfg_scale * (eps_stu_pos - eps_stu_neg)
            norm_scale = score_dir.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp(min=1e-4)
            step_delta = (score_dir / norm_scale) * 0.08
            target_csd = (z_stu_det - step_delta).detach()

            if self.struct_action_indices.numel() > 0:
                s_idx = self.struct_action_indices.to(dev)
                act_gate = action_map.float().index_select(1, s_idx).abs().amax(dim=(1, 2, 3)).clamp(0.0, 1.0)
            else:
                act_gate = action_map.float().abs().amax(dim=(1, 2, 3)).clamp(0.0, 1.0)
            act_gate = act_gate.view(-1, 1, 1, 1)

        return 0.5 * (((hat_z_0.float() - target_csd) ** 2) * act_gate).mean()

    def _apply_zero_action_lock(
        self, src_01: torch.Tensor, pred_01: torch.Tensor, action_map: torch.Tensor
    ) -> torch.Tensor:
        thresh = max(1e-4, float(getattr(self.config, "ZERO_ACTION_LOCK_THRESHOLD", 0.03)))
        max_abs_act = action_map.float().abs().amax(dim=1, keepdim=True)
        gate = (max_abs_act / thresh).clamp(0.0, 1.0)
        return (src_01 + gate * (pred_01 - src_01)).clamp(0.0, 1.0)

    def forward_align_only(self, src_01: torch.Tensor, clean_01: torch.Tensor) -> torch.Tensor:
        align_loss, _ = self.semantic_encoder.compute_self_teacher_loss(src_01, clean_01)
        return align_loss.reshape(1)

    def forward_train(
        self,
        src_01: torch.Tensor,
        tgt_01: torch.Tensor,
        clean_01: torch.Tensor,
        action_input: torch.Tensor,
        fidelity_scale: float = 1.0,
        detail_scale: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Full training pass returning RGB prediction, clean latent `z_tgt`, and universal `evae_loss`."""
        bsz, _, h, w = src_01.shape
        action_map = self.action_overlay.ensure_spatial_map(action_input, h, w)

        vision_trainable = torch.is_grad_enabled() and any(
            p.requires_grad for p in self.semantic_encoder.vision_lora_params
        )
        if vision_trainable:
            align_loss, sem_out = self.semantic_encoder.compute_self_teacher_loss(src_01, clean_01)
        else:
            with torch.no_grad():
                sem_out = self.semantic_encoder(src_01, use_lora=True)
            align_loss = src_01.new_zeros(1, dtype=torch.float32)

        cross_tokens, global_action_cond = self.action_overlay(
            action_map, sem_out["tokens"], sem_out["global"]
        )
        sem_grid = sem_out["grid"]

        # Encode clean/selective target `tgt_01` with frozen base AutoencoderKL (LoRA OFF, 0 autograd VRAM)
        # collecting both `z_tgt` AND clean `tgt_skips` for OSDD-style universal eVAE skip supervision!
        with torch.no_grad(), generator_lora_context(
            None, lambda_pix=0.0, lambda_sem=0.0, device=tgt_01.device, spatial_action_map=None
        ):
            z_tgt, tgt_skips = self.skip_vae.encode_target_latents(tgt_01, return_skips=True)
            z_tgt = z_tgt.detach()

        use_bridge = torch.rand(1, device=src_01.device).item() < getattr(
            self.config, "MULTI_STEP_TRAIN_PROB", 0.0
        )

        set_generator_lora_state(
            global_action_cond,
            lambda_pix=fidelity_scale,
            lambda_sem=detail_scale,
            device=src_01.device,
            spatial_action_map=action_map,
        )
        set_spatial_action_map(action_map, sem_grid=sem_grid, device=src_01.device)

        z_src, enc_skips = self.skip_vae.encode_with_skips(src_01)
        evae_loss = self.skip_vae.compute_evae_alignment_loss(enc_skips, tgt_skips)
        del tgt_skips

        if use_bridge:
            t_vec = torch.empty(bsz, device=src_01.device).uniform_(0.35, 1.0)
            t_4d = t_vec.view(bsz, 1, 1, 1)
            z_t = (1.0 - t_4d) * z_tgt + t_4d * z_src
        else:
            t_vec = torch.ones(bsz, device=src_01.device)
            z_t = z_src

        hat_z_0 = self._predict_clean_latent(
            z_t=z_t,
            t_cont=t_vec,
            cross_tokens=cross_tokens,
            global_action_cond=global_action_cond,
            action_map=action_map,
            sem_grid=sem_grid,
            fidelity_scale=fidelity_scale,
            detail_scale=detail_scale,
        )

        vae_rgb = self.skip_vae.decode_with_skips(hat_z_0, enc_skips, action_map)
        toned_rgb = self.pixel_refiner(
            src_01=src_01,
            vae_rgb_01=vae_rgb,
            action_map=action_map,
            fidelity_scale=fidelity_scale,
            detail_scale=detail_scale,
        )
        pred_rgb = self._apply_zero_action_lock(src_01, toned_rgb, action_map)

        return {
            "pred": pred_rgb,
            "hat_z_0": hat_z_0,
            "z_tgt": z_tgt,
            "align_loss": align_loss.reshape(1),
            "evae_loss": evae_loss.reshape(1),
            "t": t_vec,
        }

    @staticmethod
    def _compute_tile_starts(length: int, tile_size: int, overlap: int) -> List[int]:
        if length <= tile_size:
            return [0]
        stride = max(1, tile_size - overlap)
        n_tiles = 1 + math.ceil((length - tile_size) / stride)
        max_start = length - tile_size
        if n_tiles <= 1:
            return [0]
        return [int(round(i * max_start / (n_tiles - 1))) for i in range(n_tiles)]

    @staticmethod
    def _build_tile_feather_window(
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        full_h: int,
        full_w: int,
        overlap: int,
        device: torch.device,
    ) -> torch.Tensor:
        th = y1 - y0
        tw = x1 - x0
        wy = torch.ones(th, device=device, dtype=torch.float32)
        wx = torch.ones(tw, device=device, dtype=torch.float32)

        ry = min(max(0, overlap), th // 2)
        if ry > 0:
            ramp_y = 0.5 * (
                1.0 - torch.cos(math.pi * (torch.arange(ry, device=device, dtype=torch.float32) + 0.5) / ry)
            )
            if y0 > 0:
                wy[:ry] = ramp_y
            if y1 < full_h:
                wy[-ry:] = torch.minimum(wy[-ry:], ramp_y.flip(0))

        rx = min(max(0, overlap), tw // 2)
        if rx > 0:
            ramp_x = 0.5 * (
                1.0 - torch.cos(math.pi * (torch.arange(rx, device=device, dtype=torch.float32) + 0.5) / rx)
            )
            if x0 > 0:
                wx[:rx] = ramp_x
            if x1 < full_w:
                wx[-rx:] = torch.minimum(wx[-rx:], ramp_x.flip(0))

        return (wy[:, None] * wx[None, :]).clamp_(min=1e-5).unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def _sample_single(
        self,
        src_01: torch.Tensor,
        action_input: torch.Tensor,
        steps: int,
        fidelity_scale: float,
        detail_scale: float,
        return_trajectory: bool = False,
    ):
        bsz, _, h, w = src_01.shape
        action_map = self.action_overlay.ensure_spatial_map(action_input, h, w)

        if float(action_map.abs().max().item()) <= 1e-5:
            if return_trajectory:
                return src_01.clone(), [src_01.clone() for _ in range(max(1, int(steps)) + 1)]
            return src_01.clone()

        eff_detail = self._compute_inference_detail_scale(action_map, detail_scale)

        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        steps = max(1, int(steps))
        trajectory: List[torch.Tensor] = [src_01.clone()] if return_trajectory else []
        cur_rgb = src_01

        for i in range(steps):
            rem_scale = 1.0 if i == 0 else max(0.25, 1.0 - float(i) / float(steps))
            act_step = action_map * rem_scale if rem_scale < 1.0 else action_map

            if pad_h > 0 or pad_w > 0:
                cur_padded = F.pad(cur_rgb, (0, pad_w, 0, pad_h), mode="reflect")
                act_padded = F.pad(act_step, (0, pad_w, 0, pad_h), mode="replicate")
            else:
                cur_padded = cur_rgb
                act_padded = act_step

            sem_out = self.semantic_encoder(cur_padded, use_lora=True)
            cross_tokens, global_action_cond = self.action_overlay(
                act_padded, sem_out["tokens"], sem_out["global"]
            )
            sem_grid = sem_out["grid"]
            del sem_out

            with generator_lora_context(
                global_action_cond,
                lambda_pix=fidelity_scale,
                lambda_sem=eff_detail,
                device=cur_padded.device,
                spatial_action_map=act_padded,
            ):
                z_enc, enc_skips = self.skip_vae.encode_with_skips(cur_padded)
                t_ones = torch.ones(bsz, device=cur_padded.device, dtype=torch.float32)
                hat_z_0 = self._predict_clean_latent(
                    z_t=z_enc,
                    t_cont=t_ones,
                    cross_tokens=cross_tokens,
                    global_action_cond=global_action_cond,
                    action_map=act_padded,
                    sem_grid=sem_grid,
                    fidelity_scale=fidelity_scale,
                    detail_scale=eff_detail,
                )
                vae_rgb = self.skip_vae.decode_with_skips(
                    hat_z_0, enc_skips, act_padded, release_skips=True
                )[:, :, :h, :w]

            del z_enc, hat_z_0, enc_skips, cross_tokens, global_action_cond, sem_grid
            toned_rgb = self.pixel_refiner(
                src_01=cur_rgb,
                vae_rgb_01=vae_rgb,
                action_map=act_step,
                fidelity_scale=fidelity_scale,
                detail_scale=eff_detail,
            )
            step_rgb = self._apply_zero_action_lock(src_01, toned_rgb, action_map)

            if steps == 1:
                cur_rgb = step_rgb
            else:
                blend = 1.0 if i == steps - 1 else min(1.0, (1.0 / float(steps - i)) + 0.35)
                cur_rgb = self._apply_zero_action_lock(
                    src_01, (cur_rgb + blend * (step_rgb - cur_rgb)).clamp(0.0, 1.0), action_map
                )

            if return_trajectory:
                trajectory.append(cur_rgb)

        if return_trajectory:
            return cur_rgb, trajectory
        return cur_rgb

    @torch.no_grad()
    def sample(
        self,
        src_01: torch.Tensor,
        action_input: torch.Tensor,
        steps: Optional[int] = None,
        fidelity_scale: Optional[float] = None,
        detail_scale: Optional[float] = None,
        return_trajectory: bool = False,
        tile_size: Optional[int] = None,
        tile_overlap: Optional[int] = None,
    ):
        steps = int(steps or getattr(self.config, "DEFAULT_INFERENCE_STEPS", 1))
        fidelity_scale = (
            float(self.config.DEFAULT_FIDELITY_SCALE) if fidelity_scale is None else float(fidelity_scale)
        )
        detail_scale = (
            float(self.config.DEFAULT_DETAIL_SCALE) if detail_scale is None else float(detail_scale)
        )
        if tile_size is None:
            tile_size = int(getattr(self.config, "INFERENCE_TILE_SIZE", 512) or 0)
        else:
            tile_size = int(tile_size)
        if tile_overlap is None:
            tile_overlap = int(getattr(self.config, "INFERENCE_TILE_OVERLAP", 96) or 0)
        else:
            tile_overlap = int(tile_overlap)

        _, _, h, w = src_01.shape
        if float(action_input.abs().max().item()) <= 1e-5:
            if return_trajectory:
                return src_01.clone(), [src_01.clone() for _ in range(steps + 1)]
            return src_01.clone()

        if tile_size <= 0 or (h <= tile_size and w <= tile_size):
            return self._sample_single(
                src_01=src_01,
                action_input=action_input,
                steps=steps,
                fidelity_scale=fidelity_scale,
                detail_scale=detail_scale,
                return_trajectory=return_trajectory,
            )

        tile_h = min(h, tile_size)
        tile_w = min(w, tile_size)
        tile_overlap = min(tile_overlap, min(tile_h, tile_w) // 2)
        y_starts = self._compute_tile_starts(h, tile_h, tile_overlap)
        x_starts = self._compute_tile_starts(w, tile_w, tile_overlap)

        full_act_4d = None
        if action_input.ndim == 4:
            full_act_4d = self.action_overlay.ensure_spatial_map(action_input, h, w)

        acc_rgb = torch.zeros_like(src_01, dtype=torch.float32)
        acc_w = torch.zeros((1, 1, h, w), device=src_01.device, dtype=torch.float32)
        acc_traj = (
            [torch.zeros_like(src_01, dtype=torch.float32) for _ in range(steps + 1)]
            if return_trajectory
            else None
        )

        for y0 in y_starts:
            y1 = y0 + tile_h
            for x0 in x_starts:
                x1 = x0 + tile_w
                src_tile = src_01[:, :, y0:y1, x0:x1]
                act_tile = full_act_4d[:, :, y0:y1, x0:x1] if full_act_4d is not None else action_input

                win = self._build_tile_feather_window(
                    y0, y1, x0, x1, h, w, tile_overlap, device=src_01.device
                )
                if return_trajectory:
                    tile_out, tile_traj = self._sample_single(
                        src_01=src_tile,
                        action_input=act_tile,
                        steps=steps,
                        fidelity_scale=fidelity_scale,
                        detail_scale=detail_scale,
                        return_trajectory=True,
                    )
                    acc_rgb[:, :, y0:y1, x0:x1].add_(tile_out.float() * win)
                    for s_idx, frame_t in enumerate(tile_traj):
                        acc_traj[s_idx][:, :, y0:y1, x0:x1].add_(frame_t.float() * win)
                else:
                    tile_out = self._sample_single(
                        src_01=src_tile,
                        action_input=act_tile,
                        steps=steps,
                        fidelity_scale=fidelity_scale,
                        detail_scale=detail_scale,
                        return_trajectory=False,
                    )
                    acc_rgb[:, :, y0:y1, x0:x1].add_(tile_out.float() * win)

                acc_w[:, :, y0:y1, x0:x1].add_(win)

        inv_w = 1.0 / acc_w.clamp(min=1e-6)
        final_rgb = (acc_rgb * inv_w).clamp_(0.0, 1.0).to(src_01.dtype)

        if return_trajectory:
            traj_out = [(frame_acc * inv_w).clamp_(0.0, 1.0).to(src_01.dtype) for frame_acc in acc_traj]
            return final_rgb, traj_out
        return final_rgb

    def forward(
        self,
        src_01: torch.Tensor,
        action_input: Optional[torch.Tensor] = None,
        tgt_01: Optional[torch.Tensor] = None,
        clean_01: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        fidelity_scale: Optional[float] = None,
        detail_scale: Optional[float] = None,
        align_only: bool = False,
    ):
        if align_only:
            if clean_01 is None:
                raise ValueError("clean_01 is required when align_only=True")
            return self.forward_align_only(src_01=src_01, clean_01=clean_01)
        if action_input is None:
            raise ValueError("action_input is required when align_only=False")
        if tgt_01 is not None and clean_01 is not None:
            f_scale = 1.0 if fidelity_scale is None else float(fidelity_scale)
            d_scale = 0.0 if detail_scale is None else float(detail_scale)
            return self.forward_train(
                src_01=src_01,
                tgt_01=tgt_01,
                clean_01=clean_01,
                action_input=action_input,
                fidelity_scale=f_scale,
                detail_scale=d_scale,
            )
        return self.sample(
            src_01=src_01,
            action_input=action_input,
            steps=steps,
            fidelity_scale=fidelity_scale,
            detail_scale=detail_scale,
        )


class _GeneratorParallelStage(nn.Module):
    """Pure PyTorch/diffusers Generator stage for multi-GPU DataParallel."""

    def __init__(self, restorer: ControllableRestorer):
        super().__init__()
        self.config = restorer.config
        self.prediction_type = restorer.prediction_type
        self.register_buffer("alphas_cumprod", restorer.alphas_cumprod.clone(), persistent=False)
        self.register_buffer("struct_action_indices", restorer.struct_action_indices.clone(), persistent=False)
        self.action_overlay = restorer.action_overlay
        self.skip_vae = restorer.skip_vae
        self.unet = restorer.unet
        self.pixel_refiner = restorer.pixel_refiner
        self.velocity_scale = restorer.velocity_scale

    def forward(
        self,
        src_01: torch.Tensor,
        tgt_01: torch.Tensor,
        action_map: torch.Tensor,
        sem_tokens: torch.Tensor,
        sem_global: torch.Tensor,
        sem_grid: torch.Tensor,
        fidelity_scale: float = 1.0,
        detail_scale: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        bsz = src_01.shape[0]
        cross_tokens, global_action_cond = self.action_overlay(action_map, sem_tokens, sem_global)

        with torch.no_grad(), generator_lora_context(
            None, lambda_pix=0.0, lambda_sem=0.0, device=tgt_01.device, spatial_action_map=None
        ):
            z_tgt, tgt_skips = self.skip_vae.encode_target_latents(tgt_01, return_skips=True)
            z_tgt = z_tgt.detach()

        use_bridge = torch.rand(1, device=src_01.device).item() < getattr(
            self.config, "MULTI_STEP_TRAIN_PROB", 0.0
        )

        set_generator_lora_state(
            global_action_cond,
            lambda_pix=fidelity_scale,
            lambda_sem=detail_scale,
            device=src_01.device,
            spatial_action_map=action_map,
        )
        set_spatial_action_map(action_map, sem_grid=sem_grid, device=src_01.device)

        z_src, enc_skips = self.skip_vae.encode_with_skips(src_01)
        evae_loss = self.skip_vae.compute_evae_alignment_loss(enc_skips, tgt_skips)
        del tgt_skips

        if use_bridge:
            t_vec = torch.empty(bsz, device=src_01.device).uniform_(0.35, 1.0)
            t_4d = t_vec.view(bsz, 1, 1, 1)
            z_t = (1.0 - t_4d) * z_tgt + t_4d * z_src
        else:
            t_vec = torch.ones(bsz, device=src_01.device)
            z_t = z_src

        hat_z_0 = ControllableRestorer._predict_clean_latent(
            self,
            z_t=z_t,
            t_cont=t_vec,
            cross_tokens=cross_tokens,
            global_action_cond=global_action_cond,
            action_map=action_map,
            sem_grid=sem_grid,
            fidelity_scale=fidelity_scale,
            detail_scale=detail_scale,
        )

        vae_rgb = self.skip_vae.decode_with_skips(hat_z_0, enc_skips, action_map)
        toned_rgb = self.pixel_refiner(
            src_01=src_01,
            vae_rgb_01=vae_rgb,
            action_map=action_map,
            fidelity_scale=fidelity_scale,
            detail_scale=detail_scale,
        )
        thresh = max(1e-4, float(getattr(self.config, "ZERO_ACTION_LOCK_THRESHOLD", 0.03)))
        gate = (action_map.float().abs().amax(dim=1, keepdim=True) / thresh).clamp(0.0, 1.0)
        pred_rgb = (src_01 + gate * (toned_rgb - src_01)).clamp(0.0, 1.0)
        return {
            "pred": pred_rgb,
            "hat_z_0": hat_z_0,
            "z_tgt": z_tgt,
            "evae_loss": evae_loss.reshape(1),
            "t": t_vec,
        }


class SplitMultiGPURestorer(nn.Module):
    """Runs remote-code Vision Encoder on primary GPU and parallelizes Generator across all GPUs."""

    def __init__(self, raw_model: ControllableRestorer, device_ids: List[int]):
        super().__init__()
        self.__dict__["raw_model"] = raw_model
        gen_stage = _GeneratorParallelStage(raw_model)
        self.dp_generator = nn.DataParallel(gen_stage, device_ids=device_ids)

    def train(self, mode: bool = True):
        super().train(mode)
        self.__dict__["raw_model"].train(mode)
        return self

    def forward(
        self,
        src_01: torch.Tensor,
        action_input: Optional[torch.Tensor] = None,
        tgt_01: Optional[torch.Tensor] = None,
        clean_01: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        fidelity_scale: Optional[float] = None,
        detail_scale: Optional[float] = None,
        align_only: bool = False,
    ):
        raw_model: ControllableRestorer = self.__dict__["raw_model"]
        if align_only:
            return raw_model.forward_align_only(src_01=src_01, clean_01=clean_01)
        if action_input is None:
            raise ValueError("action_input is required when align_only=False")
        if tgt_01 is not None and clean_01 is not None:
            _, _, h, w = src_01.shape
            action_map = raw_model.action_overlay.ensure_spatial_map(action_input, h, w)
            align_loss, sem_out = raw_model.semantic_encoder.compute_self_teacher_loss(src_01, clean_01)
            f_scale = 1.0 if fidelity_scale is None else float(fidelity_scale)
            d_scale = 0.0 if detail_scale is None else float(detail_scale)
            gen_out = self.dp_generator(
                src_01,
                tgt_01,
                action_map,
                sem_out["tokens"],
                sem_out["global"],
                sem_out["grid"],
                fidelity_scale=f_scale,
                detail_scale=d_scale,
            )
            return {
                "pred": gen_out["pred"],
                "hat_z_0": gen_out["hat_z_0"],
                "z_tgt": gen_out["z_tgt"],
                "align_loss": align_loss.reshape(1),
                "evae_loss": gen_out["evae_loss"].mean().reshape(1),
                "t": gen_out["t"],
            }
        return raw_model.sample(
            src_01=src_01,
            action_input=action_input,
            steps=steps,
            fidelity_scale=fidelity_scale,
            detail_scale=detail_scale,
        )
