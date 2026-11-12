"""End-to-End Controllable Pretrained Restoration Model.

Combines:
1. Pretrained Semantic Vision Encoder (C-RADIOv2.5-B or DINOv3+SigLIP2) + Self-Teacher LoRA
2. Skip-Connected VAE with 1x1 Spatial Action Bridges
3. Pretrained Diffusion/Flow UNet (sd-turbo / tiny-sd) with:
   - Action-Modulated Dual-LoRA (Pixel-Faithful LoRA + Semantic-Detail LoRA)
   - DataParallel-safe 1x1 Spatial Action Overlay modulation across multi-scale blocks
   - 1-to-4 step Latent Rectified Flow sampling + Zero-Action pixel preservation lock
"""

from contextlib import contextmanager
import gc
import math
import os
from typing import Dict, List, Optional, Tuple

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
    set_generator_lora_state,
)
from .semantic_encoder import SemanticEncoder, load_hf_public_safe
from .skip_vae import SkipConnectedVAE, _MockTinyVAE


_DEFAULT_SPATIAL_ACTION_MAP: Optional[torch.Tensor] = None
_SPATIAL_MAP_BY_DEVICE: Dict[str, Optional[torch.Tensor]] = {}


def set_spatial_action_map(
    action_map: Optional[torch.Tensor],
    device: Optional[torch.device] = None,
) -> None:
    """Persist device-scoped spatial action map across forward + CUDA autograd backward recomputation.

    NOTE: Do NOT use `threading.local()` here — PyTorch's C++ CUDA autograd engine executes
    non-reentrant checkpoint `unpack_hook` recomputation on a dedicated per-GPU worker thread
    distinct from the Python main thread that ran `forward()`.
    """
    global _DEFAULT_SPATIAL_ACTION_MAP
    key = _resolve_device_key(device=device, tensor=action_map)
    if key is not None:
        _SPATIAL_MAP_BY_DEVICE[key] = action_map
    else:
        _DEFAULT_SPATIAL_ACTION_MAP = action_map


def _get_spatial_action_map(device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    if device is not None:
        key = _resolve_device_key(device=device)
        if key is not None and key in _SPATIAL_MAP_BY_DEVICE:
            return _SPATIAL_MAP_BY_DEVICE[key]
    return _DEFAULT_SPATIAL_ACTION_MAP


def clear_spatial_action_map(device: Optional[torch.device] = None) -> None:
    """Release device-scoped spatial action map tensor references after backward completes."""
    global _DEFAULT_SPATIAL_ACTION_MAP
    if device is not None:
        key = _resolve_device_key(device=device)
        if key is not None:
            _SPATIAL_MAP_BY_DEVICE.pop(key, None)
            return
    _SPATIAL_MAP_BY_DEVICE.clear()
    _DEFAULT_SPATIAL_ACTION_MAP = None


@contextmanager
def _spatial_action_context(
    action_map: Optional[torch.Tensor],
    device: Optional[torch.device] = None,
):
    global _DEFAULT_SPATIAL_ACTION_MAP
    key = _resolve_device_key(device=device, tensor=action_map)
    if key is not None:
        had_key = key in _SPATIAL_MAP_BY_DEVICE
        prev_val = _SPATIAL_MAP_BY_DEVICE.get(key)
        _SPATIAL_MAP_BY_DEVICE[key] = action_map
        try:
            yield
        finally:
            if had_key:
                _SPATIAL_MAP_BY_DEVICE[key] = prev_val
            else:
                _SPATIAL_MAP_BY_DEVICE.pop(key, None)
    else:
        prev_default = _DEFAULT_SPATIAL_ACTION_MAP
        prev_map = dict(_SPATIAL_MAP_BY_DEVICE)
        _SPATIAL_MAP_BY_DEVICE.clear()
        _DEFAULT_SPATIAL_ACTION_MAP = action_map
        try:
            yield
        finally:
            _SPATIAL_MAP_BY_DEVICE.clear()
            _SPATIAL_MAP_BY_DEVICE.update(prev_map)
            _DEFAULT_SPATIAL_ACTION_MAP = prev_default


def _build_default_alphas_cumprod(
    num_train_timesteps: int = 1000,
    beta_start: float = 0.00085,
    beta_end: float = 0.012,
) -> torch.Tensor:
    """Exact SD 1.5 / SD 2.1 / sd-turbo `scaled_linear` cumulative alpha schedule."""
    betas = torch.linspace(
        math.sqrt(beta_start), math.sqrt(beta_end), num_train_timesteps, dtype=torch.float64
    ) ** 2
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0).float()


def _precompute_openclip_prompt_embeds(
    repo_id: str,
    prompt: str,
    cross_attn_dim: int,
    prompt_seq_len: int = 77,
) -> torch.Tensor:
    """Precompute native CLIP/OpenCLIP prompt tokens once on CPU (0 MB runtime text-encoder GPU VRAM)."""
    is_rank0 = os.environ.get("RANK", "0") == "0"
    try:
        from transformers import AutoTokenizer, CLIPTextModel

        tokenizer = load_hf_public_safe(AutoTokenizer.from_pretrained, repo_id, subfolder="tokenizer")
        text_encoder = load_hf_public_safe(CLIPTextModel.from_pretrained, repo_id, subfolder="text_encoder")
        text_encoder.eval()
        max_len = int(getattr(tokenizer, "model_max_length", prompt_seq_len) or prompt_seq_len)
        if max_len <= 0 or max_len > 256:
            max_len = prompt_seq_len
        with torch.no_grad():
            text_inputs = tokenizer(
                [prompt],
                padding="max_length",
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            )
            out = text_encoder(text_inputs.input_ids)
            embeds = (out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state).detach().clone().float().cpu()
        del text_encoder, tokenizer, out, text_inputs
        gc.collect()
        if embeds.ndim == 3 and embeds.shape[-1] == cross_attn_dim:
            if is_rank0:
                print(
                    f"[PromptAnchor] Precomputed native CLIP prompt {tuple(embeds.shape)} on CPU (0 MB text-encoder VRAM)",
                    flush=True,
                )
            return embeds
    except Exception as exc:
        if is_rank0:
            print(f"[PromptAnchor] Fallback zero prompt anchor ({exc})", flush=True)
    return torch.zeros(1, prompt_seq_len, cross_attn_dim, dtype=torch.float32)


class _ActionModulatedTimeEmbedding(nn.Module):
    """Injects Dual-Branch (Pixel + Semantic) slider FiLM offsets directly into UNet `temb`.

    Gives the 8 continuous sliders direct per-channel scale/shift leverage over all ~22 ResNet
    blocks in `sd-turbo` (critical for global color-cast neutralization, exposure recovery & debanding).
    """

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
        action_cond, lambda_pix, lambda_sem = _get_generator_lora_state(temb.device)
        if action_cond is None:
            return temb

        cond_f = action_cond.float()
        if cond_f.device != temb.device:
            cond_f = cond_f.to(temb.device)
        if cond_f.shape[0] != temb.shape[0] and temb.shape[0] % max(1, cond_f.shape[0]) == 0:
            cond_f = cond_f.repeat_interleave(temb.shape[0] // cond_f.shape[0], dim=0)

        delta = temb.new_zeros(temb.shape, dtype=torch.float32)
        if lambda_pix != 0.0:
            delta = delta + float(lambda_pix) * self.pix_mlp(cond_f)
        if lambda_sem != 0.0:
            delta = delta + float(lambda_sem) * self.sem_mlp(cond_f)
        return temb + delta.to(temb.dtype)


class _SpatialModulatedUNetStage(nn.Module):
    """Wraps a UNet block with a DataParallel-replicated 1x1 PointwiseSpatialModulator."""

    def __init__(self, orig_block: nn.Module, out_channels: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.orig_block = orig_block
        self.spatial_mod = PointwiseSpatialModulator(
            num_actions=num_actions,
            channels=out_channels,
            hidden_dim=hidden_dim,
        )

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
        return (mod_sample,) + out[1:] if isinstance(out, tuple) else mod_sample


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

        # Cross-attention linear names match diffusers ("to_q", "to_k", "to_v", "to_out")
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

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor, return_dict: bool = True):
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
    """Action-Conditioned Pretrained Restoration Model with Dual-LoRA & Spatial Overlay."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        backbone_key = config.GENERATOR_BACKBONE.lower()

        # 1. Pretrained Vision Semantic Encoder + Self-Teacher LoRA
        self.semantic_encoder = SemanticEncoder(config)

        # 2. Load Pretrained AutoencoderKL, UNet2DConditionModel & native DDPM 1-step schedule
        alphas_cumprod = _build_default_alphas_cumprod()
        prediction_type = "epsilon"
        repo_id = None
        if backbone_key == "mock_tiny":
            raw_vae = _MockTinyVAE(base_ch=32)
            raw_unet = _MockTinyUNet()
            prediction_type = "sample"  # Keep random-init unit test mock well-scaled without t=999 1/sqrt(alpha_bar)
        else:
            from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

            repo_id = config.GENERATOR_REPO_MAP.get(backbone_key, config.GENERATOR_BACKBONE)
            raw_vae = load_hf_public_safe(AutoencoderKL.from_pretrained, repo_id, subfolder="vae")
            raw_unet = load_hf_public_safe(UNet2DConditionModel.from_pretrained, repo_id, subfolder="unet")
            try:
                sched = load_hf_public_safe(DDPMScheduler.from_pretrained, repo_id, subfolder="scheduler")
                if getattr(sched, "alphas_cumprod", None) is not None:
                    alphas_cumprod = sched.alphas_cumprod.detach().clone().float()
                prediction_type = str(getattr(sched.config, "prediction_type", "epsilon")).lower()
            except Exception:
                pass

        self.prediction_type = prediction_type
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)

        # 3. Wrap VAE with Action-Suppressed Skip Bridges + Encoder & Decoder Dual-LoRA
        self.skip_vae = SkipConnectedVAE(raw_vae, config)

        # 4. Inject Action-Modulated Dual-LoRA (Pixel-LoRA + Semantic-LoRA, incl. conv_in/conv_out) into UNet
        self.unet = raw_unet
        self.unet_pix_lora_params, self.unet_sem_lora_params = inject_generator_dual_lora(
            self.unet,
            action_dim=config.ACTION_HIDDEN_DIM,
            rank=config.UNET_LORA_RANK,
            alpha=config.UNET_LORA_ALPHA,
            include_conv=True,
        )

        # Enable non-reentrant gradient checkpointing on UNet blocks before wrapping spatial overlays
        if getattr(config, "GRADIENT_CHECKPOINTING", True) and hasattr(self.unet, "enable_gradient_checkpointing"):
            try:
                self.unet.enable_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.unet.enable_gradient_checkpointing()

        # 5. Wrap UNet multi-scale blocks with 1x1 Spatial Modulators & hook direct slider FiLM into `temb`
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
                    "a high-resolution, sharp, clean, natural photograph, detailed texture, smooth gradients, accurate colors, crisp focus",
                )
            )
            base_prompt_embeds = _precompute_openclip_prompt_embeds(
                repo_id=repo_id,
                prompt=pos_prompt,
                cross_attn_dim=cross_attn_dim,
                prompt_seq_len=prompt_seq_len,
            )
            prompt_seq_len = int(base_prompt_embeds.shape[1])
        else:
            base_prompt_embeds = torch.zeros(1, prompt_seq_len, cross_attn_dim, dtype=torch.float32)

        self.spatial_mod_params: List[nn.Parameter] = []
        self.temb_pix_params: List[nn.Parameter] = []
        self.temb_sem_params: List[nn.Parameter] = []
        self._attach_unet_spatial_overlays(config, cross_attn_dim=cross_attn_dim, prompt_seq_len=prompt_seq_len)

        # 6. Action & Semantic Token Encoder anchored on SD-Turbo's 77-token OpenCLIP prompt embedding
        self.action_overlay = ActionOverlayEncoder(
            num_actions=config.NUM_ACTIONS,
            sem_spatial_dim=self.semantic_encoder.spatial_dim,
            sem_global_dim=self.semantic_encoder.global_dim,
            cross_attn_dim=cross_attn_dim,
            action_hidden_dim=config.ACTION_HIDDEN_DIM,
            prompt_seq_len=prompt_seq_len,
            base_prompt_embeds=base_prompt_embeds,
        )

        # 7. Native Euler/DDPM Step Scale + Zero-Init Residual Head
        latent_ch = int(getattr(self.unet.config, "out_channels", 4))
        init_scale = float(getattr(config, "INITIAL_VELOCITY_SCALE", 1.0))
        self.velocity_scale = nn.Parameter(torch.full((1, latent_ch, 1, 1), init_scale))
        self.velocity_zero_conv = nn.Conv2d(latent_ch, latent_ch, kernel_size=1)
        nn.init.zeros_(self.velocity_zero_conv.weight)
        nn.init.zeros_(self.velocity_zero_conv.bias)

    def _attach_unet_spatial_overlays(self, config, cross_attn_dim: int, prompt_seq_len: int = 77):
        """Probe exact runtime UNet block & `temb` dimensions and wrap with action modulators."""
        num_actions = config.NUM_ACTIONS
        mod_hidden = getattr(config, "SPATIAL_MOD_HIDDEN_DIM", 64)
        action_dim = int(getattr(config, "ACTION_HIDDEN_DIM", 128))

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

        hooks.append(self.unet.mid_block.register_forward_hook(_make_probe_hook("mid")))
        for idx, blk in enumerate(self.unet.up_blocks):
            hooks.append(blk.register_forward_hook(_make_probe_hook(f"up_{idx}")))

        try:
            with torch.no_grad():
                dummy_lat = torch.zeros(1, in_ch, 32, 32, device=unet_dev, dtype=unet_dtype)
                dummy_t = torch.tensor([int(getattr(config, "RESTORATION_MAX_TIMESTEP", 999))], device=unet_dev, dtype=torch.long)
                dummy_enc = torch.zeros(1, prompt_seq_len, cross_attn_dim, device=unet_dev, dtype=unet_dtype)
                self.unet(dummy_lat, dummy_t, encoder_hidden_states=dummy_enc)
        finally:
            for handle in hooks:
                handle.remove()

        if temb_attr is not None and "temb" in probed_channels:
            orig_temb = getattr(self.unet, temb_attr)
            wrapped_temb = _ActionModulatedTimeEmbedding(orig_temb, action_dim, probed_channels["temb"]).to(
                device=unet_dev
            )
            setattr(self.unet, temb_attr, wrapped_temb)
            self.temb_pix_params.extend(list(wrapped_temb.pix_mlp.parameters()))
            self.temb_sem_params.extend(list(wrapped_temb.sem_mlp.parameters()))

        mid_ch = probed_channels["mid"]
        wrapped_mid = _SpatialModulatedUNetStage(self.unet.mid_block, mid_ch, num_actions, mod_hidden).to(
            device=unet_dev
        )
        self.unet.mid_block = wrapped_mid
        self.spatial_mod_params.extend(list(wrapped_mid.spatial_mod.parameters()))

        for idx in range(len(self.unet.up_blocks)):
            out_ch = probed_channels[f"up_{idx}"]
            wrapped_up = _SpatialModulatedUNetStage(self.unet.up_blocks[idx], out_ch, num_actions, mod_hidden).to(
                device=unet_dev
            )
            self.unet.up_blocks[idx] = wrapped_up
            self.spatial_mod_params.extend(list(wrapped_up.spatial_mod.parameters()))

    def get_stage_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Return cleanly separated parameter groups for 3-phase staged or joint training."""
        vision_params = list(self.semantic_encoder.vision_lora_params)
        pixel_params = (
            list(self.unet_pix_lora_params)
            + list(self.temb_pix_params)
            + list(self.skip_vae.vae_pix_lora_params)
            + list(self.skip_vae.skip_bridge_parameters())
            + list(self.action_overlay.parameters())
            + list(self.spatial_mod_params)
            + [self.velocity_scale]
            + list(self.velocity_zero_conv.parameters())
        )
        semantic_params = (
            list(self.unet_sem_lora_params)
            + list(self.temb_sem_params)
            + list(self.skip_vae.vae_sem_lora_params)
        )
        return {
            "vision_lora": vision_params,
            "pixel_stage": pixel_params,
            "semantic_stage": semantic_params,
        }

    def set_training_stage(self, stage: str):
        """Configure parameter `requires_grad` flags for 'align_warmup', 'pixel_stage', 'semantic_stage', or 'joint'."""
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
        """Export only the lightweight adapter/LoRA/skip/overlay weights (~40-60 MB instead of ~4 GB)."""
        groups = self.get_stage_parameter_groups()
        keep_ids = {
            id(p)
            for group in groups.values()
            for p in group
        }
        out: Dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            if id(param) in keep_ids:
                out[name] = param.detach().cpu()
        return out

    def load_trainable_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = False):
        """Load lightweight adapter/LoRA/skip/overlay weights directly (bypasses frozen HF custom _load_from_state_dict hooks)."""
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
        """Clear device-scoped LoRA & spatial overlay references after backward() finishes."""
        clear_generator_lora_state()
        clear_spatial_action_map()

    def _predict_clean_latent(
        self,
        z_t: torch.Tensor,
        t_cont: torch.Tensor,
        cross_tokens: torch.Tensor,
        global_action_cond: torch.Tensor,
        action_map: torch.Tensor,
        fidelity_scale: float,
        detail_scale: float,
    ) -> torch.Tensor:
        """Predict clean target latent `hat_z_0` via Pix2Pix-Turbo / S3Diff 1-step DDPM formulation."""
        if t_cont.ndim == 0:
            t_cont = t_cont.unsqueeze(0)
        if t_cont.shape[0] == 1 and z_t.shape[0] > 1:
            t_cont = t_cont.expand(z_t.shape[0])

        max_t = int(getattr(self.config, "RESTORATION_MAX_TIMESTEP", 999))
        num_sched = int(self.alphas_cumprod.numel())
        t_unet = (t_cont.float() * float(max_t)).round().long().clamp(1, num_sched - 1)

        abar_table = (
            self.alphas_cumprod
            if self.alphas_cumprod.device == z_t.device
            else self.alphas_cumprod.to(z_t.device)
        )
        alpha_bar = abar_table.index_select(0, t_unet).view(-1, 1, 1, 1).float()
        sqrt_alpha_bar = torch.sqrt(alpha_bar.clamp(min=1e-6))        # = 0.068265 at t=999
        sqrt_one_minus = torch.sqrt((1.0 - alpha_bar).clamp(min=0.0))  # = 0.997667 at t=999

        unet_dtype = infer_module_dtype(self.unet, fallback=z_t.dtype)
        # Pix2Pix-Turbo / S3Diff: feed unit-variance encoded latent `z_t` UN-ATTENUATED at t=999!
        # (`DDPMScheduler.scale_model_input` is identity; z_src already has variance ~1.0 matching N(0, I))
        z_in = z_t.to(unet_dtype)
        if self.training and torch.is_grad_enabled() and not z_in.requires_grad:
            z_in = z_in.detach().requires_grad_(True)

        with generator_lora_context(
            global_action_cond,
            lambda_pix=fidelity_scale,
            lambda_sem=detail_scale,
            device=z_in.device,
        ):
            with _spatial_action_context(action_map, device=z_in.device):
                raw_pred = self.unet(
                    z_in,
                    t_unet,
                    encoder_hidden_states=cross_tokens.to(unet_dtype),
                ).sample.float()

        t_4d = t_cont.float().view(-1, 1, 1, 1)
        zero_res = t_4d * self.velocity_zero_conv(z_t.float()).float()
        if self.prediction_type == "v_prediction":
            return sqrt_alpha_bar * z_t.float() - sqrt_one_minus * (self.velocity_scale * raw_pred) - zero_res
        if self.prediction_type == "sample":
            return z_t.float() + self.velocity_scale * (raw_pred - z_t.float()) - zero_res

        # Default ("epsilon", e.g. sd-turbo / tiny-sd / bk-sdm-tiny):
        # Exact FP32 `DDPMScheduler.step` 1-step jump (t=999 -> prev_t=-1):
        # hat_z_0 = (z_enc - sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)
        eps_pred = self.velocity_scale * raw_pred
        return (z_t.float() - sqrt_one_minus * eps_pred) / sqrt_alpha_bar - zero_res

    def _apply_zero_action_lock(
        self, src_01: torch.Tensor, pred_01: torch.Tensor, action_map: torch.Tensor
    ) -> torch.Tensor:
        """Guarantee exact pixel preservation wherever all 8 sliders at (y, x) equal 0."""
        thresh = max(1e-4, float(getattr(self.config, "ZERO_ACTION_LOCK_THRESHOLD", 0.03)))
        max_act = action_map.float().amax(dim=1, keepdim=True)  # [B, 1, H, W]
        gate = (max_act / thresh).clamp(0.0, 1.0)
        return (src_01 + gate * (pred_01 - src_01)).clamp(0.0, 1.0)

    def forward_align_only(self, src_01: torch.Tensor, clean_01: torch.Tensor) -> torch.Tensor:
        """Fast Phase-1 forward pass updating only the Vision Encoder Self-Teacher LoRA."""
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
        """Full training pass supporting 1-step direct & intermediate Rectified Flow states."""
        bsz, _, h, w = src_01.shape
        action_map = self.action_overlay.ensure_spatial_map(action_input, h, w)

        # 1. Self-Teacher Vision Semantics + Alignment Loss (skip teacher/grad pass when vision_lora is frozen)
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

        # 2. Sample optional Rectified Flow bridge target (disabled by default: MULTI_STEP_TRAIN_PROB = 0.0)
        use_bridge = torch.rand(1, device=src_01.device).item() < getattr(
            self.config, "MULTI_STEP_TRAIN_PROB", 0.0
        )
        z_tgt = None
        if use_bridge:
            with torch.no_grad(), generator_lora_context(
                None, lambda_pix=0.0, lambda_sem=0.0, device=tgt_01.device
            ):
                z_tgt = self.skip_vae.encode_target_latents(tgt_01)

        # 3. Persist device-scoped Dual-LoRA & Spatial Overlay state across forward + backward
        # so gradient-checkpointed UNet and Skip-VAE blocks recompute with identical conditioning
        # even when PyTorch executes CUDA backward hooks on a separate C++ autograd worker thread.
        set_generator_lora_state(
            global_action_cond,
            lambda_pix=fidelity_scale,
            lambda_sem=detail_scale,
            device=src_01.device,
        )
        set_spatial_action_map(action_map, device=src_01.device)

        z_src, enc_skips = self.skip_vae.encode_with_skips(src_01)

        if use_bridge and z_tgt is not None:
            t_vec = torch.empty(bsz, device=src_01.device).uniform_(0.35, 1.0)
            t_4d = t_vec.view(bsz, 1, 1, 1)
            z_t = (1.0 - t_4d) * z_tgt + t_4d * z_src
        else:
            t_vec = torch.ones(bsz, device=src_01.device)
            z_t = z_src

        # 4. Predict clean/target latent endpoint hat_z_0 via native Euler/DDPM preconditioning
        hat_z_0 = self._predict_clean_latent(
            z_t=z_t,
            t_cont=t_vec,
            cross_tokens=cross_tokens,
            global_action_cond=global_action_cond,
            action_map=action_map,
            fidelity_scale=fidelity_scale,
            detail_scale=detail_scale,
        )

        # 5. Decode back to RGB via Action-Gated Skip-VAE & apply Zero-Action Lock
        raw_rgb = self.skip_vae.decode_with_skips(hat_z_0, enc_skips, action_map)
        pred_rgb = self._apply_zero_action_lock(src_01, raw_rgb, action_map)

        return {
            "pred": pred_rgb,
            "align_loss": align_loss.reshape(1),
            "t": t_vec,
        }

    @staticmethod
    def _compute_tile_starts(length: int, tile_size: int, overlap: int) -> List[int]:
        """Compute evenly spaced tile start coordinates covering [0, length) with >= `overlap` margin."""
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
        """Build [1, 1, th, tw] C1-smooth raised-cosine feather window (flat 1.0 on outer image borders)."""
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
        """Run 1-to-N step Rectified Flow restoration on a single patch/tile."""
        bsz, _, h, w = src_01.shape
        action_map = self.action_overlay.ensure_spatial_map(action_input, h, w)

        # Fast shortcut if the entire action overlay is zero everywhere
        if float(action_map.max().item()) <= 1e-5:
            if return_trajectory:
                return src_01.clone(), [src_01.clone() for _ in range(steps + 1)]
            return src_01.clone()

        # Pad non-8-divisible resolutions cleanly for the 8x latent UNet, then crop back
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
            del sem_out

            # Keep generator_lora_context active across encode + UNet + decode so BOTH VAE Encoder & Decoder LoRAs fire!
            with generator_lora_context(
                global_action_cond,
                lambda_pix=fidelity_scale,
                lambda_sem=detail_scale,
                device=cur_padded.device,
            ):
                z_enc, enc_skips = self.skip_vae.encode_with_skips(cur_padded)
                t_ones = torch.ones(bsz, device=cur_padded.device, dtype=torch.float32)
                hat_z_0 = self._predict_clean_latent(
                    z_t=z_enc,
                    t_cont=t_ones,
                    cross_tokens=cross_tokens,
                    global_action_cond=global_action_cond,
                    action_map=act_padded,
                    fidelity_scale=fidelity_scale,
                    detail_scale=detail_scale,
                )
                raw_rgb = self.skip_vae.decode_with_skips(
                    hat_z_0, enc_skips, act_padded, release_skips=True
                )[:, :, :h, :w]

            del z_enc, hat_z_0, enc_skips, cross_tokens, global_action_cond
            step_rgb = self._apply_zero_action_lock(src_01, raw_rgb, action_map)

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
        """Controllable 1-to-N step Rectified Flow restoration (auto-tiled for 2K/4K images)."""
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
        if float(action_input.max().item()) <= 1e-5:
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

        # Seamless raised-cosine overlapping tile restoration for high-res (1080p / 2K / 4K) images
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
        """Unified DataParallel-friendly forward pass for Stage-1 align, Stage-2/3 train, and inference."""
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
    """Pure PyTorch/diffusers Generator stage (~95% of FLOPs) for multi-GPU DataParallel."""

    def __init__(self, restorer: ControllableRestorer):
        super().__init__()
        self.config = restorer.config
        self.prediction_type = restorer.prediction_type
        self.register_buffer("alphas_cumprod", restorer.alphas_cumprod.clone(), persistent=False)
        self.action_overlay = restorer.action_overlay
        self.skip_vae = restorer.skip_vae
        self.unet = restorer.unet
        self.velocity_scale = restorer.velocity_scale
        self.velocity_zero_conv = restorer.velocity_zero_conv

    def forward(
        self,
        src_01: torch.Tensor,
        tgt_01: torch.Tensor,
        action_map: torch.Tensor,
        sem_tokens: torch.Tensor,
        sem_global: torch.Tensor,
        fidelity_scale: float = 1.0,
        detail_scale: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        bsz = src_01.shape[0]
        cross_tokens, global_action_cond = self.action_overlay(action_map, sem_tokens, sem_global)

        use_bridge = torch.rand(1, device=src_01.device).item() < getattr(
            self.config, "MULTI_STEP_TRAIN_PROB", 0.0
        )
        z_tgt = None
        if use_bridge:
            with torch.no_grad(), generator_lora_context(
                None, lambda_pix=0.0, lambda_sem=0.0, device=tgt_01.device
            ):
                z_tgt = self.skip_vae.encode_target_latents(tgt_01)

        set_generator_lora_state(
            global_action_cond,
            lambda_pix=fidelity_scale,
            lambda_sem=detail_scale,
            device=src_01.device,
        )
        set_spatial_action_map(action_map, device=src_01.device)

        z_src, enc_skips = self.skip_vae.encode_with_skips(src_01)

        if use_bridge and z_tgt is not None:
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
            fidelity_scale=fidelity_scale,
            detail_scale=detail_scale,
        )

        raw_rgb = self.skip_vae.decode_with_skips(hat_z_0, enc_skips, action_map)
        thresh = max(1e-4, float(getattr(self.config, "ZERO_ACTION_LOCK_THRESHOLD", 0.03)))
        gate = (action_map.float().amax(dim=1, keepdim=True) / thresh).clamp(0.0, 1.0)
        pred_rgb = (src_01 + gate * (raw_rgb - src_01)).clamp(0.0, 1.0)
        return {"pred": pred_rgb, "t": t_vec}


class SplitMultiGPURestorer(nn.Module):
    """Runs remote-code Vision Encoder on primary GPU and parallelizes the 950M Generator across all GPUs.

    Preserves 100% identical parameter names in `raw_model.trainable_state_dict()`.
    """

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
                fidelity_scale=f_scale,
                detail_scale=d_scale,
            )
            return {
                "pred": gen_out["pred"],
                "align_loss": align_loss.reshape(1),
                "t": gen_out["t"],
            }
        return raw_model.sample(
            src_01=src_01,
            action_input=action_input,
            steps=steps,
            fidelity_scale=fidelity_scale,
            detail_scale=detail_scale,
        )
