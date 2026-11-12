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
import threading
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_overlay import ActionOverlayEncoder, PointwiseSpatialModulator
from .lora_layers import (
    generator_lora_context,
    infer_module_device,
    infer_module_dtype,
    inject_generator_dual_lora,
)
from .semantic_encoder import SemanticEncoder, load_hf_public_safe
from .skip_vae import SkipConnectedVAE, _MockTinyVAE


_SPATIAL_OVERLAY_STATE = threading.local()


@contextmanager
def _spatial_action_context(action_map: Optional[torch.Tensor]):
    prev = getattr(_SPATIAL_OVERLAY_STATE, "action_map", None)
    _SPATIAL_OVERLAY_STATE.action_map = action_map
    try:
        yield
    finally:
        _SPATIAL_OVERLAY_STATE.action_map = prev


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
        action_map = getattr(_SPATIAL_OVERLAY_STATE, "action_map", None)
        if action_map is None:
            return out

        if isinstance(out, tuple):
            sample = out[0]
            if action_map.device != sample.device:
                action_map = action_map.to(sample.device)
            mod_sample = self.spatial_mod(sample, action_map)
            return (mod_sample,) + out[1:]

        if action_map.device != out.device:
            action_map = action_map.to(out.device)
        return self.spatial_mod(out, action_map)


class _MockTinyUNetBlock(nn.Module):
    """Lightweight cross-attention + conv block mimicking diffusers UNet2DConditionModel."""

    def __init__(self, in_ch: int, out_ch: int, cross_dim: int, stride: int = 1, upsample: bool = False):
        super().__init__()
        self.has_cross_attention = True
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

    def forward(self, x: torch.Tensor, temb: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
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

        # 2. Load Pretrained AutoencoderKL & UNet2DConditionModel (no CLIP text encoder bloat!)
        if backbone_key == "mock_tiny":
            raw_vae = _MockTinyVAE(base_ch=32)
            raw_unet = _MockTinyUNet()
        else:
            from diffusers import AutoencoderKL, UNet2DConditionModel

            repo_id = config.GENERATOR_REPO_MAP.get(backbone_key, config.GENERATOR_BACKBONE)
            raw_vae = load_hf_public_safe(AutoencoderKL.from_pretrained, repo_id, subfolder="vae")
            raw_unet = load_hf_public_safe(UNet2DConditionModel.from_pretrained, repo_id, subfolder="unet")

        # 3. Wrap VAE with Action-Gated Skip Bridges + Encoder Dual-LoRA
        self.skip_vae = SkipConnectedVAE(raw_vae, config)

        # 4. Inject Action-Modulated Dual-LoRA (Pixel-LoRA + Semantic-LoRA) into UNet
        self.unet = raw_unet
        self.unet_pix_lora_params, self.unet_sem_lora_params = inject_generator_dual_lora(
            self.unet,
            action_dim=config.ACTION_HIDDEN_DIM,
            rank=config.UNET_LORA_RANK,
            alpha=config.UNET_LORA_ALPHA,
            include_conv=True,
        )

        # 5. Wrap UNet multi-scale blocks with 1x1 Pointwise Spatial Action Overlay Modulators
        cross_attn_dim = getattr(self.unet.config, "cross_attention_dim", 768)
        if isinstance(cross_attn_dim, (list, tuple)):
            cross_attn_dim = int(cross_attn_dim[-1])
        else:
            cross_attn_dim = int(cross_attn_dim)

        self.spatial_mod_params: List[nn.Parameter] = []
        self._attach_unet_spatial_overlays(config, cross_attn_dim=cross_attn_dim)

        # 6. Action & Semantic Token Encoder
        self.action_overlay = ActionOverlayEncoder(
            num_actions=config.NUM_ACTIONS,
            sem_spatial_dim=self.semantic_encoder.spatial_dim,
            sem_global_dim=self.semantic_encoder.global_dim,
            cross_attn_dim=cross_attn_dim,
            action_hidden_dim=config.ACTION_HIDDEN_DIM,
        )

        # 7. Smooth Latent Velocity Scaling + Zero-Init Residual Head
        latent_ch = int(getattr(self.unet.config, "out_channels", 4))
        init_scale = float(getattr(config, "INITIAL_VELOCITY_SCALE", 0.15))
        self.velocity_scale = nn.Parameter(torch.full((1, latent_ch, 1, 1), init_scale))
        self.velocity_zero_conv = nn.Conv2d(latent_ch, latent_ch, kernel_size=1)
        nn.init.zeros_(self.velocity_zero_conv.weight)
        nn.init.zeros_(self.velocity_zero_conv.bias)

    def _attach_unet_spatial_overlays(self, config, cross_attn_dim: int):
        """Probe exact runtime UNet block output channels and wrap with 1x1 spatial modulators."""
        num_actions = config.NUM_ACTIONS
        mod_hidden = getattr(config, "SPATIAL_MOD_HIDDEN_DIM", 64)

        in_ch = int(getattr(self.unet.config, "in_channels", 4))
        unet_dev = infer_module_device(self.unet)
        unet_dtype = infer_module_dtype(self.unet)

        probed_channels: Dict[str, int] = {}
        hooks = []

        def _make_probe_hook(stage_key: str):
            def _hook(_module, _inputs, output):
                sample = output[0] if isinstance(output, tuple) else output
                probed_channels[stage_key] = int(sample.shape[1])

            return _hook

        hooks.append(self.unet.mid_block.register_forward_hook(_make_probe_hook("mid")))
        for idx, blk in enumerate(self.unet.up_blocks):
            hooks.append(blk.register_forward_hook(_make_probe_hook(f"up_{idx}")))

        try:
            with torch.no_grad():
                dummy_lat = torch.zeros(1, in_ch, 32, 32, device=unet_dev, dtype=unet_dtype)
                dummy_t = torch.tensor([500], device=unet_dev, dtype=torch.long)
                dummy_enc = torch.zeros(1, 4, cross_attn_dim, device=unet_dev, dtype=unet_dtype)
                self.unet(dummy_lat, dummy_t, encoder_hidden_states=dummy_enc)
        finally:
            for handle in hooks:
                handle.remove()

        mid_ch = probed_channels["mid"]
        wrapped_mid = _SpatialModulatedUNetStage(self.unet.mid_block, mid_ch, num_actions, mod_hidden)
        self.unet.mid_block = wrapped_mid
        self.spatial_mod_params.extend(list(wrapped_mid.spatial_mod.parameters()))

        for idx in range(len(self.unet.up_blocks)):
            out_ch = probed_channels[f"up_{idx}"]
            wrapped_up = _SpatialModulatedUNetStage(self.unet.up_blocks[idx], out_ch, num_actions, mod_hidden)
            self.unet.up_blocks[idx] = wrapped_up
            self.spatial_mod_params.extend(list(wrapped_up.spatial_mod.parameters()))

    def get_stage_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Return cleanly separated parameter groups for 3-phase staged or joint training."""
        vision_params = list(self.semantic_encoder.vision_lora_params)
        pixel_params = (
            list(self.unet_pix_lora_params)
            + list(self.skip_vae.vae_pix_lora_params)
            + list(self.skip_vae.skip_bridge_parameters())
            + list(self.action_overlay.parameters())
            + list(self.spatial_mod_params)
            + [self.velocity_scale]
            + list(self.velocity_zero_conv.parameters())
        )
        semantic_params = list(self.unet_sem_lora_params) + list(self.skip_vae.vae_sem_lora_params)
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
        """Load lightweight adapter/LoRA/skip/overlay weights on top of frozen pretrained backbones."""
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if strict and unexpected:
            raise RuntimeError(f"Unexpected keys when loading controllable adapters: {unexpected}")
        return missing, unexpected

    def _predict_velocity(
        self,
        z_t: torch.Tensor,
        t_cont: torch.Tensor,
        cross_tokens: torch.Tensor,
        global_action_cond: torch.Tensor,
        action_map: torch.Tensor,
        fidelity_scale: float,
        detail_scale: float,
    ) -> torch.Tensor:
        """Predict latent Rectified Flow velocity v_theta(z_t, t) with Dual-LoRA & Spatial Overlay."""
        if t_cont.ndim == 0:
            t_cont = t_cont.unsqueeze(0)
        if t_cont.shape[0] == 1 and z_t.shape[0] > 1:
            t_cont = t_cont.expand(z_t.shape[0])

        t_unet = (t_cont.float() * 999.0).round().long().clamp(1, 999)
        unet_dtype = infer_module_dtype(self.unet, fallback=z_t.dtype)

        with generator_lora_context(global_action_cond, lambda_pix=fidelity_scale, lambda_sem=detail_scale):
            with _spatial_action_context(action_map):
                raw_pred = self.unet(
                    z_t.to(unet_dtype),
                    t_unet,
                    encoder_hidden_states=cross_tokens.to(unet_dtype),
                ).sample.float()

        vel = self.velocity_scale * raw_pred + self.velocity_zero_conv(z_t.float())
        return vel

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

        # 1. Self-Teacher Vision Semantics + Alignment Loss
        align_loss, sem_out = self.semantic_encoder.compute_self_teacher_loss(src_01, clean_01)
        cross_tokens, global_action_cond = self.action_overlay(
            action_map, sem_out["tokens"], sem_out["global"]
        )

        # 2. Encode source image (with Encoder Dual-LoRA active)
        with generator_lora_context(global_action_cond, lambda_pix=fidelity_scale, lambda_sem=detail_scale):
            z_src, enc_skips = self.skip_vae.encode_with_skips(src_01)

        # 3. Mix 1-step direct (t=1.0) and multi-step Rectified Flow bridge (t in [0.25, 1.0])
        use_bridge = torch.rand(1, device=src_01.device).item() < getattr(
            self.config, "MULTI_STEP_TRAIN_PROB", 0.30
        )
        if use_bridge:
            with torch.no_grad():
                z_tgt = self.skip_vae.encode_target_latents(tgt_01)
            t_vec = torch.empty(bsz, device=src_01.device).uniform_(0.25, 1.0)
            t_4d = t_vec.view(bsz, 1, 1, 1)
            z_t = (1.0 - t_4d) * z_tgt + t_4d * z_src
        else:
            t_vec = torch.ones(bsz, device=src_01.device)
            t_4d = t_vec.view(bsz, 1, 1, 1)
            z_t = z_src

        # 4. Predict clean/target latent endpoint hat_z_0 = z_t - t * v_theta(z_t, t)
        vel = self._predict_velocity(
            z_t=z_t,
            t_cont=t_vec,
            cross_tokens=cross_tokens,
            global_action_cond=global_action_cond,
            action_map=action_map,
            fidelity_scale=fidelity_scale,
            detail_scale=detail_scale,
        )
        hat_z_0 = z_t - t_4d * vel

        # 5. Decode back to RGB via Action-Gated Skip-VAE & apply Zero-Action Lock
        raw_rgb = self.skip_vae.decode_with_skips(hat_z_0, enc_skips, action_map)
        pred_rgb = self._apply_zero_action_lock(src_01, raw_rgb, action_map)

        return {
            "pred": pred_rgb,
            "align_loss": align_loss.reshape(1),
            "t": t_vec,
        }

    @torch.no_grad()
    def sample(
        self,
        src_01: torch.Tensor,
        action_input: torch.Tensor,
        steps: Optional[int] = None,
        fidelity_scale: Optional[float] = None,
        detail_scale: Optional[float] = None,
        return_trajectory: bool = False,
    ):
        """Controllable 1-to-N step Rectified Flow restoration at inference time."""
        steps = int(steps or getattr(self.config, "DEFAULT_INFERENCE_STEPS", 1))
        fidelity_scale = (
            float(self.config.DEFAULT_FIDELITY_SCALE) if fidelity_scale is None else float(fidelity_scale)
        )
        detail_scale = (
            float(self.config.DEFAULT_DETAIL_SCALE) if detail_scale is None else float(detail_scale)
        )

        bsz, _, h, w = src_01.shape
        action_map = self.action_overlay.ensure_spatial_map(action_input, h, w)

        # Fast shortcut if the entire action overlay is zero everywhere
        if float(action_map.max().item()) <= 1e-5:
            if return_trajectory:
                return src_01.clone(), [src_01.clone()]
            return src_01.clone()

        # Pad non-8-divisible resolutions cleanly for the 8x latent UNet, then crop back
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            src_padded = F.pad(src_01, (0, pad_w, 0, pad_h), mode="reflect")
            act_padded = F.pad(action_map, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            src_padded = src_01
            act_padded = action_map

        sem_out = self.semantic_encoder(src_padded, use_lora=True)
        cross_tokens, global_action_cond = self.action_overlay(
            act_padded, sem_out["tokens"], sem_out["global"]
        )

        with generator_lora_context(global_action_cond, lambda_pix=fidelity_scale, lambda_sem=detail_scale):
            z_cur, enc_skips = self.skip_vae.encode_with_skips(src_padded)

        t_schedule = torch.linspace(1.0, 0.0, steps + 1, device=src_01.device)
        trajectory: List[torch.Tensor] = [src_01.clone()] if return_trajectory else []

        for i in range(steps):
            t_cur = t_schedule[i].expand(bsz)
            dt = float(t_schedule[i] - t_schedule[i + 1])
            vel = self._predict_velocity(
                z_t=z_cur,
                t_cont=t_cur,
                cross_tokens=cross_tokens,
                global_action_cond=global_action_cond,
                action_map=act_padded,
                fidelity_scale=fidelity_scale,
                detail_scale=detail_scale,
            )
            z_cur = z_cur - dt * vel

            if return_trajectory and i < steps - 1:
                mid_rgb = self.skip_vae.decode_with_skips(z_cur, enc_skips, act_padded)[:, :, :h, :w]
                trajectory.append(self._apply_zero_action_lock(src_01, mid_rgb, action_map))

        raw_rgb = self.skip_vae.decode_with_skips(z_cur, enc_skips, act_padded)[:, :, :h, :w]
        final_rgb = self._apply_zero_action_lock(src_01, raw_rgb, action_map)

        if return_trajectory:
            trajectory.append(final_rgb)
            return final_rgb, trajectory
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

        with generator_lora_context(global_action_cond, lambda_pix=fidelity_scale, lambda_sem=detail_scale):
            z_src, enc_skips = self.skip_vae.encode_with_skips(src_01)

        use_bridge = torch.rand(1, device=src_01.device).item() < getattr(
            self.config, "MULTI_STEP_TRAIN_PROB", 0.30
        )
        if use_bridge:
            with torch.no_grad():
                z_tgt = self.skip_vae.encode_target_latents(tgt_01)
            t_vec = torch.empty(bsz, device=src_01.device).uniform_(0.25, 1.0)
            t_4d = t_vec.view(bsz, 1, 1, 1)
            z_t = (1.0 - t_4d) * z_tgt + t_4d * z_src
        else:
            t_vec = torch.ones(bsz, device=src_01.device)
            t_4d = t_vec.view(bsz, 1, 1, 1)
            z_t = z_src

        t_unet = (t_vec.float() * 999.0).round().long().clamp(1, 999)
        unet_dtype = infer_module_dtype(self.unet, fallback=z_t.dtype)

        with generator_lora_context(global_action_cond, lambda_pix=fidelity_scale, lambda_sem=detail_scale):
            with _spatial_action_context(action_map):
                raw_pred = self.unet(
                    z_t.to(unet_dtype),
                    t_unet,
                    encoder_hidden_states=cross_tokens.to(unet_dtype),
                ).sample.float()

        vel = self.velocity_scale * raw_pred + self.velocity_zero_conv(z_t.float())
        hat_z_0 = z_t - t_4d * vel

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
