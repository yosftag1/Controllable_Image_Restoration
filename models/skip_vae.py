"""DataParallel-safe Universal eVAE Skip-Connected AutoencoderKL with Frozen Natural Decoder.

Follows OSDD's Dual-Path Feature Alignment (`eVAE`) + Universal Slider Control:
1. `vae.encoder` receives Expandable Slider-Routed UIR-LoRA Experts to align degraded spatial
   features toward the selective target `x_tgt` according to the user's sliders (`zero estimator`).
2. `vae.decoder` stays 100% FROZEN by default (`VAE_DEC_LORA_RANK = 0`), guaranteeing every decoded
   spatial edge and texture lies on Stable Diffusion's natural photographic manifold.
3. `_UniversalSpatialSkipBridge` replaces hardcoded `retain_env -> 0` blur suppression with a
   universal slider-conditioned spatial alignment adapter (`3x3` + `dilated 3x3` + `1x5/5x1` strip
   depthwise convs modulated via FiLM from `action_map`), preserving sharp high-res geometry across
   motion blur, defocus, denoising, and future knobs (e.g. lens distortion, lens flare).
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .lora_layers import infer_module_dtype, inject_generator_dual_lora, resolve_expert_specs


class _UniversalSpatialSkipBridge(nn.Module):
    """Universal slider-conditioned spatial skip alignment bridge from Encoder to Frozen Decoder."""

    def __init__(
        self,
        enc_channels: int,
        dec_channels: int,
        num_actions: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        num_groups = 32
        while enc_channels % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=enc_channels, affine=False)

        mid_ch = min(96, max(32, enc_channels // 4))
        self.in_proj = nn.Conv2d(enc_channels, mid_ch, kernel_size=1, bias=False)

        # Universal spatial realignment depthwise bank (local 3x3 + dilated context + oriented strips)
        self.dw_local = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch, bias=False)
        self.dw_dilated = nn.Conv2d(
            mid_ch, mid_ch, kernel_size=3, padding=2, dilation=2, groups=mid_ch, bias=False
        )
        self.dw_strip_h = nn.Conv2d(mid_ch, mid_ch, kernel_size=(1, 5), padding=(0, 2), groups=mid_ch, bias=False)
        self.dw_strip_v = nn.Conv2d(mid_ch, mid_ch, kernel_size=(5, 1), padding=(2, 0), groups=mid_ch, bias=False)

        for dw in (self.dw_local, self.dw_dilated, self.dw_strip_h, self.dw_strip_v):
            nn.init.kaiming_uniform_(dw.weight, a=math.sqrt(5))
            dw.weight.data.mul_(0.25)

        self.act_mlp = nn.Sequential(
            nn.Conv2d(num_actions, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, mid_ch * 3, kernel_size=1),
        )
        nn.init.zeros_(self.act_mlp[-1].weight)
        nn.init.zeros_(self.act_mlp[-1].bias)

        # bias=False guarantees zero DC color/brightness offset leak into the frozen VAE decoder!
        self.zero_proj = nn.Conv2d(mid_ch, dec_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.zero_proj.weight)

    def forward(
        self,
        dec_feat: torch.Tensor,
        enc_skip: torch.Tensor,
        action_map: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_hw = dec_feat.shape[-2:]
        if enc_skip.shape[-2:] != target_hw:
            enc_skip_f = F.interpolate(enc_skip.float(), size=target_hw, mode="bilinear", align_corners=False)
        else:
            enc_skip_f = enc_skip.float()

        if action_map.shape[-2:] != target_hw:
            a_local = F.interpolate(action_map.float(), size=target_hw, mode="area")
        else:
            a_local = action_map.float()

        normed_skip = self.norm(enc_skip_f)
        h = self.in_proj(normed_skip)

        raw_gamma, raw_spatial_mix, raw_gate = self.act_mlp(a_local).chunk(3, dim=1)
        gamma = torch.tanh(raw_gamma)
        spatial_mix = torch.tanh(raw_spatial_mix)
        gate = torch.sigmoid(raw_gate)

        # Modulate encoder features by user sliders, then apply universal spatial realignment bank
        h_mod = h * (1.0 + gamma)
        h_spatial = (
            self.dw_local(h_mod)
            + self.dw_dilated(h_mod)
            + 0.5 * (self.dw_strip_h(h_mod) + self.dw_strip_v(h_mod))
        )
        h_aligned = F.silu(h_mod + (0.5 + 0.5 * spatial_mix) * h_spatial)

        bridge = self.zero_proj(gate * h_aligned) * keep_mask.to(dtype=torch.float32, device=dec_feat.device)
        return dec_feat + bridge.to(dec_feat.dtype)


class _MockTinyVAE(nn.Module):
    """Compact AutoencoderKL-compatible 8x VAE for fast offline unit testing."""

    class _Cfg:
        scaling_factor = 0.18215
        latent_channels = 4

    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.config = self._Cfg()
        self.quant_conv = nn.Conv2d(8, 8, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(4, 4, kernel_size=1)

        self.encoder = nn.Module()
        self.encoder.conv_in = nn.Conv2d(3, base_ch, kernel_size=3, padding=1)
        self.encoder.down_blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1), nn.SiLU()),
                nn.Sequential(nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1), nn.SiLU()),
                nn.Sequential(nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1), nn.SiLU()),
                nn.Sequential(nn.Conv2d(base_ch * 4, base_ch * 4, 3, stride=1, padding=1), nn.SiLU()),
            ]
        )
        self.encoder.mid_block = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.SiLU(),
        )
        self.encoder.conv_norm_out = nn.GroupNorm(8, base_ch * 4)
        self.encoder.conv_act = nn.SiLU()
        self.encoder.conv_out = nn.Conv2d(base_ch * 4, 8, kernel_size=3, padding=1)

        self.decoder = nn.Module()
        self.decoder.conv_in = nn.Conv2d(4, base_ch * 4, kernel_size=3, padding=1)
        self.decoder.mid_block = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.SiLU(),
        )
        self.decoder.up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
                    nn.SiLU(),
                ),
                nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.SiLU()),
            ]
        )
        self.decoder.conv_norm_out = nn.GroupNorm(8, base_ch)
        self.decoder.conv_act = nn.SiLU()
        self.decoder.conv_out = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)


class SkipConnectedVAE(nn.Module):
    """Wraps diffusers AutoencoderKL with Slider-Routed eVAE Encoder LoRA & Universal Skip Alignment."""

    def __init__(self, vae: nn.Module, config):
        super().__init__()
        self.vae = vae
        self.config = config
        self.scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
        self.gradient_checkpointing = bool(getattr(config, "GRADIENT_CHECKPOINTING", True))

        for p in self.vae.parameters():
            p.requires_grad = False

        hypir_scale = (
            float(getattr(config, "HYPIR_LORA_RANK_SCALE", 2.0))
            if bool(getattr(config, "HYPIR_MODE", False))
            else 1.0
        )
        unet_base_r = max(1, int(getattr(config, "UNET_LORA_RANK", 28)))
        enc_rank = int(getattr(config, "VAE_ENC_LORA_RANK", 16) or 0)
        enc_alpha = float(getattr(config, "VAE_ENC_LORA_ALPHA", float(enc_rank)))
        enc_sem_rank = max(6, int(getattr(config, "SEM_DETAIL_LORA_RANK", 16)) * enc_rank // unet_base_r)

        enc_expert_specs = resolve_expert_specs(
            getattr(config, "LORA_EXPERT_SPECS", None),
            getattr(config, "DEGRADATION_TYPES", None),
            signed_types=getattr(config, "SIGNED_ACTION_TYPES", ("exposure", "warmth")),
            default_rank=enc_rank,
            rank_scale=(float(enc_rank) / float(unet_base_r)) * hypir_scale,
        )

        if enc_rank > 0:
            enc_pix_params, enc_sem_params = inject_generator_dual_lora(
                self.vae.encoder,
                action_dim=config.ACTION_HIDDEN_DIM,
                rank=enc_rank,
                alpha=enc_alpha,
                include_conv=True,
                expert_specs=enc_expert_specs,
                sem_rank=enc_sem_rank,
                sem_alpha=float(enc_sem_rank),
                use_dora=bool(getattr(config, "USE_DORA", True)),
                unfreeze_norms=False,
            )
        else:
            enc_pix_params, enc_sem_params = [], []

        dec_rank = int(getattr(config, "VAE_DEC_LORA_RANK", 0) or 0)
        dec_alpha = float(getattr(config, "VAE_DEC_LORA_ALPHA", float(dec_rank)))
        if dec_rank > 0:
            dec_pix_params, dec_sem_params = inject_generator_dual_lora(
                self.vae.decoder,
                action_dim=config.ACTION_HIDDEN_DIM,
                rank=dec_rank,
                alpha=dec_alpha,
                include_conv=True,
                expert_specs=enc_expert_specs,
                sem_rank=max(4, dec_rank // 2),
                sem_alpha=float(max(4, dec_rank // 2)),
                use_dora=bool(getattr(config, "USE_DORA", True)),
                unfreeze_norms=False,
            )
        else:
            dec_pix_params, dec_sem_params = [], []

        self.vae_pix_lora_params = list(enc_pix_params) + list(dec_pix_params)
        self.vae_sem_lora_params = list(enc_sem_params) + list(dec_sem_params)

        with torch.no_grad():
            enc_dev = next(self.vae.encoder.parameters()).device
            dummy = torch.zeros(1, 3, 64, 64, device=enc_dev)
            z_dummy, enc_skips = self._raw_encode(dummy)
            dec_channels = self._probe_decoder_channels(z_dummy)

        num_actions = config.NUM_ACTIONS
        mod_hidden = getattr(config, "SPATIAL_MOD_HIDDEN_DIM", 64)
        self.skip_dropout_prob = float(getattr(config, "SKIP_DROPOUT_PROB", 0.0))

        self.up_skip_bridges = nn.ModuleList(
            [
                _UniversalSpatialSkipBridge(
                    enc_skips[2].shape[1],
                    dec_channels[0],
                    num_actions,
                    hidden_dim=mod_hidden,
                ),
                _UniversalSpatialSkipBridge(
                    enc_skips[1].shape[1],
                    dec_channels[1],
                    num_actions,
                    hidden_dim=mod_hidden,
                ),
                _UniversalSpatialSkipBridge(
                    enc_skips[0].shape[1],
                    dec_channels[2],
                    num_actions,
                    hidden_dim=mod_hidden,
                ),
            ]
        )

    def _raw_encode(
        self, x_minus1_to_1: torch.Tensor, collect_skips: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        enc = self.vae.encoder
        vae_dtype = infer_module_dtype(enc, fallback=x_minus1_to_1.dtype)
        use_ckpt = getattr(self, "gradient_checkpointing", False) and self.training and torch.is_grad_enabled()

        h = enc.conv_in(x_minus1_to_1.to(vae_dtype))
        if use_ckpt and not h.requires_grad:
            h = h.requires_grad_(True)
        skips: List[torch.Tensor] = []

        num_down = len(enc.down_blocks)
        for idx, down_block in enumerate(enc.down_blocks):
            h = grad_checkpoint(down_block, h, use_reentrant=False) if use_ckpt else down_block(h)
            if collect_skips and idx < num_down - 1:
                skips.append(h)

        h = grad_checkpoint(enc.mid_block, h, use_reentrant=False) if use_ckpt else enc.mid_block(h)
        h = enc.conv_norm_out(h)
        h = enc.conv_act(h)
        moments = enc.conv_out(h)
        if getattr(self.vae, "quant_conv", None) is not None:
            moments = self.vae.quant_conv(moments)

        mean, _ = torch.chunk(moments, 2, dim=1)
        latents = mean * self.scaling_factor
        return latents, skips

    def _probe_decoder_channels(self, z: torch.Tensor) -> List[int]:
        dec = self.vae.decoder
        vae_dtype = infer_module_dtype(dec, fallback=z.dtype)
        h = z.to(vae_dtype) / self.scaling_factor
        if getattr(self.vae, "post_quant_conv", None) is not None:
            h = self.vae.post_quant_conv(h)
        h = dec.conv_in(h)
        h = dec.mid_block(h)

        ch_list: List[int] = []
        for idx, up_block in enumerate(dec.up_blocks):
            if idx < 3:
                ch_list.append(int(h.shape[1]))
            h = up_block(h)
        return ch_list

    def encode_with_skips(self, rgb_01: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x_norm = rgb_01 * 2.0 - 1.0
        return self._raw_encode(x_norm, collect_skips=True)

    def encode_target_latents(self, rgb_01: torch.Tensor, return_skips: bool = False):
        """Encode clean/selective target RGB image in [0, 1] into frozen-VAE target latents (& skips)."""
        x_norm = rgb_01 * 2.0 - 1.0
        latents, skips = self._raw_encode(x_norm, collect_skips=return_skips)
        if return_skips:
            return latents, skips
        return latents

    @staticmethod
    def compute_evae_alignment_loss(
        student_skips: List[torch.Tensor],
        target_skips: List[torch.Tensor],
    ) -> torch.Tensor:
        """OSDD-style universal multi-scale feature alignment between student eVAE & clean teacher skips."""
        if not student_skips or not target_skips:
            return torch.zeros((), dtype=torch.float32)
        n = min(len(student_skips), len(target_skips))
        total = student_skips[0].new_zeros((), dtype=torch.float32)
        scale_weights = (0.30, 0.35, 0.35)
        w_sum = 0.0
        for idx in range(n):
            sw = scale_weights[idx] if idx < len(scale_weights) else (1.0 / float(n))
            s_f = student_skips[idx].float()
            t_f = target_skips[idx].detach().float()
            s_norm = F.normalize(s_f, dim=1, eps=1e-6)
            t_norm = F.normalize(t_f, dim=1, eps=1e-6)
            cos_loss = (1.0 - (s_norm * t_norm).sum(dim=1)).mean()
            l1_loss = F.smooth_l1_loss(s_f, t_f, beta=0.25)
            total = total + sw * (0.65 * cos_loss + 0.35 * l1_loss)
            w_sum += sw
        return total / max(1e-6, w_sum)

    def decode_with_skips(
        self,
        latents: torch.Tensor,
        enc_skips: List[torch.Tensor],
        action_map: torch.Tensor,
        skip_scale: float = 1.0,
        release_skips: bool = False,
    ) -> torch.Tensor:
        """Decode latents back to [0, 1] RGB through frozen natural-manifold AutoencoderKL decoder."""
        dec = self.vae.decoder
        vae_dtype = infer_module_dtype(dec, fallback=latents.dtype)
        use_ckpt = getattr(self, "gradient_checkpointing", False) and self.training and torch.is_grad_enabled()

        bsz = latents.shape[0]
        if self.training and torch.is_grad_enabled() and self.skip_dropout_prob > 0.0:
            keep_prob = max(0.0, min(1.0, 1.0 - self.skip_dropout_prob))
            keep_mask = (
                torch.rand((bsz, 1, 1, 1), device=latents.device, dtype=torch.float32) < keep_prob
            ).float() * float(skip_scale)
        else:
            keep_mask = torch.full((bsz, 1, 1, 1), float(skip_scale), device=latents.device, dtype=torch.float32)

        h = (latents / self.scaling_factor).to(vae_dtype)
        if getattr(self.vae, "post_quant_conv", None) is not None:
            h = self.vae.post_quant_conv(h)

        h = dec.conv_in(h)
        if use_ckpt and not h.requires_grad:
            h = h.requires_grad_(True)
        h = grad_checkpoint(dec.mid_block, h, use_reentrant=False) if use_ckpt else dec.mid_block(h)

        skip_indices = [2, 1, 0]
        for idx, up_block in enumerate(dec.up_blocks):
            if idx < 3 and len(enc_skips) > skip_indices[idx]:
                s_idx = skip_indices[idx]
                enc_skip = enc_skips[s_idx]
                if release_skips and not use_ckpt:
                    enc_skips[s_idx] = None
                if enc_skip is not None:
                    h = (
                        grad_checkpoint(
                            self.up_skip_bridges[idx], h, enc_skip, action_map, keep_mask, use_reentrant=False
                        )
                        if use_ckpt
                        else self.up_skip_bridges[idx](h, enc_skip, action_map, keep_mask)
                    )
                    del enc_skip
            h = grad_checkpoint(up_block, h, use_reentrant=False) if use_ckpt else up_block(h)

        h = dec.conv_norm_out(h)
        h = dec.conv_act(h)
        rgb_minus1_to_1 = dec.conv_out(h).float()
        rgb_01 = (rgb_minus1_to_1 * 0.5 + 0.5).clamp(0.0, 1.0)
        return rgb_01

    def skip_bridge_parameters(self) -> List[nn.Parameter]:
        return list(self.up_skip_bridges.parameters())
