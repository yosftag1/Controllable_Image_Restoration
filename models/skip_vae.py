"""Skip-Connected AutoencoderKL with Lightweight Encoder LoRA & Frozen Natural Decoder.

1. `vae.encoder` receives lightweight pixel-stage LoRA (when `VAE_ENC_LORA_RANK > 0`).
2. `vae.decoder` stays 100% frozen so every decoded edge lies on SD's natural image manifold.
3. `_SpatialSkipBridge` passes action-gated multi-scale encoder skips into the frozen decoder,
   smoothly fading fine high-res skips as `detail_scale` increases so Stage-2 micro-texture is never washed out.
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .lora_layers import infer_module_dtype, inject_generator_dual_lora


class _SpatialSkipBridge(nn.Module):
    """Lean action-gated spatial skip bridge from VAE Encoder to Frozen VAE Decoder."""

    def __init__(
        self,
        enc_channels: int,
        dec_channels: int,
        num_actions: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        enc_groups = 32
        while enc_channels % enc_groups != 0 and enc_groups > 1:
            enc_groups //= 2
        self.norm = nn.GroupNorm(num_groups=enc_groups, num_channels=enc_channels, affine=False)

        mid_ch = min(96, max(32, enc_channels // 4))
        self.in_proj = nn.Conv2d(enc_channels, mid_ch, kernel_size=1, bias=False)
        self.dw_conv = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch, bias=False)
        nn.init.kaiming_uniform_(self.dw_conv.weight, a=math.sqrt(5))
        self.dw_conv.weight.data.mul_(0.50)

        self.act_mlp = nn.Sequential(
            nn.Conv2d(num_actions, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, mid_ch, kernel_size=1),
        )
        nn.init.normal_(self.act_mlp[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.act_mlp[-1].bias)

        # Zero-init without bias guarantees step-0 identity and zero DC color shift
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

        h = self.in_proj(self.norm(enc_skip_f))
        h = F.silu(h + self.dw_conv(h))
        gate = torch.sigmoid(self.act_mlp(a_local))

        bridge = self.zero_proj(gate * h) * keep_mask.to(dtype=torch.float32, device=dec_feat.device)
        return dec_feat + bridge.to(dec_feat.dtype)


class _MockAsymmetricDownsample2D(nn.Module):
    """Mimics diffusers AutoencoderKL Downsample2D: asymmetric pad (0,1,0,1) + Conv2d(k=3, s=2, p=0)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.downsamplers_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.downsamplers_conv(F.pad(x, (0, 1, 0, 1))))


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
                _MockAsymmetricDownsample2D(base_ch, base_ch),
                _MockAsymmetricDownsample2D(base_ch, base_ch * 2),
                _MockAsymmetricDownsample2D(base_ch * 2, base_ch * 4),
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
    """Wraps AutoencoderKL with lightweight Encoder LoRA, action-gated skip bridges & frozen Decoder."""

    def __init__(self, vae: nn.Module, config):
        super().__init__()
        self.vae = vae
        self.config = config
        self.scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
        self.gradient_checkpointing = bool(getattr(config, "GRADIENT_CHECKPOINTING", True))

        for p in self.vae.parameters():
            p.requires_grad = False

        enc_rank = int(getattr(config, "VAE_ENC_LORA_RANK", 8) or 0)
        enc_alpha = float(getattr(config, "VAE_ENC_LORA_ALPHA", float(enc_rank)))
        if enc_rank > 0:
            enc_pix_params, _ = inject_generator_dual_lora(
                self.vae.encoder,
                rank=enc_rank,
                alpha=enc_alpha,
                expert_specs=None,
                sem_rank=0,
                sem_alpha=0.0,
                include_conv=True,
            )
        else:
            enc_pix_params = []

        self.vae_pix_lora_params = list(enc_pix_params)

        with torch.no_grad():
            enc_dev = next(self.vae.encoder.parameters()).device
            dummy = torch.zeros(1, 3, 64, 64, device=enc_dev)
            z_dummy, enc_skips = self._raw_encode(dummy)
            dec_channels = self._probe_decoder_channels(z_dummy)

        deg_names = list(getattr(config, "DEGRADATION_TYPES", []))
        signed_set = set(getattr(config, "SIGNED_ACTION_TYPES", ["exposure", "warmth"]))
        struct_idx = [i for i, n in enumerate(deg_names) if n not in signed_set]
        self.register_buffer("struct_action_indices", torch.tensor(struct_idx, dtype=torch.long), persistent=False)
        blur_set = {"deblur_motion", "sharpen_focus"}
        blur_idx = [i for i, n in enumerate(deg_names) if n in blur_set]
        self.register_buffer("blur_action_indices", torch.tensor(blur_idx, dtype=torch.long), persistent=False)

        num_actions = config.NUM_ACTIONS
        mod_hidden = getattr(config, "SPATIAL_MOD_HIDDEN_DIM", 64)
        self.skip_dropout_prob = float(getattr(config, "SKIP_DROPOUT_PROB", 0.25))

        self.up_skip_bridges = nn.ModuleList(
            [
                _SpatialSkipBridge(
                    enc_skips[2].shape[1],
                    dec_channels[0],
                    num_actions,
                    hidden_dim=mod_hidden,
                ),
                _SpatialSkipBridge(
                    enc_skips[1].shape[1],
                    dec_channels[1],
                    num_actions,
                    hidden_dim=mod_hidden,
                ),
                _SpatialSkipBridge(
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

    def encode_target_latents(self, rgb_01: torch.Tensor) -> torch.Tensor:
        """Encode clean/selective target RGB image in [0, 1] into frozen-VAE target latents."""
        x_norm = rgb_01 * 2.0 - 1.0
        latents, _ = self._raw_encode(x_norm, collect_skips=False)
        return latents

    def decode_with_skips(
        self,
        latents: torch.Tensor,
        enc_skips: List[torch.Tensor],
        action_map: torch.Tensor,
        skip_scale: float = 1.0,
        detail_scale: float = 0.0,
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
            # Ensure at least one sample keeps skips so bridge parameters always receive gradients under DDP
            if float(keep_mask.sum().item()) == 0.0 and float(skip_scale) > 0.0:
                keep_mask[0] = float(skip_scale)
        else:
            keep_mask = torch.full((bsz, 1, 1, 1), float(skip_scale), device=latents.device, dtype=torch.float32)

        h = (latents / self.scaling_factor).to(vae_dtype)
        if getattr(self.vae, "post_quant_conv", None) is not None:
            h = self.vae.post_quant_conv(h)

        h = dec.conv_in(h)
        if use_ckpt and not h.requires_grad:
            h = h.requires_grad_(True)
        h = grad_checkpoint(dec.mid_block, h, use_reentrant=False) if use_ckpt else dec.mid_block(h)

        detail_env = min(1.0, max(0.0, float(detail_scale)))
        blur_fine_fade = (0.0, 0.25, 0.55)
        nonblur_fine_fade = (0.0, 0.05, 0.15)
        skip_indices = [2, 1, 0]
        for idx, up_block in enumerate(dec.up_blocks):
            if idx < 3 and len(enc_skips) > skip_indices[idx]:
                s_idx = skip_indices[idx]
                enc_skip = enc_skips[s_idx]
                if release_skips and not use_ckpt:
                    enc_skips[s_idx] = None
                if enc_skip is not None:
                    if (
                        detail_env > 1e-4
                        and (blur_fine_fade[idx] > 0.0 or nonblur_fine_fade[idx] > 0.0)
                        and self.struct_action_indices.numel() > 0
                    ):
                        thw = h.shape[-2:]
                        a_loc = (
                            F.interpolate(action_map.float(), size=thw, mode="area")
                            if action_map.shape[-2:] != thw
                            else action_map.float()
                        )
                        si = self.struct_action_indices.to(a_loc.device)
                        struct_act = a_loc.index_select(1, si).abs().amax(dim=1, keepdim=True).clamp(0.0, 1.0)
                        if self.blur_action_indices.numel() > 0:
                            bi = self.blur_action_indices.to(a_loc.device)
                            blur_act = a_loc.index_select(1, bi).abs().amax(dim=1, keepdim=True).clamp(0.0, 1.0)
                        else:
                            blur_act = torch.zeros_like(struct_act)
                        fade_amt = torch.maximum(
                            nonblur_fine_fade[idx] * struct_act,
                            blur_fine_fade[idx] * blur_act,
                        )
                        eff_keep = keep_mask * (1.0 - fade_amt * detail_env).clamp(0.0, 1.0)
                    else:
                        eff_keep = keep_mask

                    h = (
                        grad_checkpoint(
                            self.up_skip_bridges[idx], h, enc_skip, action_map, eff_keep, use_reentrant=False
                        )
                        if use_ckpt
                        else self.up_skip_bridges[idx](h, enc_skip, action_map, eff_keep)
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
