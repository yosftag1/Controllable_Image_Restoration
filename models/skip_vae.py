"""DataParallel-safe Skip-Connected AutoencoderKL with 1x1 Action-Gated Bridges.

Preserves untouched high-frequency pixel details (motion blur trajectories, grain,
sharp edges, colors) across the 8x latent bottleneck via zero-initialized
Encoder -> Decoder skip connections modulated locally by the Spatial Action Overlay.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .action_overlay import PointwiseSpatialModulator
from .lora_layers import infer_module_dtype, inject_generator_dual_lora


class _ActionZeroSkipBridge(nn.Module):
    """Pointwise 1x1 action-modulated zero-convolution skip bridge from Encoder to Decoder."""

    def __init__(self, enc_channels: int, dec_channels: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.spatial_mod = PointwiseSpatialModulator(
            num_actions=num_actions,
            channels=enc_channels,
            hidden_dim=hidden_dim,
        )
        self.zero_proj = nn.Conv2d(enc_channels, dec_channels, kernel_size=1)
        nn.init.zeros_(self.zero_proj.weight)
        nn.init.zeros_(self.zero_proj.bias)

    def forward(self, dec_feat: torch.Tensor, enc_skip: torch.Tensor, action_map: torch.Tensor) -> torch.Tensor:
        if enc_skip.shape[-2:] != dec_feat.shape[-2:]:
            enc_skip = F.interpolate(enc_skip.float(), size=dec_feat.shape[-2:], mode="bilinear", align_corners=False)
        modulated = self.spatial_mod(enc_skip, action_map)
        bridge = self.zero_proj(modulated.float())
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
                nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1), nn.SiLU()),
                nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1), nn.SiLU()),
                nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(base_ch * 2, base_ch, 3, padding=1), nn.SiLU()),
                nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.SiLU()),
            ]
        )
        self.decoder.conv_norm_out = nn.GroupNorm(8, base_ch)
        self.decoder.conv_act = nn.SiLU()
        self.decoder.conv_out = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)


class SkipConnectedVAE(nn.Module):
    """Wraps diffusers AutoencoderKL with explicit multi-scale Action-Gated Skip Bridges."""

    def __init__(self, vae: nn.Module, config):
        super().__init__()
        self.vae = vae
        self.config = config
        self.scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
        self.gradient_checkpointing = bool(getattr(config, "GRADIENT_CHECKPOINTING", True))

        # Freeze entire base VAE first
        for p in self.vae.parameters():
            p.requires_grad = False

        # Inject lightweight Dual-LoRA into VAE Encoder (S3Diff-style degradation-adaptive encoding)
        enc_rank = getattr(config, "VAE_ENC_LORA_RANK", 8)
        enc_alpha = getattr(config, "VAE_ENC_LORA_ALPHA", 8.0)
        self.vae_pix_lora_params, self.vae_sem_lora_params = inject_generator_dual_lora(
            self.vae.encoder,
            action_dim=config.ACTION_HIDDEN_DIM,
            rank=enc_rank,
            alpha=enc_alpha,
            include_conv=True,
        )

        # Dynamically probe Encoder & Decoder intermediate channel dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            z_dummy, enc_skips = self._raw_encode(dummy)
            dec_channels = self._probe_decoder_channels(z_dummy)

        num_actions = config.NUM_ACTIONS
        mod_hidden = getattr(config, "SPATIAL_MOD_HIDDEN_DIM", 64)

        # enc_skips = [skip_1x (H), skip_2x (H/2), skip_4x (H/4), skip_8x (H/8)]
        # dec_channels = [pre_up0 (H/8), pre_up1 (H/4), pre_up2 (H/2), post_up3 (H)]
        self.up_skip_bridges = nn.ModuleList(
            [
                _ActionZeroSkipBridge(enc_skips[3].shape[1], dec_channels[0], num_actions, mod_hidden),
                _ActionZeroSkipBridge(enc_skips[2].shape[1], dec_channels[1], num_actions, mod_hidden),
                _ActionZeroSkipBridge(enc_skips[1].shape[1], dec_channels[2], num_actions, mod_hidden),
            ]
        )
        self.fullres_skip_bridge = _ActionZeroSkipBridge(
            enc_skips[0].shape[1], dec_channels[3], num_actions, mod_hidden
        )

    def _raw_encode(self, x_minus1_to_1: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        enc = self.vae.encoder
        vae_dtype = infer_module_dtype(enc, fallback=x_minus1_to_1.dtype)
        use_ckpt = getattr(self, "gradient_checkpointing", False) and self.training and torch.is_grad_enabled()

        h = enc.conv_in(x_minus1_to_1.to(vae_dtype))
        if use_ckpt and not h.requires_grad:
            h = h.requires_grad_(True)
        skips: List[torch.Tensor] = [h]  # skip[0]: full resolution (H, W)

        num_down = len(enc.down_blocks)
        for idx, down_block in enumerate(enc.down_blocks):
            h = grad_checkpoint(down_block, h, use_reentrant=False) if use_ckpt else down_block(h)
            if idx < num_down - 1:
                skips.append(h)          # skip[1]: H/2, skip[2]: H/4, skip[3]: H/8

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
        ch_list.append(int(h.shape[1]))  # post up_blocks[-1] full-resolution feature channels
        return ch_list

    def encode_with_skips(self, rgb_01: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode RGB image in [0, 1] into scaled latents + 4 multi-scale skip tensors."""
        x_norm = rgb_01 * 2.0 - 1.0
        return self._raw_encode(x_norm)

    def encode_target_latents(self, rgb_01: torch.Tensor) -> torch.Tensor:
        """Encode clean/target RGB image in [0, 1] without storing skip activations."""
        latents, _ = self.encode_with_skips(rgb_01)
        return latents

    def decode_with_skips(
        self,
        latents: torch.Tensor,
        enc_skips: List[torch.Tensor],
        action_map: torch.Tensor,
        skip_scale: float = 1.0,
        release_skips: bool = False,
    ) -> torch.Tensor:
        """Decode latents back to [0, 1] RGB using explicit encoder skips & 1x1 action overlay."""
        dec = self.vae.decoder
        vae_dtype = infer_module_dtype(dec, fallback=latents.dtype)
        use_ckpt = getattr(self, "gradient_checkpointing", False) and self.training and torch.is_grad_enabled()

        h = (latents / self.scaling_factor).to(vae_dtype)
        if getattr(self.vae, "post_quant_conv", None) is not None:
            h = self.vae.post_quant_conv(h)

        h = dec.conv_in(h)
        if use_ckpt and not h.requires_grad:
            h = h.requires_grad_(True)
        h = grad_checkpoint(dec.mid_block, h, use_reentrant=False) if use_ckpt else dec.mid_block(h)

        # Reverse order matches [skip_8x (3), skip_4x (2), skip_2x (1)]
        skip_indices = [3, 2, 1]
        for idx, up_block in enumerate(dec.up_blocks):
            if idx < 3 and skip_scale > 0.0:
                s_idx = skip_indices[idx]
                enc_skip = enc_skips[s_idx]
                if release_skips and not use_ckpt:
                    enc_skips[s_idx] = None
                bridged = (
                    grad_checkpoint(self.up_skip_bridges[idx], h, enc_skip, action_map, use_reentrant=False)
                    if use_ckpt
                    else self.up_skip_bridges[idx](h, enc_skip, action_map)
                )
                del enc_skip
                h = h + skip_scale * (bridged - h)
                del bridged
            h = grad_checkpoint(up_block, h, use_reentrant=False) if use_ckpt else up_block(h)

        if skip_scale > 0.0:
            full_skip = enc_skips[0]
            if release_skips and not use_ckpt:
                enc_skips[0] = None
            bridged_full = (
                grad_checkpoint(self.fullres_skip_bridge, h, full_skip, action_map, use_reentrant=False)
                if use_ckpt
                else self.fullres_skip_bridge(h, full_skip, action_map)
            )
            del full_skip
            h = h + skip_scale * (bridged_full - h)
            del bridged_full

        h = dec.conv_norm_out(h)
        h = dec.conv_act(h)
        rgb_minus1_to_1 = dec.conv_out(h).float()
        rgb_01 = (rgb_minus1_to_1 * 0.5 + 0.5).clamp(0.0, 1.0)
        return rgb_01

    def skip_bridge_parameters(self) -> List[nn.Parameter]:
        params = list(self.up_skip_bridges.parameters()) + list(self.fullres_skip_bridge.parameters())
        return params
