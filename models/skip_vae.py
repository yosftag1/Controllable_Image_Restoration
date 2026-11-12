"""DataParallel-safe Skip-Connected AutoencoderKL with 1x1 Action-Gated Bridges.

Preserves untouched high-frequency pixel details (motion blur trajectories, grain,
sharp edges, colors) across the 8x latent bottleneck via zero-initialized
Encoder -> Decoder skip connections modulated locally by the Spatial Action Overlay.
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .lora_layers import infer_module_dtype, inject_generator_dual_lora


# Purely radiometric/tonal edits where spatial edges in the input are still clean once mean/scale-normalized
_RADIOMETRIC_ONLY_NAMES = {"color_degradation", "exposure_issue"}


class _ActionGatedSkipBridge(nn.Module):
    """Action-suppressed, zero-mean GroupNorm 1x1 skip bridge from Encoder to Decoder.

    Prevents the Skip-VAE shortcut pathology:
    1. `GroupNorm(..., affine=False)` + `bias=False` strips global color casts & exposure shifts
       from encoder features so radiometric fixes always come from the UNet's clean latent `hat_z_0`.
    2. Explicit structural suppression envelope `(1 - s * a_struct)^2` shuts off degraded encoder
       skips whenever spatial/quantization restoration sliders (deblur, SR, deband, denoise, JPEG)
       are active, forcing the decoder to reconstruct clean structure from the UNet.
    """

    def __init__(
        self,
        enc_channels: int,
        dec_channels: int,
        num_actions: int,
        struct_indices: Sequence[int],
        suppression_strength: float = 1.0,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.suppression_strength = float(suppression_strength)
        idx_tensor = torch.tensor(list(struct_indices), dtype=torch.long)
        self.register_buffer("struct_indices", idx_tensor, persistent=False)

        num_groups = 32
        while enc_channels % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=enc_channels, affine=False)

        self.mlp_1x1 = nn.Sequential(
            nn.Conv2d(num_actions, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
        )
        self.to_gamma_gate = nn.Conv2d(hidden_dim, enc_channels * 2, kernel_size=1)
        nn.init.zeros_(self.to_gamma_gate.weight)
        nn.init.zeros_(self.to_gamma_gate.bias)

        # bias=False guarantees zero DC color/brightness offset leak into decoder
        self.zero_proj = nn.Conv2d(enc_channels, dec_channels, kernel_size=1, bias=False)
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

        if self.struct_indices.numel() > 0:
            s_idx = self.struct_indices.to(a_local.device)
            a_struct = a_local.index_select(1, s_idx).amax(dim=1, keepdim=True)
        else:
            a_struct = a_local.amax(dim=1, keepdim=True)

        # Hard upper-bound retention envelope: -> 0.0 when structural sliders (blur/SR/banding/noise/JPEG) -> 1.0
        retain_env = (1.0 - self.suppression_strength * a_struct).clamp(0.0, 1.0).square()

        normed_skip = self.norm(enc_skip_f)
        h_act = self.mlp_1x1(a_local)
        raw_gamma, raw_gate = self.to_gamma_gate(h_act).chunk(2, dim=1)
        gamma = torch.tanh(raw_gamma)
        learned_gate = torch.sigmoid(raw_gate)

        gated_skip = retain_env * learned_gate * (normed_skip * (1.0 + gamma))
        bridge = self.zero_proj(gated_skip) * keep_mask.to(dtype=torch.float32, device=dec_feat.device)
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

        # Inject lightweight Dual-LoRA into VAE Encoder & Decoder (Pix2Pix-Turbo / S3Diff adaptive VAE)
        enc_rank = getattr(config, "VAE_ENC_LORA_RANK", 8)
        enc_alpha = getattr(config, "VAE_ENC_LORA_ALPHA", 8.0)
        enc_pix_params, enc_sem_params = inject_generator_dual_lora(
            self.vae.encoder,
            action_dim=config.ACTION_HIDDEN_DIM,
            rank=enc_rank,
            alpha=enc_alpha,
            include_conv=True,
        )
        dec_rank = getattr(config, "VAE_DEC_LORA_RANK", enc_rank)
        dec_alpha = getattr(config, "VAE_DEC_LORA_ALPHA", enc_alpha)
        dec_pix_params, dec_sem_params = inject_generator_dual_lora(
            self.vae.decoder,
            action_dim=config.ACTION_HIDDEN_DIM,
            rank=dec_rank,
            alpha=dec_alpha,
            include_conv=True,
        )
        self.vae_pix_lora_params = list(enc_pix_params) + list(dec_pix_params)
        self.vae_sem_lora_params = list(enc_sem_params) + list(dec_sem_params)

        # Dynamically probe Encoder & Decoder intermediate channel dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            z_dummy, enc_skips = self._raw_encode(dummy)
            dec_channels = self._probe_decoder_channels(z_dummy)

        num_actions = config.NUM_ACTIONS
        mod_hidden = getattr(config, "SPATIAL_MOD_HIDDEN_DIM", 64)
        self.skip_dropout_prob = float(getattr(config, "SKIP_DROPOUT_PROB", 0.50))

        deg_names = list(getattr(config, "DEGRADATION_TYPES", []))
        struct_indices = [
            i for i in range(num_actions)
            if i >= len(deg_names) or deg_names[i] not in _RADIOMETRIC_ONLY_NAMES
        ]

        # Multi-scale mid/coarse bridges ONLY ([H/8, H/4, H/2] — NO raw HxW conv_in pixel-copy bypass!):
        # enc_skips = [skip_2x (H/2), skip_4x (H/4), skip_8x (H/8)]
        # dec_channels = [pre_up0 (H/8), pre_up1 (H/4), pre_up2 (H/2)]
        self.up_skip_bridges = nn.ModuleList(
            [
                _ActionGatedSkipBridge(
                    enc_skips[2].shape[1], dec_channels[0], num_actions, struct_indices,
                    suppression_strength=0.85, hidden_dim=mod_hidden,
                ),
                _ActionGatedSkipBridge(
                    enc_skips[1].shape[1], dec_channels[1], num_actions, struct_indices,
                    suppression_strength=1.00, hidden_dim=mod_hidden,
                ),
                _ActionGatedSkipBridge(
                    enc_skips[0].shape[1], dec_channels[2], num_actions, struct_indices,
                    suppression_strength=1.00, hidden_dim=mod_hidden,
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
                skips.append(h)          # skip[0]: H/2, skip[1]: H/4, skip[2]: H/8

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
        """Encode RGB image in [0, 1] into scaled latents + 3 multi-scale skip tensors (H/2, H/4, H/8)."""
        x_norm = rgb_01 * 2.0 - 1.0
        return self._raw_encode(x_norm, collect_skips=True)

    def encode_target_latents(self, rgb_01: torch.Tensor) -> torch.Tensor:
        """Encode clean/target RGB image in [0, 1] without allocating skip activations."""
        x_norm = rgb_01 * 2.0 - 1.0
        latents, _ = self._raw_encode(x_norm, collect_skips=False)
        return latents

    def decode_with_skips(
        self,
        latents: torch.Tensor,
        enc_skips: List[torch.Tensor],
        action_map: torch.Tensor,
        skip_scale: float = 1.0,
        release_skips: bool = False,
    ) -> torch.Tensor:
        """Decode latents back to [0, 1] RGB using action-suppressed skips & DDP-safe skip dropout."""
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

        # Reverse order matches [skip_8x (2), skip_4x (1), skip_2x (0)]
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
