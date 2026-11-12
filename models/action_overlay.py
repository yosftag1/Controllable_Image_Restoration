"""Action Embedding & Pointwise 1x1 Spatial Overlay Modulator ("Pixel Overlay Sheet").

Implements:
1. PointwiseSpatialModulator: Strictly 1x1 convolutional SPADE-style modulation
   with zero spatial receptive field so whole-image, single-object, and multi-region
   overlays look locally identical to every pixel.
2. ActionOverlayEncoder: Encodes the spatial action overlay A in [0, 1]^(B x K x H x W)
   and Vision Semantic features into:
   - Cross-Attention tokens [B, N_sem + K, cross_attn_dim]
   - Global Action Conditioning vector [B, action_hidden_dim] (combining spatial mean & max
     so small localized edits still activate full LoRA capacity!)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousScalarFourier(nn.Module):
    """Sinusoidal Fourier features for continuous slider values a_k in [0, 1]."""

    def __init__(self, dim: int = 32):
        super().__init__()
        half = max(1, dim // 2)
        freqs = torch.exp(torch.linspace(0.0, math.log(100.0), half))
        self.register_buffer("freqs", freqs)
        self.out_dim = half * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: arbitrary shape [...] -> [..., out_dim]
        args = x.unsqueeze(-1).float() * self.freqs * math.pi
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class PointwiseSpatialModulator(nn.Module):
    """1x1-only spatial action modulator (zero receptive field across neighboring pixels)."""

    def __init__(self, num_actions: int, channels: int, hidden_dim: int = 64):
        super().__init__()
        num_groups = 32
        while channels % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, affine=False)

        # Strictly kernel_size=1 so each spatial coordinate (y, x) only reads its own tag!
        self.mlp_1x1 = nn.Sequential(
            nn.Conv2d(num_actions, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
        )
        self.to_gamma_beta = nn.Conv2d(hidden_dim, channels * 2, kernel_size=1)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, feat: torch.Tensor, action_map: torch.Tensor) -> torch.Tensor:
        """Modulate `feat` [B, C, H_f, W_f] using `action_map` [B, K, H, W]."""
        target_hw = feat.shape[-2:]
        if action_map.shape[-2:] != target_hw:
            a_local = F.interpolate(action_map.float(), size=target_hw, mode="area")
        else:
            a_local = action_map.float()

        h = self.mlp_1x1(a_local)
        gamma, beta = self.to_gamma_beta(h).chunk(2, dim=1)
        normed = self.norm(feat.float())
        modulated = feat.float() + gamma * normed + beta
        return modulated.to(feat.dtype)


class ActionOverlayEncoder(nn.Module):
    """Fuses spatial action overlays and semantic vision features into generator tokens."""

    def __init__(
        self,
        num_actions: int,
        sem_spatial_dim: int,
        sem_global_dim: int,
        cross_attn_dim: int,
        action_hidden_dim: int = 128,
        tokens_per_action: int = 4,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.cross_attn_dim = cross_attn_dim
        self.action_hidden_dim = action_hidden_dim
        self.tokens_per_action = max(1, int(tokens_per_action))

        # Per-operation learned identity basis + preserve/active anchors
        self.op_identity = nn.Parameter(torch.randn(num_actions, action_hidden_dim) * 0.02)
        self.preserve_anchor = nn.Parameter(torch.randn(1, 1, action_hidden_dim) * 0.02)
        self.active_anchor = nn.Parameter(torch.randn(1, 1, action_hidden_dim) * 0.02)

        self.scalar_fourier = ContinuousScalarFourier(dim=32)
        in_tok_dim = action_hidden_dim + self.scalar_fourier.out_dim * 2 + 2

        # Dedicated per-slider encoder so the 8 sliders are never drowned out by high-dim vision vectors
        self.op_feat_mlp = nn.Sequential(
            nn.Linear(in_tok_dim, action_hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(action_hidden_dim * 2, action_hidden_dim),
            nn.LayerNorm(action_hidden_dim),
        )

        # Expand each operation into `tokens_per_action` salient cross-attention register tokens (e.g. 8*4 = 32 tokens)
        self.op_token_proj = nn.Sequential(
            nn.Linear(action_hidden_dim, cross_attn_dim * self.tokens_per_action),
        )
        self.op_token_norm = nn.LayerNorm(cross_attn_dim)

        # Project clean-aligned semantic vision patch tokens into UNet cross-attention space
        self.sem_token_proj = nn.Sequential(
            nn.LayerNorm(sem_spatial_dim),
            nn.Linear(sem_spatial_dim, cross_attn_dim),
            nn.GELU(),
            nn.Linear(cross_attn_dim, cross_attn_dim),
            nn.LayerNorm(cross_attn_dim),
        )

        # Balanced Global Conditioning: dedicated slider highway + normalized semantic context
        self.action_summary_mlp = nn.Sequential(
            nn.Linear(num_actions * action_hidden_dim, action_hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(action_hidden_dim * 2, action_hidden_dim),
            nn.LayerNorm(action_hidden_dim),
        )
        self.sem_global_proj = nn.Sequential(
            nn.LayerNorm(sem_global_dim),
            nn.Linear(sem_global_dim, action_hidden_dim),
            nn.SiLU(),
        )
        self.global_fusion_mlp = nn.Sequential(
            nn.Linear(action_hidden_dim * 2, action_hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(action_hidden_dim * 2, action_hidden_dim),
        )

    @staticmethod
    def ensure_spatial_map(action_input: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Accept either [B, K] global vector or [B, K, H, W] spatial map and return [B, K, H, W]."""
        if action_input.ndim == 2:
            return action_input.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width).contiguous()
        if action_input.ndim == 4:
            if action_input.shape[-2:] != (height, width):
                return F.interpolate(action_input.float(), size=(height, width), mode="bilinear", align_corners=False)
            return action_input
        raise ValueError(f"Expected action_input to have 2 or 4 dims, got shape {tuple(action_input.shape)}")

    def forward(
        self,
        action_map: torch.Tensor,
        sem_tokens: torch.Tensor,
        sem_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cross-attention conditioning tokens and global Dual-LoRA action vector.

        Args:
            action_map: [B, K, H, W] float tensor in [0, 1]
            sem_tokens: [B, N_p, D_spat] from SemanticEncoder
            sem_global: [B, D_glob] from SemanticEncoder

        Returns:
            cross_attn_tokens: [B, N_p + K, cross_attn_dim]
            global_action_cond: [B, action_hidden_dim]
        """
        bsz = action_map.shape[0]
        a_fp32 = action_map.float()

        # Both spatial mean and spatial max so localized small-mask edits keep full strength!
        a_mean = a_fp32.mean(dim=(-2, -1))          # [B, K]
        a_max = a_fp32.amax(dim=(-2, -1))           # [B, K]

        # Smooth active gate in [0, 1] distinguishing 0.0 ("preserve") from > 0.0 ("edit")
        active_gate = (a_max / 0.05).clamp(0.0, 1.0).unsqueeze(-1)  # [B, K, 1]
        status_vec = (1.0 - active_gate) * self.preserve_anchor + active_gate * self.active_anchor
        basis = self.op_identity.unsqueeze(0).expand(bsz, -1, -1) + status_vec  # [B, K, H_act]

        f_mean = self.scalar_fourier(a_mean)        # [B, K, F]
        f_max = self.scalar_fourier(a_max)          # [B, K, F]
        raw_stats = torch.stack([a_mean, a_max], dim=-1)  # [B, K, 2]

        op_in = torch.cat([basis, f_mean, f_max, raw_stats], dim=-1)
        op_feat = self.op_feat_mlp(op_in)           # [B, K, action_hidden_dim]

        op_tokens = self.op_token_proj(op_feat).view(
            bsz, self.num_actions * self.tokens_per_action, self.cross_attn_dim
        )
        op_tokens = self.op_token_norm(op_tokens)   # [B, K * tokens_per_action, cross_attn_dim]

        vis_tokens = self.sem_token_proj(sem_tokens.float())  # [B, N_p, cross_attn_dim]
        cross_attn_tokens = torch.cat([vis_tokens, op_tokens], dim=1)

        # Direct residual highway from the 8 sliders so image content never drowns out slider values
        action_vec = self.action_summary_mlp(op_feat.reshape(bsz, -1))
        sem_vec = self.sem_global_proj(sem_global.float())
        global_action_cond = action_vec + self.global_fusion_mlp(torch.cat([action_vec, sem_vec], dim=-1))

        return cross_attn_tokens, global_action_cond
