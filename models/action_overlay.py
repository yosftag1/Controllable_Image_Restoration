"""Action Embedding & Pointwise 1x1 Spatial Overlay Modulator (Signed + Unsigned Controls).

Implements:
1. PointwiseSpatialModulator: Strictly 1x1 convolutional SPADE-style modulation
   with zero spatial receptive field so whole-image, single-object, and multi-region
   overlays behave locally identically at every pixel.
2. ActionOverlayEncoder: Encodes signed/unsigned spatial action overlays
   A in [-1, 1]^(B x K x H x W) and Vision Semantic features into 77-token
   cross-attention conditioning preserving the base OpenCLIP prompt prefix.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousScalarFourier(nn.Module):
    """Odd/even sinusoidal Fourier features distinguishing signed [-1, +1] and unsigned [0, 1] sliders."""

    def __init__(self, dim: int = 32):
        super().__init__()
        half = max(1, dim // 2)
        freqs = torch.exp(torch.linspace(0.0, math.log(64.0), half))
        self.register_buffer("freqs", freqs)
        self.out_dim = half * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        """Modulate `feat` [B, C, H_f, W_f] using signed/unsigned `action_map` [B, K, H, W]."""
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
    """Anchors signed/unsigned slider & semantic tokens onto SD-2.1's 77-token OpenCLIP manifold."""

    def __init__(
        self,
        num_actions: int,
        sem_spatial_dim: int,
        sem_global_dim: int,
        cross_attn_dim: int,
        action_hidden_dim: int = 128,
        tokens_per_action: int = 4,
        prompt_seq_len: int = 77,
        base_prompt_embeds: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.cross_attn_dim = cross_attn_dim
        self.action_hidden_dim = action_hidden_dim

        if base_prompt_embeds is not None:
            prompt_seq_len = int(base_prompt_embeds.shape[1])
            embed_buf = base_prompt_embeds.detach().clone().float()
        else:
            prompt_seq_len = max(num_actions + 4, int(prompt_seq_len))
            embed_buf = torch.zeros(1, prompt_seq_len, cross_attn_dim, dtype=torch.float32)

        self.prompt_seq_len = prompt_seq_len
        max_op_budget = (self.prompt_seq_len - 33) if self.prompt_seq_len >= 64 else (self.prompt_seq_len - 4)
        self.tokens_per_action = min(
            max(1, int(tokens_per_action)),
            max(1, max_op_budget // max(1, num_actions)),
        )
        self.op_token_count = self.num_actions * self.tokens_per_action
        avail_after_ops = max(1, self.prompt_seq_len - self.op_token_count - 1)
        self.vis_spatial_count = min(16, avail_after_ops)
        self.prefix_keep = max(0, self.prompt_seq_len - (self.vis_spatial_count + 1 + self.op_token_count))
        self.register_buffer("base_prompt_embeds", embed_buf, persistent=False)

        self.op_identity = nn.Parameter(torch.randn(num_actions, action_hidden_dim) * 0.02)
        self.preserve_anchor = nn.Parameter(torch.randn(1, 1, action_hidden_dim) * 0.02)
        self.pos_active_anchor = nn.Parameter(torch.randn(1, 1, action_hidden_dim) * 0.02)
        self.neg_active_anchor = nn.Parameter(torch.randn(1, 1, action_hidden_dim) * 0.02)

        self.scalar_fourier = ContinuousScalarFourier(dim=32)
        in_tok_dim = action_hidden_dim + self.scalar_fourier.out_dim * 2 + 5

        self.op_feat_mlp = nn.Sequential(
            nn.Linear(in_tok_dim, action_hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(action_hidden_dim * 2, action_hidden_dim),
            nn.LayerNorm(action_hidden_dim),
        )

        self.op_token_proj = nn.Linear(action_hidden_dim, cross_attn_dim * self.tokens_per_action)
        self.op_token_norm = nn.LayerNorm(cross_attn_dim)
        self.op_token_out = nn.Linear(cross_attn_dim, cross_attn_dim)
        nn.init.normal_(self.op_token_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.op_token_out.bias)

        self.sem_token_proj = nn.Sequential(
            nn.LayerNorm(sem_spatial_dim),
            nn.Linear(sem_spatial_dim, cross_attn_dim),
            nn.GELU(),
            nn.LayerNorm(cross_attn_dim),
        )
        self.sem_token_out = nn.Linear(cross_attn_dim, cross_attn_dim)
        nn.init.normal_(self.sem_token_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.sem_token_out.bias)

        self.sem_global_token_proj = nn.Sequential(
            nn.LayerNorm(sem_global_dim),
            nn.Linear(sem_global_dim, cross_attn_dim),
            nn.GELU(),
            nn.LayerNorm(cross_attn_dim),
        )
        self.sem_global_token_out = nn.Linear(cross_attn_dim, cross_attn_dim)
        nn.init.normal_(self.sem_global_token_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.sem_global_token_out.bias)

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
    ) -> torch.Tensor:
        """Compute 77-token cross-attention conditioning [B, 77, cross_attn_dim]."""
        bsz = action_map.shape[0]
        a_fp32 = action_map.float()

        a_mean = a_fp32.mean(dim=(-2, -1))
        a_max = a_fp32.amax(dim=(-2, -1))
        a_min = a_fp32.amin(dim=(-2, -1))
        a_abs_max = torch.maximum(a_max.abs(), a_min.abs())
        a_peak = torch.where(a_max.abs() >= a_min.abs(), a_max, a_min)

        active_gate = (a_abs_max / 0.05).clamp(0.0, 1.0).unsqueeze(-1)
        sign_blend = (0.5 * (torch.tanh(a_peak * 6.0) + 1.0)).unsqueeze(-1)
        dir_anchor = sign_blend * self.pos_active_anchor + (1.0 - sign_blend) * self.neg_active_anchor
        status_vec = (1.0 - active_gate) * self.preserve_anchor + active_gate * dir_anchor
        basis = self.op_identity.unsqueeze(0).expand(bsz, -1, -1) + status_vec

        f_mean = self.scalar_fourier(a_mean)
        f_peak = self.scalar_fourier(a_peak)
        raw_stats = torch.stack([a_mean, a_peak, a_min, a_max, a_abs_max], dim=-1)

        op_in = torch.cat([basis, f_mean, f_peak, raw_stats], dim=-1)
        op_feat = self.op_feat_mlp(op_in)

        op_tokens = self.op_token_proj(op_feat).view(bsz, self.op_token_count, self.cross_attn_dim)
        op_tokens = self.op_token_out(F.silu(self.op_token_norm(op_tokens)))

        n_p = sem_tokens.shape[1]
        side = int(math.isqrt(n_p))
        target_side = int(math.isqrt(self.vis_spatial_count))
        if side * side == n_p and target_side * target_side == self.vis_spatial_count:
            grid_2d = sem_tokens.float().transpose(1, 2).reshape(bsz, -1, side, side)
            pooled_vis = F.adaptive_avg_pool2d(grid_2d, (target_side, target_side)).flatten(2).transpose(1, 2)
        else:
            pooled_vis = F.adaptive_avg_pool1d(sem_tokens.float().transpose(1, 2), self.vis_spatial_count).transpose(1, 2)

        vis_spatial_tokens = self.sem_token_out(self.sem_token_proj(pooled_vis))
        vis_global_token = self.sem_global_token_out(
            self.sem_global_token_proj(sem_global.float().unsqueeze(1))
        )

        tail_deltas = torch.cat([vis_spatial_tokens, vis_global_token, op_tokens], dim=1)
        if self.prefix_keep > 0:
            zero_prefix = tail_deltas.new_zeros((bsz, self.prefix_keep, self.cross_attn_dim))
            delta_tokens = torch.cat([zero_prefix, tail_deltas], dim=1)
        else:
            delta_tokens = tail_deltas

        base_prompt = self.base_prompt_embeds
        if base_prompt.device != delta_tokens.device or base_prompt.dtype != delta_tokens.dtype:
            base_prompt = base_prompt.to(device=delta_tokens.device, dtype=delta_tokens.dtype)
        return base_prompt.expand(bsz, -1, -1) + delta_tokens
