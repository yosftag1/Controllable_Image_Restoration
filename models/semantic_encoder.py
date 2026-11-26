"""Frozen Pretrained Vision Semantic Encoder (DINOv2-with-Registers & Offline Mock).

Provides clean semantic patch grids for:
1. Cross-attention image context in `ActionOverlayEncoder`
2. Zero-shot 1-click object mask extraction (`extract_one_click_semantic_mask`)
"""

import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_hf_public_safe(loader_fn, repo_id: str, **kwargs):
    """Load a Hugging Face model/config, retrying anonymously (token=False) if an env token is stale."""
    try:
        return loader_fn(repo_id, **kwargs)
    except Exception as err:
        msg = str(err)
        if ("401" in msg or "Invalid username or password" in msg) and "token" not in kwargs:
            return loader_fn(repo_id, token=False, **kwargs)
        raise


class _MockTinyVisionBackbone(nn.Module):
    """Compact ViT-style patch encoder for offline verification without HF downloads."""

    def __init__(self, embed_dim: int = 192, patch_size: int = 16, depth: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(embed_dim),
                        "q_proj": nn.Linear(embed_dim, embed_dim),
                        "k_proj": nn.Linear(embed_dim, embed_dim),
                        "v_proj": nn.Linear(embed_dim, embed_dim),
                        "out_proj": nn.Linear(embed_dim, embed_dim),
                        "norm2": nn.LayerNorm(embed_dim),
                        "fc1": nn.Linear(embed_dim, embed_dim * 2),
                        "fc2": nn.Linear(embed_dim * 2, embed_dim),
                    }
                )
                for _ in range(depth)
            ]
        )
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        feat = self.patch_embed(x)  # [B, D, Hp, Wp]
        tokens = feat.flatten(2).transpose(1, 2)  # [B, N, D]
        intermediates = []
        for blk in self.blocks:
            h = blk["norm1"](tokens)
            q = blk["q_proj"](h)
            k = blk["k_proj"](h)
            v = blk["v_proj"](h)
            attn = F.scaled_dot_product_attention(q, k, v)
            tokens = tokens + blk["out_proj"](attn)
            tokens = tokens + blk["fc2"](F.gelu(blk["fc1"](blk["norm2"](tokens))))
            if return_intermediates:
                intermediates.append(self.norm_out(tokens))
        tokens = self.norm_out(tokens)
        summary = tokens.mean(dim=1)
        if return_intermediates:
            return summary, tokens, intermediates
        return summary, tokens


class SemanticEncoder(nn.Module):
    """Frozen DINOv2-with-Registers (or mock_tiny) Vision Semantic Encoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_type = config.SEMANTIC_ENCODER_TYPE.lower()
        self.patch_size = 16

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if self.encoder_type in ("dinov2_reg", "dinov2"):
            self.encoder_type = "dinov2_reg"
            from transformers import AutoModel

            dino_primary = getattr(config, "DINOV2_REG_MODEL_ID", "facebook/dinov2-with-registers-base")
            dino_fallback = getattr(config, "DINOV2_FALLBACK_MODEL_ID", "facebook/dinov2-with-registers-small")
            try:
                self.dinov2_reg = load_hf_public_safe(AutoModel.from_pretrained, dino_primary)
                used_dino = dino_primary
            except Exception:
                self.dinov2_reg = load_hf_public_safe(AutoModel.from_pretrained, dino_fallback)
                used_dino = dino_fallback

            for p in self.dinov2_reg.parameters():
                p.requires_grad = False
            self.dinov2_reg.eval()
            self.patch_size = int(getattr(getattr(self.dinov2_reg, "config", None), "patch_size", 14))
            if os.environ.get("RANK", "0") == "0":
                print(f"[SemanticEncoder] Loaded frozen {used_dino} (patch={self.patch_size})", flush=True)

        elif self.encoder_type == "mock_tiny":
            self.mock = _MockTinyVisionBackbone(embed_dim=192, patch_size=16, depth=3)
            for p in self.mock.parameters():
                p.requires_grad = False
            self.mock.eval()
            self.patch_size = 16
        else:
            raise ValueError(
                f"Unsupported SEMANTIC_ENCODER_TYPE='{config.SEMANTIC_ENCODER_TYPE}'. "
                "Expected 'dinov2_reg' or 'mock_tiny'."
            )

        probe_hw = 64 if self.encoder_type == "mock_tiny" else 256
        with torch.no_grad():
            dummy = torch.zeros(1, 3, probe_hw, probe_hw)
            out = self.forward(dummy)
            self.global_dim = int(out["global"].shape[-1])
            self.spatial_dim = int(out["tokens"].shape[-1])

    def train(self, mode: bool = True):
        """Keep frozen pretrained vision backbone strictly in eval() mode."""
        super().train(mode)
        for attr in ("dinov2_reg", "mock"):
            if hasattr(self, attr):
                getattr(self, attr).eval()
        return self

    def _resize_to_patch_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        h, w = x.shape[-2:]
        ps = self.patch_size
        nh = max(ps, int(round(h / ps)) * ps)
        nw = max(ps, int(round(w / ps)) * ps)
        if nh != h or nw != w:
            x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        return x, nh // ps, nw // ps

    @torch.no_grad()
    def forward(self, rgb_01: torch.Tensor, use_lora: bool = False) -> Dict[str, torch.Tensor]:
        """Extract frozen semantic global vector, patch tokens, and 2D spatial grid."""
        del use_lora
        rgb_01 = rgb_01.clamp(0.0, 1.0)
        bsz = rgb_01.shape[0]
        x_aligned, hp, wp = self._resize_to_patch_multiple(rgb_01)
        num_patches = hp * wp

        if self.encoder_type == "dinov2_reg":
            x_norm = (x_aligned - self.imagenet_mean) / self.imagenet_std
            dino_out = self.dinov2_reg(pixel_values=x_norm)
            seq = dino_out.last_hidden_state.float()
            # Trailing `hp * wp` tokens skip [CLS] + register tokens cleanly
            spatial_tokens = seq[:, -num_patches:, :]
            global_vec = seq[:, 0, :]
        else:
            global_vec, spatial_tokens = self.mock(x_aligned)

        spatial_grid = spatial_tokens.transpose(1, 2).reshape(bsz, -1, hp, wp)
        return {
            "global": global_vec,
            "tokens": spatial_tokens,
            "grid": spatial_grid,
            "mask_grid": spatial_grid,
        }

    def extract_discriminator_grids(
        self,
        rgb_01: torch.Tensor,
        max_size: int = 252,
        enable_grad: bool = False,
    ) -> list:
        """Extract 3 multi-layer 2D spatial DINOv2 feature grids for the Stage-2 ADD/S3Diff discriminator.

        Reuses the already-loaded frozen DINOv2 backbone at zero extra VRAM while allowing input gradients
        (`enable_grad=True`) during Generator adversarial updates.
        """
        ps = self.patch_size
        x = rgb_01.clamp(0.0, 1.0)
        bsz, _, h, w = x.shape
        if max_size > 0 and max(h, w) > max_size:
            scale = float(max_size) / float(max(h, w))
            th = max(ps, int(round((h * scale) / ps)) * ps)
            tw = max(ps, int(round((w * scale) / ps)) * ps)
            x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
        else:
            x, _, _ = self._resize_to_patch_multiple(x)
        hp, wp = x.shape[-2] // ps, x.shape[-1] // ps
        num_patches = hp * wp

        with torch.set_grad_enabled(bool(enable_grad)):
            if self.encoder_type == "dinov2_reg":
                x_norm = (x - self.imagenet_mean.to(device=x.device, dtype=x.dtype)) / self.imagenet_std.to(
                    device=x.device, dtype=x.dtype
                )
                dino_out = self.dinov2_reg(pixel_values=x_norm, output_hidden_states=True)
                hidden_states = dino_out.hidden_states
                n_layers = len(hidden_states) - 1  # exclude patch embedding at index 0
                if n_layers >= 3:
                    pick = [max(1, n_layers // 3), max(2, (2 * n_layers) // 3), n_layers]
                else:
                    pick = list(range(1, len(hidden_states)))
                grids = []
                for li in pick:
                    seq = hidden_states[li][:, -num_patches:, :].float()
                    grids.append(seq.transpose(1, 2).reshape(bsz, -1, hp, wp))
                return grids
            else:
                _, _, intermediates = self.mock(x, return_intermediates=True)
                return [
                    seq.float().transpose(1, 2).reshape(bsz, -1, hp, wp)
                    for seq in intermediates[-3:]
                ]
