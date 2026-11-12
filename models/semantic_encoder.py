"""Pretrained Vision Foundation Semantic Encoders with Self-Teacher LoRA.

Supports:
- "c_radio":        NVIDIA C-RADIOv2-B / C-RADIOv3-B (unified DINO + SigLIP + SAM distillation)
- "dinov3_siglip2": Meta DINOv3 / DINOv2-Registers (geometry) + Google SigLIP2-Base (appearance)
- "mock_tiny":      Lightweight local ViT-patch encoder for fast offline unit testing
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_layers import inject_vision_lora, make_module_dataparallel_safe, vision_lora_context


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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.patch_embed(x)  # [B, D, Hp, Wp]
        tokens = feat.flatten(2).transpose(1, 2)  # [B, N, D]
        for blk in self.blocks:
            h = blk["norm1"](tokens)
            q = blk["q_proj"](h)
            k = blk["k_proj"](h)
            v = blk["v_proj"](h)
            attn = F.scaled_dot_product_attention(q, k, v)
            tokens = tokens + blk["out_proj"](attn)
            tokens = tokens + blk["fc2"](F.gelu(blk["fc1"](blk["norm2"](tokens))))
        tokens = self.norm_out(tokens)
        summary = tokens.mean(dim=1)
        return summary, tokens


class SemanticEncoder(nn.Module):
    """Unified Vision Semantic Encoder with zero-extra-VRAM Self-Teacher LoRA switch."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_type = config.SEMANTIC_ENCODER_TYPE.lower()
        self.patch_size = 16
        self.dino_patch_size = 16

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        rank = getattr(config, "VISION_LORA_RANK", 8)
        alpha = getattr(config, "VISION_LORA_ALPHA", 16.0)
        dropout = getattr(config, "VISION_LORA_DROPOUT", 0.0)
        self.vision_lora_params: List[nn.Parameter] = []

        if self.encoder_type == "c_radio":
            from transformers import AutoConfig, AutoModel

            radio_id = getattr(config, "RADIO_MODEL_ID", "nvidia/C-RADIOv3-B")
            if radio_id == "nvidia/C-RADIOv2.5-B":
                radio_id = "nvidia/C-RADIOv3-B"
            alt_radio_id = getattr(config, "RADIO_ALT_MODEL_ID", "nvidia/C-RADIOv2-B")

            try:
                try:
                    radio_cfg = load_hf_public_safe(AutoConfig.from_pretrained, radio_id, trust_remote_code=True)
                    chosen_id = radio_id
                except Exception:
                    radio_cfg = load_hf_public_safe(AutoConfig.from_pretrained, alt_radio_id, trust_remote_code=True)
                    chosen_id = alt_radio_id

                # Disable internal bfloat16 autocast in RADIO config so outer PyTorch FP16 AMP runs natively on T4 GPUs
                if isinstance(getattr(radio_cfg, "args", None), dict):
                    radio_cfg.args["amp"] = False

                self.radio = load_hf_public_safe(
                    AutoModel.from_pretrained,
                    chosen_id,
                    config=radio_cfg,
                    trust_remote_code=True,
                )
                self.patch_size = int(getattr(self.radio, "patch_size", 16))
                self.radio.eval()
                self.vision_lora_params.extend(
                    inject_vision_lora(self.radio, rank=rank, alpha=alpha, dropout=dropout)
                )
                make_module_dataparallel_safe(self.radio)
                print(f"[SemanticEncoder] Loaded {chosen_id} (patch_size={self.patch_size})")
            except ModuleNotFoundError as mod_err:
                print(
                    f"[Warning] {chosen_id} remote code requires '{mod_err.name}' "
                    f"(install via: `!pip install -q open-clip-torch`). "
                    f"Falling back automatically to public DINOv2-Registers + SigLIP2!"
                )
                self.encoder_type = "dinov3_siglip2"
                config.SEMANTIC_ENCODER_TYPE = "dinov3_siglip2"

        if self.encoder_type == "dinov3_siglip2":
            from transformers import AutoModel

            dino_primary = getattr(config, "DINOV3_MODEL_ID", "facebook/dinov3-vits16-pretrain-lvd1689m")
            dino_fallback = getattr(config, "DINOV2_FALLBACK_MODEL_ID", "facebook/dinov2-with-registers-small")
            try:
                self.dinov3 = load_hf_public_safe(AutoModel.from_pretrained, dino_primary)
                used_dino = dino_primary
            except Exception:
                print(
                    f"[SemanticEncoder] '{dino_primary}' requires gated HF login; "
                    f"using public ungated '{dino_fallback}' instead."
                )
                self.dinov3 = load_hf_public_safe(AutoModel.from_pretrained, dino_fallback)
                used_dino = dino_fallback

            siglip_id = getattr(config, "SIGLIP2_MODEL_ID", "google/siglip2-base-patch16-224")
            siglip_full = load_hf_public_safe(AutoModel.from_pretrained, siglip_id)
            self.siglip2 = getattr(siglip_full, "vision_model", siglip_full)
            self.dino_patch_size = int(getattr(getattr(self.dinov3, "config", None), "patch_size", 16))
            self.dinov3.eval()
            self.siglip2.eval()

            self.vision_lora_params.extend(
                inject_vision_lora(self.dinov3, rank=rank, alpha=alpha, dropout=dropout)
            )
            self.vision_lora_params.extend(
                inject_vision_lora(self.siglip2, rank=rank, alpha=alpha, dropout=dropout)
            )
            make_module_dataparallel_safe(self.dinov3)
            make_module_dataparallel_safe(self.siglip2)
            print(f"[SemanticEncoder] Loaded {used_dino} (patch={self.dino_patch_size}) + {siglip_id}")

        elif self.encoder_type == "mock_tiny":
            self.mock = _MockTinyVisionBackbone(embed_dim=192, patch_size=16, depth=3)
            self.mock.eval()
            self.vision_lora_params.extend(
                inject_vision_lora(self.mock, rank=rank, alpha=alpha, dropout=dropout)
            )
            make_module_dataparallel_safe(self.mock)
        elif self.encoder_type != "c_radio":
            raise ValueError(
                f"Unsupported SEMANTIC_ENCODER_TYPE='{config.SEMANTIC_ENCODER_TYPE}'. "
                "Expected 'c_radio', 'dinov3_siglip2', or 'mock_tiny'."
            )

        # Dynamically probe output feature dimensions
        probe_hw = 64 if self.encoder_type == "mock_tiny" else 256
        with torch.no_grad(), vision_lora_context(enabled=False):
            dummy = torch.zeros(1, 3, probe_hw, probe_hw)
            glob, spat_tokens, _ = self._forward_raw(dummy)
            self.global_dim = int(glob.shape[-1])
            self.spatial_dim = int(spat_tokens.shape[-1])

    def train(self, mode: bool = True):
        """Keep frozen pretrained vision backbones in eval() mode (protecting norm/dropout stats)."""
        super().train(mode)
        if hasattr(self, "radio"):
            self.radio.eval()
        if hasattr(self, "dinov3"):
            self.dinov3.eval()
        if hasattr(self, "siglip2"):
            self.siglip2.eval()
        if hasattr(self, "mock"):
            self.mock.eval()
        return self

    def _pad_or_resize_to_patch_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        h, w = x.shape[-2:]
        ps = self.patch_size
        nh = max(ps, int(round(h / ps)) * ps)
        nw = max(ps, int(round(w / ps)) * ps)
        if self.encoder_type == "c_radio" and hasattr(self, "radio") and hasattr(self.radio, "get_nearest_supported_resolution"):
            nh, nw = self.radio.get_nearest_supported_resolution(nh, nw)
            nh, nw = int(nh), int(nw)
        if nh != h or nw != w:
            x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        return x, nh // ps, nw // ps

    def _forward_raw(self, rgb_01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Internal forward expecting rgb_01 in [0, 1]. Returns (global_vec, spatial_tokens, spatial_grid)."""
        rgb_01 = rgb_01.clamp(0.0, 1.0)
        x_aligned, hp, wp = self._pad_or_resize_to_patch_multiple(rgb_01)
        num_patches = hp * wp

        if self.encoder_type == "c_radio":
            # C-RADIO expects raw [0, 1] RGB tensors (internal input_conditioner handles normalization)
            out = self.radio(x_aligned)
            if isinstance(out, (tuple, list)):
                summary, spatial_tokens = out[0], out[1]
            else:
                summary = getattr(out, "pooler_output", None)
                spatial_tokens = getattr(out, "last_hidden_state", out)
            if summary is None:
                summary = spatial_tokens.mean(dim=1)
            if summary.ndim == 3:
                summary = summary.mean(dim=1)
            spatial_tokens = spatial_tokens[:, -num_patches:, :]

        elif self.encoder_type == "dinov3_siglip2":
            # 1. DINOv3 / DINOv2-Registers: ImageNet normalized; match exact (hp, wp) token grid
            x_dino = (x_aligned - self.imagenet_mean) / self.imagenet_std
            dino_h, dino_w = hp * self.dino_patch_size, wp * self.dino_patch_size
            if x_dino.shape[-2:] != (dino_h, dino_w):
                x_dino = F.interpolate(x_dino, size=(dino_h, dino_w), mode="bilinear", align_corners=False)
            dino_out = self.dinov3(pixel_values=x_dino)
            dino_seq = dino_out.last_hidden_state
            # Slice trailing H_p * W_p patch tokens (safely skips CLS + register tokens)
            dino_spatial = dino_seq[:, -num_patches:, :]
            dino_global = getattr(dino_out, "pooler_output", None)
            if dino_global is None:
                dino_global = dino_seq[:, 0, :]

            # 2. SigLIP2: [-1, 1] normalized; resize to 224x224 if dynamic pos interpolation isn't enabled
            x_sig = (x_aligned - 0.5) / 0.5
            try:
                sig_out = self.siglip2(pixel_values=x_sig, interpolate_pos_encoding=True)
                sig_spatial = sig_out.last_hidden_state[:, -num_patches:, :]
                if sig_spatial.shape[1] != num_patches:
                    raise TypeError("Fallback to explicit grid resize")
            except TypeError:
                x_sig_224 = F.interpolate(x_sig, size=(224, 224), mode="bilinear", align_corners=False)
                sig_out = self.siglip2(pixel_values=x_sig_224)
                sig_seq = sig_out.last_hidden_state
                side = int(sig_seq.shape[1] ** 0.5)
                sig_grid_224 = sig_seq[:, -side * side :, :].transpose(1, 2).reshape(
                    rgb_01.shape[0], -1, side, side
                )
                sig_grid_resized = F.interpolate(sig_grid_224, size=(hp, wp), mode="bilinear", align_corners=False)
                sig_spatial = sig_grid_resized.flatten(2).transpose(1, 2)

            sig_global = getattr(sig_out, "pooler_output", None)
            if sig_global is None:
                sig_global = sig_spatial.mean(dim=1)

            summary = torch.cat([dino_global, sig_global], dim=-1)
            spatial_tokens = torch.cat([dino_spatial, sig_spatial], dim=-1)

        else:
            summary, spatial_tokens = self.mock(x_aligned)

        spatial_grid = spatial_tokens.transpose(1, 2).reshape(rgb_01.shape[0], -1, hp, wp)
        return summary, spatial_tokens, spatial_grid

    def forward(self, rgb_01: torch.Tensor, use_lora: bool = True) -> Dict[str, torch.Tensor]:
        """Extract semantic features with LoRA switch enabled (degraded input) or disabled (clean teacher)."""
        with vision_lora_context(enabled=use_lora):
            global_vec, spatial_tokens, spatial_grid = self._forward_raw(rgb_01)
        return {
            "global": global_vec,
            "tokens": spatial_tokens,
            "grid": spatial_grid,
        }

    def compute_self_teacher_loss(
        self, src_01: torch.Tensor, clean_01: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Align degraded student features (LoRA ON) to clean teacher features (LoRA OFF, no_grad).

        Returns:
            (align_loss, student_features_dict)
        """
        with torch.no_grad():
            teacher = self.forward(clean_01, use_lora=False)

        student = self.forward(src_01, use_lora=True)

        # Token-wise cosine alignment + global cosine alignment
        s_tok = F.normalize(student["tokens"].float(), dim=-1)
        t_tok = F.normalize(teacher["tokens"].float().detach(), dim=-1)
        loss_spatial = (1.0 - (s_tok * t_tok).sum(dim=-1)).mean()

        s_glob = F.normalize(student["global"].float(), dim=-1)
        t_glob = F.normalize(teacher["global"].float().detach(), dim=-1)
        loss_global = (1.0 - (s_glob * t_glob).sum(dim=-1)).mean()

        align_loss = 0.7 * loss_spatial + 0.3 * loss_global
        return align_loss, student
