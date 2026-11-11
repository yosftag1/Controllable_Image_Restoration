"""Loss functions for Action-Conditioned Dual-LoRA Controllable Restoration."""

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiLayerVGGPerceptual(nn.Module):
    """Multi-scale VGG19 perceptual feature loss (safe under offline fallback)."""

    LAYER_MAP = {
        "relu1_2": 3,
        "relu2_2": 8,
        "relu3_4": 17,
        "relu4_4": 26,
        "relu5_4": 35,
    }

    def __init__(self, layers: Iterable[str] = ("relu2_2", "relu3_4", "relu4_4"), use_pretrained: bool = True):
        super().__init__()
        self.target_indices = sorted(self.LAYER_MAP[name] for name in layers)
        max_idx = max(self.target_indices)

        try:
            weights = models.VGG19_Weights.DEFAULT if use_pretrained else None
            vgg = models.vgg19(weights=weights)
        except Exception:
            vgg = models.vgg19(weights=None)

        self.slices = nn.ModuleList()
        prev = 0
        for idx in self.target_indices:
            self.slices.append(nn.Sequential(*list(vgg.features.children())[prev : idx + 1]))
            prev = idx + 1

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred_01: torch.Tensor, target_01: torch.Tensor, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = (pred_01.float() - self.mean) / self.std
        y = (target_01.float() - self.mean) / self.std

        total = pred_01.new_zeros((), dtype=torch.float32)
        for block in self.slices:
            x = block(x)
            with torch.no_grad():
                y = block(y)
            diff = F.l1_loss(x, y.detach(), reduction="none").mean(dim=(1, 2, 3))
            if sample_weights is not None:
                total = total + (diff * sample_weights.float()).mean()
            else:
                total = total + diff.mean()
        return total / max(1, len(self.slices))


class ControllableRestorationLoss(nn.Module):
    """Combined stage-aware objective for Pixel-Faithful and Semantic-Detail stages."""

    def __init__(self, config, use_pretrained_vgg: bool = True):
        super().__init__()
        self.config = config
        self.eps = 1e-3
        use_vgg = use_pretrained_vgg and (config.GENERATOR_BACKBONE.lower() != "mock_tiny")
        self.perceptual = MultiLayerVGGPerceptual(
            layers=getattr(config, "PERCEPTUAL_LAYERS", ["relu2_2", "relu3_4", "relu4_4"]),
            use_pretrained=use_vgg,
        )

    def charbonnier_with_preservation_weight(
        self, pred: torch.Tensor, target: torch.Tensor, action_map: torch.Tensor
    ) -> torch.Tensor:
        diff_sq = (pred.float() - target.float()) ** 2
        charb = torch.sqrt(diff_sq + self.eps * self.eps)  # [B, 3, H, W]

        # Boost weight slightly (1.35x) on low-action / preserved pixels so untouched attributes never drift
        mean_act = action_map.float().mean(dim=1, keepdim=True)  # [B, 1, H, W]
        pixel_weight = 1.35 - 0.35 * mean_act
        return (charb * pixel_weight).mean()

    @staticmethod
    def spatial_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f, tgt_f = pred.float(), target.float()
        dx_p = pred_f[:, :, :, 1:] - pred_f[:, :, :, :-1]
        dx_t = tgt_f[:, :, :, 1:] - tgt_f[:, :, :, :-1]
        dy_p = pred_f[:, :, 1:, :] - pred_f[:, :, :-1, :]
        dy_t = tgt_f[:, :, 1:, :] - tgt_f[:, :, :-1, :]
        return 0.5 * (F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t))

    @staticmethod
    def fft_frequency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute in FP32 to prevent FP16 complex FFT overflow
        fp = torch.fft.rfft2(pred.float(), norm="ortho")
        ft = torch.fft.rfft2(target.float(), norm="ortho")
        amp_loss = F.l1_loss(torch.abs(fp), torch.abs(ft))
        real_imag_loss = F.l1_loss(torch.view_as_real(fp), torch.view_as_real(ft))
        return 0.5 * (amp_loss + real_imag_loss)

    def forward(
        self,
        pred_01: torch.Tensor,
        target_01: torch.Tensor,
        action_map: torch.Tensor,
        align_loss: torch.Tensor,
        stage: str = "pixel_stage",
        t_vec: Optional[torch.Tensor] = None,
        semantic_encoder: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute stage-appropriate total loss and scalar logging dictionary."""
        align_scalar = align_loss.float().mean()
        stage = stage.lower()

        loss_l1 = self.charbonnier_with_preservation_weight(pred_01, target_01, action_map)
        loss_grad = self.spatial_gradient_loss(pred_01, target_01)
        loss_freq = self.fft_frequency_loss(pred_01, target_01)

        total = (
            self.config.LOSS_L1_WEIGHT * loss_l1
            + self.config.LOSS_GRADIENT_WEIGHT * loss_grad
            + self.config.LOSS_FREQUENCY_WEIGHT * loss_freq
        )

        # Keep Self-Teacher alignment active whenever vision_lora has gradients enabled
        if stage in ("pixel_stage", "joint"):
            total = total + self.config.LOSS_ALIGN_WEIGHT * align_scalar

        loss_perc = pred_01.new_zeros((), dtype=torch.float32)
        loss_sem_cons = pred_01.new_zeros((), dtype=torch.float32)

        if stage in ("semantic_stage", "joint"):
            # Down-weight perceptual loss slightly on high-t bridge extrapolations for artifact-free edges
            if t_vec is not None:
                t_weights = (1.15 - 0.35 * t_vec.float().view(-1)).clamp(0.5, 1.0)
            else:
                t_weights = None

            loss_perc = self.perceptual(pred_01, target_01, sample_weights=t_weights)
            total = total + self.config.LOSS_PERCEPTUAL_WEIGHT * loss_perc

            if semantic_encoder is not None and self.config.LOSS_SEM_CONSISTENCY_WEIGHT > 0.0:
                pred_sem = semantic_encoder(pred_01, use_lora=False)
                with torch.no_grad():
                    tgt_sem = semantic_encoder(target_01, use_lora=False)
                p_tok = F.normalize(pred_sem["tokens"].float(), dim=-1)
                g_tok = F.normalize(tgt_sem["tokens"].float().detach(), dim=-1)
                loss_sem_cons = (1.0 - (p_tok * g_tok).sum(dim=-1)).mean()
                total = total + self.config.LOSS_SEM_CONSISTENCY_WEIGHT * loss_sem_cons

        metrics = {
            "l1": float(loss_l1.detach().item()),
            "grad": float(loss_grad.detach().item()),
            "freq": float(loss_freq.detach().item()),
            "align": float(align_scalar.detach().item()),
            "perc": float(loss_perc.detach().item()),
            "sem_cons": float(loss_sem_cons.detach().item()),
            "total": float(total.detach().item()),
        }
        return total, metrics
