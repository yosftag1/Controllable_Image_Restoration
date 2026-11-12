"""Loss functions & Vision-Aided GAN Discriminator for Controllable Restoration."""

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

try:
    from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm
except ImportError:
    from torch.nn.utils import spectral_norm as _spectral_norm


class MultiLayerVGGPerceptual(nn.Module):
    """LPIPS-style channel-normalized + contrast multi-scale VGG19 perceptual loss."""

    LAYER_MAP = {
        "relu1_2": 3,
        "relu2_2": 8,
        "relu3_4": 17,
        "relu4_4": 26,
        "relu5_4": 35,
    }

    def __init__(
        self,
        layers: Iterable[str] = ("relu1_2", "relu2_2", "relu3_4", "relu4_4"),
        use_pretrained: bool = True,
    ):
        super().__init__()
        self.target_indices = sorted(self.LAYER_MAP[name] for name in layers)

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

    def forward(
        self,
        pred_01: torch.Tensor,
        target_01: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = (pred_01.float() - self.mean) / self.std
        y = (target_01.float() - self.mean) / self.std

        total = pred_01.new_zeros((), dtype=torch.float32)
        for block in self.slices:
            x = block(x)
            with torch.no_grad():
                y = block(y)
            y_det = y.detach()

            # LPIPS unit channel normalization equalizes high-res edge (`relu1_2`) & deep semantic (`relu4_4`) scales
            x_unit = x / torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + 1e-10)
            y_unit = y_det / torch.sqrt(torch.sum(y_det * y_det, dim=1, keepdim=True) + 1e-10)
            lpips_dist = ((x_unit - y_unit) ** 2).sum(dim=1).mean(dim=(1, 2))
            raw_l1 = F.l1_loss(x, y_det, reduction="none").mean(dim=(1, 2, 3))
            diff = lpips_dist + 0.35 * raw_l1

            if sample_weights is not None:
                total = total + (diff * sample_weights.float()).mean()
            else:
                total = total + diff.mean()
        return total / max(1, len(self.slices))


class VisionAidedDiscriminator(nn.Module):
    """Action-conditioned Spectral-Norm PatchGAN + Frozen Vision (`C-RADIO`) Discriminator.

    Reuses the `[B, D_spat, H_p, W_p]` `"grid"` features already produced by `semantic_encoder(..., use_lora=False)`
    in `semantic_stage` / `joint` — adding adversarial sharpness with ZERO extra ViT forward passes.
    """

    def __init__(self, sem_dim: int, num_actions: int, base_ch: int = 64):
        super().__init__()
        rgb_in = 3 + num_actions
        self.rgb_net = nn.Sequential(
            _spectral_norm(nn.Conv2d(rgb_in, base_ch, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=False),
            _spectral_norm(nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=False),
            _spectral_norm(nn.Conv2d(base_ch * 2, base_ch * 3, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.head_rgb = _spectral_norm(nn.Conv2d(base_ch * 3, 1, kernel_size=3, padding=1))

        sem_in = int(sem_dim) + num_actions
        sem_ch = base_ch * 4
        self.sem_proj = nn.Sequential(
            _spectral_norm(nn.Conv2d(sem_in, sem_ch, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.sem_fuse = nn.Sequential(
            _spectral_norm(nn.Conv2d(sem_ch + base_ch * 3, sem_ch, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.head_sem = _spectral_norm(nn.Conv2d(sem_ch, 1, kernel_size=1))

    def forward(
        self,
        rgb_01: torch.Tensor,
        sem_grid: torch.Tensor,
        action_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_rgb = rgb_01.float() * 2.0 - 1.0
        act_f = action_map.float()
        if act_f.shape[-2:] != x_rgb.shape[-2:]:
            act_rgb = F.interpolate(act_f, size=x_rgb.shape[-2:], mode="area")
        else:
            act_rgb = act_f

        h_rgb = self.rgb_net(torch.cat([x_rgb, act_rgb], dim=1))
        logits_rgb = self.head_rgb(h_rgb).float()

        g_norm = F.normalize(sem_grid.float(), dim=1)
        if act_f.shape[-2:] != g_norm.shape[-2:]:
            act_sem = F.interpolate(act_f, size=g_norm.shape[-2:], mode="area")
        else:
            act_sem = act_f

        h_sem = self.sem_proj(torch.cat([g_norm, act_sem], dim=1))
        if h_rgb.shape[-2:] != h_sem.shape[-2:]:
            h_rgb_sem = F.interpolate(h_rgb, size=h_sem.shape[-2:], mode="bilinear", align_corners=False)
        else:
            h_rgb_sem = h_rgb

        logits_sem = self.head_sem(self.sem_fuse(torch.cat([h_sem, h_rgb_sem], dim=1))).float()
        return logits_rgb, logits_sem

    @staticmethod
    def generator_hinge_loss(
        logits_rgb: torch.Tensor,
        logits_sem: torch.Tensor,
        action_map: torch.Tensor,
    ) -> torch.Tensor:
        # Gate GAN push by per-sample slider strength so pure identity (action=0) samples get zero GAN drift
        act_strength = action_map.float().amax(dim=(1, 2, 3)).clamp(0.0, 1.0)
        g_rgb = (-logits_rgb.float()).mean(dim=(1, 2, 3))
        g_sem = (-logits_sem.float()).mean(dim=(1, 2, 3))
        return (0.5 * (g_rgb + g_sem) * act_strength).mean()

    @staticmethod
    def discriminator_hinge_loss(
        disc_module: nn.Module,
        real_rgb: torch.Tensor,
        fake_rgb: torch.Tensor,
        real_grid: torch.Tensor,
        fake_grid: torch.Tensor,
        action_map: torch.Tensor,
    ) -> torch.Tensor:
        """Single concatenated forward pass through `disc_module` (100% DDP single-reducer safe)."""
        cat_rgb = torch.cat([real_rgb.detach(), fake_rgb.detach()], dim=0)
        cat_grid = torch.cat([real_grid.detach(), fake_grid.detach()], dim=0)
        cat_act = torch.cat([action_map.detach(), action_map.detach()], dim=0)
        logits_rgb, logits_sem = disc_module(cat_rgb, cat_grid, cat_act)

        real_r, fake_r = logits_rgb.float().chunk(2, dim=0)
        real_s, fake_s = logits_sem.float().chunk(2, dim=0)
        loss_r = 0.5 * (F.relu(1.0 - real_r).mean() + F.relu(1.0 + fake_r).mean())
        loss_s = 0.5 * (F.relu(1.0 - real_s).mean() + F.relu(1.0 + fake_s).mean())
        return 0.5 * (loss_r + loss_s)


class ControllableRestorationLoss(nn.Module):
    """Combined stage-aware objective for Pixel-Faithful and Semantic-Detail stages."""

    def __init__(self, config, use_pretrained_vgg: bool = True):
        super().__init__()
        self.config = config
        self.eps = 1e-3
        use_vgg = use_pretrained_vgg and (config.GENERATOR_BACKBONE.lower() != "mock_tiny")
        self.perceptual = MultiLayerVGGPerceptual(
            layers=getattr(config, "PERCEPTUAL_LAYERS", ["relu1_2", "relu2_2", "relu3_4", "relu4_4"]),
            use_pretrained=use_vgg,
        )

    def charbonnier_with_preservation_weight(
        self, pred: torch.Tensor, target: torch.Tensor, action_map: torch.Tensor
    ) -> torch.Tensor:
        diff_sq = (pred.float() - target.float()) ** 2
        charb = torch.sqrt(diff_sq + self.eps * self.eps)  # [B, 3, H, W]

        # Focus extra supervision on actively edited regions while keeping selective preservation accurate
        max_act = action_map.float().amax(dim=1, keepdim=True)  # [B, 1, H, W]
        pixel_weight = 1.0 + 0.30 * max_act
        return (charb * pixel_weight).mean()

    @staticmethod
    def multiscale_color_tone_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Low-pass pooled RGB + opponent chroma loss for accurate color-cast, exposure & banding removal."""
        pred_f, tgt_f = pred.float(), target.float()
        total = pred_f.new_zeros((), dtype=torch.float32)
        scales = 0
        for pool_k in (8, 32):
            if pred_f.shape[-2] >= pool_k and pred_f.shape[-1] >= pool_k:
                p_low = F.avg_pool2d(pred_f, kernel_size=pool_k, stride=pool_k)
                t_low = F.avg_pool2d(tgt_f, kernel_size=pool_k, stride=pool_k)
                l_rgb = F.l1_loss(p_low, t_low)
                p_chr = torch.cat([p_low[:, 0:1] - p_low[:, 1:2], p_low[:, 2:3] - p_low[:, 1:2]], dim=1)
                t_chr = torch.cat([t_low[:, 0:1] - t_low[:, 1:2], t_low[:, 2:3] - t_low[:, 1:2]], dim=1)
                total = total + 0.5 * (l_rgb + F.l1_loss(p_chr, t_chr))
                scales += 1
        return total / max(1, scales)

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
        discriminator: Optional[nn.Module] = None,
        return_aux: bool = False,
    ):
        """Compute stage-appropriate total loss, scalar logging dictionary, and optional cached ViT grids."""
        align_scalar = align_loss.float().mean()
        stage = stage.lower()

        loss_l1 = self.charbonnier_with_preservation_weight(pred_01, target_01, action_map)
        loss_grad = self.spatial_gradient_loss(pred_01, target_01)
        loss_freq = self.fft_frequency_loss(pred_01, target_01)
        loss_color = self.multiscale_color_tone_loss(pred_01, target_01)

        l1_weight = (
            float(getattr(self.config, "LOSS_SEMANTIC_STAGE_L1_WEIGHT", 0.30))
            if stage == "semantic_stage"
            else float(self.config.LOSS_L1_WEIGHT)
        )
        color_weight = float(getattr(self.config, "LOSS_COLOR_TONE_WEIGHT", 0.30))

        total = (
            l1_weight * loss_l1
            + self.config.LOSS_GRADIENT_WEIGHT * loss_grad
            + self.config.LOSS_FREQUENCY_WEIGHT * loss_freq
            + color_weight * loss_color
        )

        # Keep Self-Teacher alignment active whenever vision_lora has gradients enabled
        if stage in ("pixel_stage", "joint"):
            total = total + self.config.LOSS_ALIGN_WEIGHT * align_scalar

        loss_perc = pred_01.new_zeros((), dtype=torch.float32)
        loss_sem_cons = pred_01.new_zeros((), dtype=torch.float32)
        loss_gan = pred_01.new_zeros((), dtype=torch.float32)
        aux: Dict[str, torch.Tensor] = {}

        if t_vec is not None:
            t_weights = (1.15 - 0.30 * t_vec.float().view(-1)).clamp(0.6, 1.0)
        else:
            t_weights = None

        pix_perc_w = float(getattr(self.config, "LOSS_PIXEL_STAGE_PERCEPTUAL_WEIGHT", 0.85))
        gan_w = float(getattr(self.config, "LOSS_GAN_WEIGHT", 0.20))

        if stage == "pixel_stage" and pix_perc_w > 0.0:
            # Strong LPIPS-VGG structural supervision in Phase 2 defeats blurry L1 averages & 8-bit banding rings
            loss_perc = self.perceptual(pred_01, target_01, sample_weights=t_weights)
            total = total + pix_perc_w * loss_perc
        elif stage in ("semantic_stage", "joint"):
            loss_perc = self.perceptual(pred_01, target_01, sample_weights=t_weights)
            total = total + self.config.LOSS_PERCEPTUAL_WEIGHT * loss_perc

            need_sem = semantic_encoder is not None and (
                self.config.LOSS_SEM_CONSISTENCY_WEIGHT > 0.0
                or (discriminator is not None and gan_w > 0.0)
            )
            if need_sem:
                pred_sem = semantic_encoder(pred_01, use_lora=False)
                with torch.no_grad():
                    tgt_sem = semantic_encoder(target_01, use_lora=False)

                if self.config.LOSS_SEM_CONSISTENCY_WEIGHT > 0.0:
                    p_tok = F.normalize(pred_sem["tokens"].float(), dim=-1)
                    g_tok = F.normalize(tgt_sem["tokens"].float().detach(), dim=-1)
                    loss_sem_cons = (1.0 - (p_tok * g_tok).sum(dim=-1)).mean()
                    total = total + self.config.LOSS_SEM_CONSISTENCY_WEIGHT * loss_sem_cons

                if discriminator is not None and gan_w > 0.0:
                    logits_rgb, logits_sem = discriminator(pred_01, pred_sem["grid"], action_map)
                    loss_gan = VisionAidedDiscriminator.generator_hinge_loss(logits_rgb, logits_sem, action_map)
                    total = total + gan_w * loss_gan
                    aux["pred_grid"] = pred_sem["grid"].detach()
                    aux["tgt_grid"] = tgt_sem["grid"].detach()

        metrics = {
            "l1": float(loss_l1.detach().item()),
            "grad": float(loss_grad.detach().item()),
            "freq": float(loss_freq.detach().item()),
            "color": float(loss_color.detach().item()),
            "align": float(align_scalar.detach().item()),
            "perc": float(loss_perc.detach().item()),
            "sem_cons": float(loss_sem_cons.detach().item()),
            "gan": float(loss_gan.detach().item()),
            "total": float(total.detach().item()),
        }
        if return_aux:
            return total, metrics, aux
        return total, metrics
