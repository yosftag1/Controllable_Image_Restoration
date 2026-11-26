"""Crisp Phase-Aligned Loss Objective & Zero-VRAM Frozen-DINOv2 Discriminator (S3Diff / ADD Style).

Stage 1 (`pixel_stage`):
  - Charbonnier RGB L1 (`LOSS_L1_WEIGHT`)
  - Latent Charbonnier (`LOSS_LATENT_WEIGHT`)
  - Un-blurred Full-Res Multi-Layer VGG Perceptual (`LOSS_PIXEL_STAGE_PERCEPTUAL_WEIGHT`)
  - Phase-Aligned Multi-Scale Edge Gradient L1 (`LOSS_PIXEL_STAGE_EDGE_WEIGHT`)
  - Idle-only preservation (`LOSS_PRESERVE_WEIGHT`)

Stage 2 (`semantic_stage`, trained at fixed `detail_scale = 1.0`):
  - Structural Anchor: 2x Low-Pass Charbonnier (`LOSS_SEMANTIC_STAGE_LP_L1_WEIGHT`) +
    Full-Res Charbonnier (`LOSS_SEMANTIC_STAGE_FULL_L1_WEIGHT`) + light Latent Anchor
  - Un-blurred VGG + True Gaussian-Windowed DISTS WITH Structural Covariance (`LOSS_PERCEPTUAL_WEIGHT`)
  - Phase-Aligned Multi-Scale Sobel + Laplacian Edge Gradient L1 (`LOSS_EDGE_GRAD_WEIGHT`)
  - Frozen DINOv2 Multi-Layer Feature Discriminator (`LOSS_GAN_WEIGHT`), strictly masked to
    100% fully restored (`s_tgt == 0`), high-resolution clean photography ONLY!
"""

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiLayerVGGPerceptual(nn.Module):
    """Un-blurred full-res (`relu1_2..relu4_4`) VGG Perceptual + True Covariance-Aware DISTS."""

    LAYER_MAP = {
        "relu1_2": 3,
        "relu2_2": 8,
        "relu3_4": 17,
        "relu4_4": 26,
        "relu5_4": 35,
    }
    LAYER_WEIGHTS = {
        3: 0.25,
        8: 0.30,
        17: 0.25,
        26: 0.20,
        35: 0.15,
    }

    def __init__(
        self,
        layers: Iterable[str] = ("relu1_2", "relu2_2", "relu3_4", "relu4_4"),
        use_pretrained: bool = True,
    ):
        super().__init__()
        self.target_indices = sorted(self.LAYER_MAP[name] for name in layers)
        raw_lw = [self.LAYER_WEIGHTS.get(idx, 1.0) for idx in self.target_indices]
        lw_sum = max(1e-6, sum(raw_lw))
        self.slice_weights = [w / lw_sum for w in raw_lw]

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

        # Separable 1D Gaussian kernel (window=7, sigma=1.5) for fast local-window covariance DISTS
        coords = torch.arange(7, dtype=torch.float32) - 3.0
        g1 = torch.exp(-(coords * coords) / (2.0 * 1.5 * 1.5))
        g1 = g1 / g1.sum()
        self.register_buffer("gauss_1d_h", g1.view(1, 1, 7, 1), persistent=False)
        self.register_buffer("gauss_1d_w", g1.view(1, 1, 1, 7), persistent=False)

    def _local_gauss_mean(self, feat: torch.Tensor) -> torch.Tensor:
        """Separable 7x7 Gaussian local average per channel (handles arbitrary C cleanly)."""
        c = feat.shape[1]
        kh = self.gauss_1d_h.expand(c, 1, 7, 1)
        kw = self.gauss_1d_w.expand(c, 1, 1, 7)
        h = F.pad(feat, (0, 0, 3, 3), mode="reflect")
        h = F.conv2d(h, kh, groups=c)
        h = F.pad(h, (3, 3, 0, 0), mode="reflect")
        return F.conv2d(h, kw, groups=c)

    def _true_dists_map(self, x_f: torch.Tensor, y_f: torch.Tensor) -> torch.Tensor:
        """True covariance-aware DISTS distance map `[B, 1, H, W]` locking crisp edge phase & contrast."""
        ch_scale = y_f.abs().mean(dim=(0, 2, 3), keepdim=True).clamp(min=0.05)
        xn = x_f / ch_scale
        yn = y_f / ch_scale

        if xn.shape[-2] >= 7 and xn.shape[-1] >= 7:
            mu_x = self._local_gauss_mean(xn)
            mu_y = self._local_gauss_mean(yn)
            var_x = (self._local_gauss_mean(xn * xn) - mu_x * mu_x).clamp(min=0.0)
            var_y = (self._local_gauss_mean(yn * yn) - mu_y * mu_y).clamp(min=0.0)
            cov_xy = self._local_gauss_mean(xn * yn) - mu_x * mu_y
        else:
            mu_x = xn.mean(dim=(2, 3), keepdim=True)
            mu_y = yn.mean(dim=(2, 3), keepdim=True)
            var_x = ((xn - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
            var_y = ((yn - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
            cov_xy = ((xn - mu_x) * (yn - mu_y)).mean(dim=(2, 3), keepdim=True)

        c1 = 1e-4
        c2 = 1e-3
        # Mean texture similarity l(x, y) and phase-locked structural covariance similarity s(x, y)
        l_sim = (2.0 * mu_x * mu_y + c1) / (mu_x * mu_x + mu_y * mu_y + c1)
        s_sim = (2.0 * cov_xy + c2) / (var_x + var_y + c2)
        dists_ch = 0.35 * (1.0 - l_sim) + 0.65 * (1.0 - s_sim)
        return dists_ch.clamp(min=0.0, max=2.0).mean(dim=1, keepdim=True)

    def forward(
        self,
        pred_01: torch.Tensor,
        target_01: torch.Tensor,
        spatial_weight: Optional[torch.Tensor] = None,
        use_dists_texture: bool = False,
    ) -> torch.Tensor:
        # Always full-res un-blurred RGB (never pre-blurring Stage-1 perceptual features!)
        x = (pred_01.float() - self.mean) / self.std
        with torch.no_grad():
            y_in = (target_01.float() - self.mean) / self.std

        total_pos = pred_01.new_zeros((), dtype=torch.float32)

        for sw, block in zip(self.slice_weights, self.slices):
            x = block(x)
            with torch.no_grad():
                y_in = block(y_in)

            with torch.amp.autocast(device_type=x.device.type, enabled=False):
                x_f = x.float()
                y_f = y_in.detach().float()

                if spatial_weight is not None:
                    w_l = F.adaptive_avg_pool2d(spatial_weight.float(), output_size=x_f.shape[-2:])
                    w_l = w_l / w_l.mean(dim=(2, 3), keepdim=True).clamp(min=1e-4)
                else:
                    w_l = None

                x_unit = x_f / torch.sqrt(torch.sum(x_f * x_f, dim=1, keepdim=True) + 1e-8)
                y_unit = y_f / torch.sqrt(torch.sum(y_f * y_f, dim=1, keepdim=True) + 1e-8)
                lpips_map = 0.5 * ((x_unit - y_unit) ** 2).sum(dim=1, keepdim=True)

                if use_dists_texture:
                    dists_map = self._true_dists_map(x_f, y_f)
                    if dists_map.shape[-2:] != lpips_map.shape[-2:]:
                        dists_map = F.interpolate(
                            dists_map, size=lpips_map.shape[-2:], mode="bilinear", align_corners=False
                        )
                    comb_map = 0.40 * lpips_map + 0.60 * dists_map
                else:
                    comb_map = lpips_map

                if w_l is not None:
                    pos_dist = (comb_map * w_l).mean(dim=(1, 2, 3))
                else:
                    pos_dist = comb_map.mean(dim=(1, 2, 3))

                total_pos = total_pos + sw * pos_dist.mean()

        return total_pos


class MultiScaleEdgeGradientLoss(nn.Module):
    """Phase-aligned multi-scale signed Sobel X/Y + Laplacian edge loss (crisp edges, zero smear)."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "luma_w",
            torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32) / 4.0
        sobel_y = sobel_x.t().contiguous()
        lap = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=torch.float32) / 4.0
        kernels = torch.stack([sobel_x, sobel_y, lap], dim=0).unsqueeze(1)  # [3, 1, 3, 3]
        self.register_buffer("edge_kernels", kernels, persistent=False)

    def _edge_loss_single_scale(
        self,
        p_y: torch.Tensor,
        t_y: torch.Tensor,
        w_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gp = F.conv2d(F.pad(p_y, (1, 1, 1, 1), mode="reflect"), self.edge_kernels)
        gt = F.conv2d(F.pad(t_y, (1, 1, 1, 1), mode="reflect"), self.edge_kernels)
        diff = torch.sqrt((gp - gt) ** 2 + 1e-6)
        if w_map is not None:
            return (diff * w_map).mean()
        return diff.mean()

    def forward(
        self,
        pred_01: torch.Tensor,
        target_01: torch.Tensor,
        spatial_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=pred_01.device.type, enabled=False):
            p_y = (pred_01.float() * self.luma_w).sum(dim=1, keepdim=True)
            t_y = (target_01.detach().float() * self.luma_w).sum(dim=1, keepdim=True)

            if spatial_weight is not None:
                w1 = F.adaptive_avg_pool2d(spatial_weight.float(), output_size=p_y.shape[-2:])
                w1 = w1 / w1.mean(dim=(2, 3), keepdim=True).clamp(min=1e-4)
            else:
                w1 = None

            loss = self._edge_loss_single_scale(p_y, t_y, w1)
            if p_y.shape[-2] >= 32 and p_y.shape[-1] >= 32:
                p_half = F.avg_pool2d(p_y, kernel_size=2, stride=2)
                t_half = F.avg_pool2d(t_y, kernel_size=2, stride=2)
                w2 = F.avg_pool2d(w1, kernel_size=2, stride=2) if w1 is not None else None
                loss = 0.65 * loss + 0.35 * self._edge_loss_single_scale(p_half, t_half, w2)
            return loss


# Backward-compatible symbol alias
MultiScaleFreqEdgeLoss = MultiScaleEdgeGradientLoss


class DINOv2FeatureDiscriminator(nn.Module):
    """Zero-extra-backbone-VRAM Frozen-DINOv2 Multi-Layer Feature Discriminator (S3Diff / SD-Turbo ADD style).

    Attaches lightweight Spectral-Norm 2D conv heads onto 3 multi-depth spatial patch grids extracted
    from the already-loaded frozen `SemanticEncoder` (DINOv2-with-Registers). Unlike a scratch pixel
    PatchGAN, it already understands crisp photographic realism from step 1 and is strictly masked
    to 100% clean, fully restored (`s_tgt == 0`), high-res ground-truth targets!
    """

    def __init__(self, feat_dim: int = 768, num_scales: int = 3, hidden_dim: int = 192):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_scales = max(1, int(num_scales))
        sn = nn.utils.spectral_norm
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    sn(nn.Conv2d(self.feat_dim, hidden_dim, kernel_size=3, stride=1, padding=1)),
                    nn.LeakyReLU(0.2, inplace=True),
                    sn(nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=1, padding=1)),
                    nn.LeakyReLU(0.2, inplace=True),
                    sn(nn.Conv2d(hidden_dim // 2, 1, kernel_size=1, stride=1, padding=0)),
                )
                for _ in range(self.num_scales)
            ]
        )

    def forward(self, grids: List[torch.Tensor]) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        for idx, head in enumerate(self.heads):
            g = grids[min(idx, len(grids) - 1)]
            with torch.amp.autocast(device_type=g.device.type, enabled=False):
                outs.append(head(g.float()))
        return outs

    @staticmethod
    def _combine_masks(
        target_hw: Tuple[int, int],
        detail_mask: Optional[torch.Tensor],
        clean_sample_mask: Optional[torch.Tensor],
        bsz: int,
        device: torch.device,
    ) -> torch.Tensor:
        if detail_mask is not None:
            m = detail_mask.float().clamp(0.0, 1.0)
            if m.shape[-2:] != target_hw:
                m = F.adaptive_avg_pool2d(m, output_size=target_hw)
        else:
            m = torch.ones((bsz, 1, target_hw[0], target_hw[1]), device=device, dtype=torch.float32)

        if clean_sample_mask is not None:
            cm = clean_sample_mask.to(device=device, dtype=torch.float32).view(bsz, 1, 1, 1).clamp(0.0, 1.0)
            m = m * cm
        return m

    def compute_g_loss(
        self,
        semantic_encoder: nn.Module,
        fake_01: torch.Tensor,
        detail_mask: Optional[torch.Tensor] = None,
        clean_sample_mask: Optional[torch.Tensor] = None,
        max_size: int = 252,
    ) -> torch.Tensor:
        """Generator Hinge GAN loss (`-E[D(fake)]`) gated strictly to clean fully-restored detail regions."""
        grids = semantic_encoder.extract_discriminator_grids(fake_01, max_size=max_size, enable_grad=True)
        logits_list = self.forward(grids)
        bsz = fake_01.shape[0]
        total = fake_01.new_zeros((), dtype=torch.float32)
        for logits in logits_list:
            m = self._combine_masks(logits.shape[-2:], detail_mask, clean_sample_mask, bsz, logits.device)
            m_sum = m.sum()
            if float(m_sum.item()) > 1e-4:
                total = total + (-(logits * m).sum() / m_sum.clamp(min=1.0))
            else:
                total = total + (logits * 0.0).mean()
        return total / float(len(logits_list))

    @classmethod
    def compute_d_loss_with_module(
        cls,
        disc_module: nn.Module,
        semantic_encoder: nn.Module,
        real_01: torch.Tensor,
        fake_01_detached: torch.Tensor,
        detail_mask: Optional[torch.Tensor] = None,
        clean_sample_mask: Optional[torch.Tensor] = None,
        max_size: int = 252,
    ) -> Tuple[torch.Tensor, float, float]:
        """Single-pass DDP-safe Hinge D loss strictly on clean high-res fully-restored targets."""
        bsz = real_01.shape[0]
        cat_x = torch.cat([real_01.detach(), fake_01_detached.detach()], dim=0)
        with torch.no_grad():
            cat_grids = semantic_encoder.extract_discriminator_grids(cat_x, max_size=max_size, enable_grad=False)
        cat_logits_list = disc_module(cat_grids)

        loss_d_total = real_01.new_zeros((), dtype=torch.float32)
        real_mean_acc = 0.0
        fake_mean_acc = 0.0
        for cat_logits in cat_logits_list:
            d_real, d_fake = cat_logits.chunk(2, dim=0)
            m = cls._combine_masks(d_real.shape[-2:], detail_mask, clean_sample_mask, bsz, d_real.device)
            m_sum = m.sum()
            if float(m_sum.item()) > 1e-4:
                loss_real = (F.relu(1.0 - d_real) * m).sum() / m_sum.clamp(min=1.0)
                loss_fake = (F.relu(1.0 + d_fake) * m).sum() / m_sum.clamp(min=1.0)
                loss_d_total = loss_d_total + 0.5 * (loss_real + loss_fake)
            else:
                # Zero-weight touch keeps all DDP discriminator heads synchronized without corrupting weights!
                loss_d_total = loss_d_total + (d_real * 0.0 + d_fake * 0.0).mean()
            real_mean_acc += float(d_real.detach().mean().item())
            fake_mean_acc += float(d_fake.detach().mean().item())

        n_sc = float(len(cat_logits_list))
        return loss_d_total / n_sc, real_mean_acc / n_sc, fake_mean_acc / n_sc


# Backward-compatible symbol alias
SpectralPatchDiscriminator = DINOv2FeatureDiscriminator


class ControllableRestorationLoss(nn.Module):
    """Crisp phase-aligned 2-stage objective (Charbonnier, Latent, Un-blurred VGG/True-DISTS, Edge Grad, Idle)."""

    def __init__(self, config, use_pretrained_vgg: bool = True):
        super().__init__()
        self.config = config
        self.eps = 1e-3
        use_vgg = use_pretrained_vgg and (config.GENERATOR_BACKBONE.lower() != "mock_tiny")
        self.perceptual = MultiLayerVGGPerceptual(
            layers=getattr(config, "PERCEPTUAL_LAYERS", ["relu1_2", "relu2_2", "relu3_4", "relu4_4"]),
            use_pretrained=use_vgg,
        )
        self.edge_grad = MultiScaleEdgeGradientLoss()
        self.freq_edge = self.edge_grad  # Backward-compatible attribute alias

        b1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32) / 16.0
        b2d = (b1d[:, None] * b1d[None, :]).view(1, 1, 5, 5).repeat(3, 1, 1, 1)
        self.register_buffer("gauss5_kernel", b2d, persistent=False)

    def _gaussian_blur5(self, rgb: torch.Tensor) -> torch.Tensor:
        padded = F.pad(rgb, (2, 2, 2, 2), mode="reflect")
        return F.conv2d(padded, self.gauss5_kernel, groups=3)

    def _lowpass_2x(self, rgb_01: torch.Tensor) -> torch.Tensor:
        """Anti-aliased 2x low-pass structural anchor (locks exact shapes & colors while freeing 1px detail)."""
        return self._gaussian_blur5(rgb_01.float())[:, :, ::2, ::2]

    @staticmethod
    def _build_spatial_focus_weight(action_map: torch.Tensor) -> torch.Tensor:
        max_act = action_map.float().abs().amax(dim=1, keepdim=True)
        raw_w = 0.75 + 0.75 * max_act
        return (raw_w / raw_w.mean(dim=(2, 3), keepdim=True).clamp(min=1e-4)).detach()

    def charbonnier_loss(
        self, pred: torch.Tensor, target: torch.Tensor, spatial_weight: torch.Tensor
    ) -> torch.Tensor:
        diff_sq = (pred.float() - target.float()) ** 2
        charb = torch.sqrt(diff_sq + self.eps * self.eps)
        return (charb * spatial_weight).mean()

    @staticmethod
    def latent_reconstruction_loss(
        hat_z_0: torch.Tensor,
        z_tgt: torch.Tensor,
        spatial_weight: Optional[torch.Tensor] = None,
        lowpass_only: bool = False,
    ) -> torch.Tensor:
        zp = hat_z_0.float()
        zt = z_tgt.float().detach()
        if lowpass_only and zp.shape[-2] >= 4 and zp.shape[-1] >= 4:
            zp = F.avg_pool2d(zp, kernel_size=2, stride=2)
            zt = F.avg_pool2d(zt, kernel_size=2, stride=2)
        charb = torch.sqrt((zp - zt) ** 2 + 1e-6)
        if spatial_weight is not None:
            w_z = F.adaptive_avg_pool2d(spatial_weight.float(), output_size=zp.shape[-2:])
            w_z = w_z / w_z.mean(dim=(2, 3), keepdim=True).clamp(min=1e-4)
            return (charb * w_z).mean()
        return charb.mean()

    @staticmethod
    def idle_preservation_loss(
        pred_01: torch.Tensor,
        src_01: torch.Tensor,
        action_map: torch.Tensor,
    ) -> torch.Tensor:
        """Strictly preserve regions where all control sliders are inactive (zero action)."""
        p_f, s_f, a_f = pred_01.float(), src_01.float(), action_map.float()
        max_abs_act = a_f.abs().amax(dim=1, keepdim=True)
        idle_mask = (1.0 - (max_abs_act / 0.12).clamp(0.0, 1.0)).square()
        return ((p_f - s_f).abs() * idle_mask).mean()

    def forward(
        self,
        pred_01: torch.Tensor,
        target_01: torch.Tensor,
        action_map: torch.Tensor,
        stage: str = "pixel_stage",
        src_01: Optional[torch.Tensor] = None,
        hat_z_0: Optional[torch.Tensor] = None,
        hat_z_1: Optional[torch.Tensor] = None,
        z_tgt: Optional[torch.Tensor] = None,
        restorer_model: Optional[nn.Module] = None,
        run_csd: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        stage = stage.lower()
        is_detail_stage = stage == "semantic_stage"
        spatial_w = self._build_spatial_focus_weight(action_map)

        if is_detail_stage and pred_01.shape[-2] >= 16 and pred_01.shape[-1] >= 16:
            pred_lp2 = self._lowpass_2x(pred_01)
            tgt_lp2 = self._lowpass_2x(target_01)
            w_lp2 = F.adaptive_avg_pool2d(spatial_w, output_size=pred_lp2.shape[-2:])
            w_lp2 = w_lp2 / w_lp2.mean(dim=(2, 3), keepdim=True).clamp(min=1e-4)
            loss_l1_lp = self.charbonnier_loss(pred_lp2, tgt_lp2, w_lp2)
            full_w = float(getattr(self.config, "LOSS_SEMANTIC_STAGE_FULL_L1_WEIGHT", 0.20))
            lp_w = float(getattr(self.config, "LOSS_SEMANTIC_STAGE_LP_L1_WEIGHT", 0.35))
            if full_w > 0.0:
                loss_l1_full = self.charbonnier_loss(pred_01, target_01, spatial_w)
                l1_term = lp_w * loss_l1_lp + full_w * loss_l1_full
                loss_l1 = l1_term / max(1e-6, lp_w + full_w)
            else:
                l1_term = lp_w * loss_l1_lp
                loss_l1 = loss_l1_lp
        else:
            loss_l1 = self.charbonnier_loss(pred_01, target_01, spatial_w)
            l1_term = float(self.config.LOSS_L1_WEIGHT) * loss_l1

        if hat_z_0 is not None and z_tgt is not None:
            loss_z0 = self.latent_reconstruction_loss(
                hat_z_0, z_tgt, spatial_weight=spatial_w, lowpass_only=is_detail_stage
            )
            if (
                not is_detail_stage
                and hat_z_1 is not None
                and hat_z_1 is not hat_z_0
                and hat_z_1.requires_grad
            ):
                loss_z1 = self.latent_reconstruction_loss(
                    hat_z_1, z_tgt, spatial_weight=spatial_w, lowpass_only=False
                )
                loss_latent = 0.35 * loss_z1 + 0.65 * loss_z0
            else:
                loss_latent = loss_z0
        else:
            loss_latent = pred_01.new_zeros((), dtype=torch.float32)

        loss_preserve = (
            self.idle_preservation_loss(pred_01, src_01, action_map)
            if src_01 is not None
            else pred_01.new_zeros((), dtype=torch.float32)
        )

        lat_weight = (
            float(getattr(self.config, "LOSS_SEMANTIC_STAGE_LATENT_LP_WEIGHT", 0.10))
            if is_detail_stage
            else float(getattr(self.config, "LOSS_LATENT_WEIGHT", 0.25))
        )
        preserve_weight = float(getattr(self.config, "LOSS_PRESERVE_WEIGHT", 0.50))
        perc_w = (
            float(getattr(self.config, "LOSS_PIXEL_STAGE_PERCEPTUAL_WEIGHT", 0.35))
            if stage == "pixel_stage"
            else float(self.config.LOSS_PERCEPTUAL_WEIGHT)
        )
        edge_w = (
            float(getattr(self.config, "LOSS_PIXEL_STAGE_EDGE_WEIGHT", 0.25))
            if stage == "pixel_stage"
            else float(getattr(self.config, "LOSS_EDGE_GRAD_WEIGHT", getattr(self.config, "LOSS_FREQ_EDGE_WEIGHT", 0.35)))
        )

        total = l1_term + lat_weight * loss_latent + preserve_weight * loss_preserve

        loss_perc = pred_01.new_zeros((), dtype=torch.float32)
        if perc_w > 0.0:
            use_dists = stage in ("semantic_stage", "joint")
            loss_perc = self.perceptual(
                pred_01,
                target_01,
                spatial_weight=spatial_w,
                use_dists_texture=use_dists,
            )
            total = total + perc_w * loss_perc

        loss_edge = pred_01.new_zeros((), dtype=torch.float32)
        if edge_w > 0.0:
            loss_edge = self.edge_grad(pred_01, target_01, spatial_weight=spatial_w)
            total = total + edge_w * loss_edge

        loss_csd = pred_01.new_zeros((), dtype=torch.float32)
        csd_w = float(getattr(self.config, "LOSS_CSD_WEIGHT", 0.0))
        if (
            run_csd
            and stage in ("semantic_stage", "joint")
            and csd_w > 0.0
            and restorer_model is not None
            and hat_z_0 is not None
            and z_tgt is not None
        ):
            loss_csd = restorer_model.compute_target_anchored_csd_loss(hat_z_0, z_tgt, action_map)
            total = total + csd_w * loss_csd

        metrics = {
            "l1": float(loss_l1.detach().item()),
            "lat": float(loss_latent.detach().item()),
            "perc": float(loss_perc.detach().item()),
            "edge": float(loss_edge.detach().item()),
            "freq": float(loss_edge.detach().item()),
            "pres": float(loss_preserve.detach().item()),
            "csd": float(loss_csd.detach().item()),
            "total": float(total.detach().item()),
        }
        return total, metrics
