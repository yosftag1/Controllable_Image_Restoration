"""Clean 4-Term Loss Objective for 2-Stage Controllable Restoration.

Stage 1 (`pixel_stage`):
  - Charbonnier L1 (`LOSS_L1_WEIGHT`)
  - 2-Step Latent Charbonnier (`0.35 * hat_z_1 + 0.65 * hat_z_0`) (`LOSS_LATENT_WEIGHT`)
  - Light VGG LPIPS (`LOSS_PIXEL_STAGE_PERCEPTUAL_WEIGHT`)
  - Idle-only preservation (`LOSS_PRESERVE_WEIGHT`)

Stage 2 (`semantic_stage`):
  - 4x Low-Pass Charbonnier L1 (`LOSS_SEMANTIC_STAGE_LP_L1_WEIGHT`) + gentle Full-Res Subpixel Anchor
    (`LOSS_SEMANTIC_STAGE_FULL_L1_WEIGHT`) to prevent 1-2px fine-detail jitter
  - 2x Low-Pass Latent Charbonnier (`LOSS_SEMANTIC_STAGE_LATENT_LP_WEIGHT`)
  - Full-res VGG LPIPS + DISTS Local Patch Texture (`LOSS_PERCEPTUAL_WEIGHT`)
  - Zero-Drift Target-Anchored CSD (`LOSS_CSD_WEIGHT`)
  - Idle-only preservation (`LOSS_PRESERVE_WEIGHT`)
"""

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiLayerVGGPerceptual(nn.Module):
    """Full-res (`relu1_2..relu4_4`) LPIPS + optional DISTS Local Patch Texture matching."""

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
        k1d = torch.tensor([0.25, 0.50, 0.25], dtype=torch.float32)
        k2d = (k1d[:, None] * k1d[None, :]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer("aa_kernel", k2d, persistent=False)

    def _antialias_rgb(self, rgb_01: torch.Tensor) -> torch.Tensor:
        padded = F.pad(rgb_01.float(), (1, 1, 1, 1), mode="reflect")
        return F.conv2d(padded, self.aa_kernel, groups=3)

    def forward(
        self,
        pred_01: torch.Tensor,
        target_01: torch.Tensor,
        spatial_weight: Optional[torch.Tensor] = None,
        use_dists_texture: bool = False,
    ) -> torch.Tensor:
        p_rgb = pred_01.float() if use_dists_texture else self._antialias_rgb(pred_01)
        x = (p_rgb - self.mean) / self.std
        with torch.no_grad():
            t_rgb = target_01.float() if use_dists_texture else self._antialias_rgb(target_01)
            y_in = (t_rgb - self.mean) / self.std

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

                if w_l is not None:
                    pos_dist = (lpips_map * w_l).mean(dim=(1, 2, 3))
                else:
                    pos_dist = lpips_map.mean(dim=(1, 2, 3))

                if use_dists_texture:
                    tex_dist = pred_01.new_zeros((x_f.shape[0],), dtype=torch.float32)
                    win_cnt = 0
                    ch_scale = y_f.abs().mean(dim=(0, 2, 3), keepdim=True).clamp(min=0.05)
                    xn = x_f / ch_scale
                    yn = y_f / ch_scale
                    for win_k, win_s in ((8, 4), (16, 8)):
                        if xn.shape[-2] >= win_k and xn.shape[-1] >= win_k:
                            mu_x = F.avg_pool2d(xn, kernel_size=win_k, stride=win_s)
                            mu_y = F.avg_pool2d(yn, kernel_size=win_k, stride=win_s)
                            var_x = (F.avg_pool2d(xn * xn, kernel_size=win_k, stride=win_s) - mu_x * mu_x).clamp(min=0.0)
                            var_y = (F.avg_pool2d(yn * yn, kernel_size=win_k, stride=win_s) - mu_y * mu_y).clamp(min=0.0)
                            std_x = torch.sqrt(var_x + 1e-6)
                            std_y = torch.sqrt(var_y + 1e-6)
                            patch_err = (0.30 * (mu_x - mu_y).abs() + 0.70 * (std_x - std_y).abs()).mean(
                                dim=1, keepdim=True
                            )
                            if spatial_weight is not None:
                                w_p = F.adaptive_avg_pool2d(spatial_weight.float(), output_size=patch_err.shape[-2:])
                                w_p = w_p / w_p.mean(dim=(2, 3), keepdim=True).clamp(min=1e-4)
                                tex_dist = tex_dist + (patch_err * w_p).mean(dim=(1, 2, 3))
                            else:
                                tex_dist = tex_dist + patch_err.mean(dim=(1, 2, 3))
                            win_cnt += 1
                    if win_cnt > 0:
                        pos_dist = 0.40 * pos_dist + 0.60 * (tex_dist / float(win_cnt))

                total_pos = total_pos + sw * pos_dist.mean()

        return total_pos


class ControllableRestorationLoss(nn.Module):
    """Clean 4-term controllable restoration objective (Charbonnier, 2-Step Latent, VGG/DISTS, Idle + CSD)."""

    def __init__(self, config, use_pretrained_vgg: bool = True):
        super().__init__()
        self.config = config
        self.eps = 1e-3
        use_vgg = use_pretrained_vgg and (config.GENERATOR_BACKBONE.lower() != "mock_tiny")
        self.perceptual = MultiLayerVGGPerceptual(
            layers=getattr(config, "PERCEPTUAL_LAYERS", ["relu1_2", "relu2_2", "relu3_4", "relu4_4"]),
            use_pretrained=use_vgg,
        )

        b1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32) / 16.0
        b2d = (b1d[:, None] * b1d[None, :]).view(1, 1, 5, 5).repeat(3, 1, 1, 1)
        self.register_buffer("gauss5_kernel", b2d, persistent=False)

    def _gaussian_blur5(self, rgb: torch.Tensor) -> torch.Tensor:
        padded = F.pad(rgb, (2, 2, 2, 2), mode="reflect")
        return F.conv2d(padded, self.gauss5_kernel, groups=3)

    def _lowpass_4x(self, rgb_01: torch.Tensor) -> torch.Tensor:
        """Anti-aliased 4x low-pass structural filter (locks shapes & color while freeing 1-3px detail)."""
        h = self._gaussian_blur5(rgb_01.float())[:, :, ::2, ::2]
        return self._gaussian_blur5(h)[:, :, ::2, ::2]

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
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        stage = stage.lower()
        is_detail_stage = stage == "semantic_stage"
        spatial_w = self._build_spatial_focus_weight(action_map)

        if is_detail_stage and pred_01.shape[-2] >= 16 and pred_01.shape[-1] >= 16:
            pred_lp4 = self._lowpass_4x(pred_01)
            tgt_lp4 = self._lowpass_4x(target_01)
            w_lp4 = F.adaptive_avg_pool2d(spatial_w, output_size=pred_lp4.shape[-2:])
            w_lp4 = w_lp4 / w_lp4.mean(dim=(2, 3), keepdim=True).clamp(min=1e-4)
            loss_l1_lp = self.charbonnier_loss(pred_lp4, tgt_lp4, w_lp4)
            full_w = float(getattr(self.config, "LOSS_SEMANTIC_STAGE_FULL_L1_WEIGHT", 0.15))
            lp_w = float(getattr(self.config, "LOSS_SEMANTIC_STAGE_LP_L1_WEIGHT", 0.70))
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
            float(getattr(self.config, "LOSS_SEMANTIC_STAGE_LATENT_LP_WEIGHT", 0.25))
            if is_detail_stage
            else float(getattr(self.config, "LOSS_LATENT_WEIGHT", 0.50))
        )
        preserve_weight = float(getattr(self.config, "LOSS_PRESERVE_WEIGHT", 0.50))
        perc_w = (
            float(getattr(self.config, "LOSS_PIXEL_STAGE_PERCEPTUAL_WEIGHT", 0.10))
            if stage == "pixel_stage"
            else float(self.config.LOSS_PERCEPTUAL_WEIGHT)
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

        loss_csd = pred_01.new_zeros((), dtype=torch.float32)
        csd_w = float(getattr(self.config, "LOSS_CSD_WEIGHT", 0.0))
        if (
            stage in ("semantic_stage", "joint")
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
            "pres": float(loss_preserve.detach().item()),
            "csd": float(loss_csd.detach().item()),
            "total": float(total.detach().item()),
        }
        return total, metrics
