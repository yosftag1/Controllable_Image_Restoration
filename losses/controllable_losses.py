"""Artifact-Free Natural-Manifold Loss Objective (V5 Universal eVAE + Target-Anchored CSD + Separated HYPIR).

Key V5 Supervision Protections:
1. Direct Clean-Latent Target Supervision (`latent_reconstruction_loss`) + Universal `eVAE` Skip
   Feature Alignment (`LOSS_EVAE_SKIP_WEIGHT`): teaches spatial realignment across both latent and
   high-res skip paths for any slider without needing a hardcoded blur estimator.
2. Zero-Extra-VRAM Target-Anchored Classifier Score Distillation (`LOSS_CSD_WEIGHT` in Stage 3):
   uses the already-loaded frozen `sd-2.1-base` UNet (`LoRA OFF`) with negative prompts (`"blurry,
   wavy fake texture, ringing, oversharpened"`) anchored to `z_tgt` (`PiSA-SR` + `TSD-SR`).
3. 4-Level Binomial Anti-Aliased Laplacian Pyramid + Asymmetric High-Frequency Artifact Guard.
4. Separate Experimental `HYPIR` Relativistic Pairing GAN (`LOSS_RPGAN_WEIGHT`, disabled by default;
   isolated behind `HYPIRConfig` / `--hypir-mode` for standalone benchmarking).
"""

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
    """Full-res (`relu1_2..relu4_4`) LPIPS + DISTS Local Patch Texture Similarity (mean & std matching)."""

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
        src_01: Optional[torch.Tensor] = None,
        spatial_weight: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
        active_change_gate: Optional[torch.Tensor] = None,
        return_contrast: bool = False,
        use_dists_texture: bool = False,
    ):
        # In Stage 3 (`use_dists_texture=True`), feed crisp unblurred RGB so full-res `relu1_2` (256x256)
        # sees 1-2px citrus pores, feather barbs, and leaf veins without binomial pre-blurring!
        p_rgb = pred_01.float() if use_dists_texture else self._antialias_rgb(pred_01)
        x = (p_rgb - self.mean) / self.std
        with torch.no_grad():
            t_rgb = target_01.float() if use_dists_texture else self._antialias_rgb(target_01)
            y_in = (t_rgb - self.mean) / self.std
            has_src = src_01 is not None and return_contrast
            if has_src:
                s_rgb = src_01.float() if use_dists_texture else self._antialias_rgb(src_01)
                s_in = (s_rgb - self.mean) / self.std
                ys_cat = torch.cat([y_in, s_in], dim=0)
            else:
                ys_cat = y_in

        total_pos = pred_01.new_zeros((), dtype=torch.float32)
        total_contrast = pred_01.new_zeros((), dtype=torch.float32)

        for sw, block in zip(self.slice_weights, self.slices):
            x = block(x)
            with torch.no_grad():
                ys_cat = block(ys_cat)
            if has_src:
                y_det, s_det = ys_cat.detach().chunk(2, dim=0)
            else:
                y_det, s_det = ys_cat.detach(), None

            if spatial_weight is not None:
                w_l = F.adaptive_avg_pool2d(spatial_weight.float(), output_size=x.shape[-2:])
                w_l = w_l / w_l.mean(dim=(2, 3), keepdim=True).clamp(min=1e-4)
            else:
                w_l = None

            x_unit = x / torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + 1e-8)
            y_unit = y_det / torch.sqrt(torch.sum(y_det * y_det, dim=1, keepdim=True) + 1e-8)
            lpips_map = 0.5 * ((x_unit - y_unit) ** 2).sum(dim=1, keepdim=True)

            if w_l is not None:
                pos_dist = (lpips_map * w_l).mean(dim=(1, 2, 3))
            else:
                pos_dist = lpips_map.mean(dim=(1, 2, 3))

            if use_dists_texture:
                # DISTS Local Patch Texture Matching (8x8 & 16x16 mean + std):
                # Rewards real local micro-contrast (pores, feathers, leaf fibers) matching Clean GT
                # WITHOUT penalizing 1-2px subpixel shifts (breaks the MMSE double-penalty blur trap!)
                tex_dist = pred_01.new_zeros((x.shape[0],), dtype=torch.float32)
                win_cnt = 0
                ch_scale = y_det.abs().mean(dim=(0, 2, 3), keepdim=True).clamp(min=1e-2)
                xn = x / ch_scale
                yn = y_det / ch_scale
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

            if sample_weights is not None:
                total_pos = total_pos + sw * (pos_dist * sample_weights.float()).mean()
            else:
                total_pos = total_pos + sw * pos_dist.mean()

            if s_det is not None:
                s_unit = s_det / torch.sqrt(torch.sum(s_det * s_det, dim=1, keepdim=True) + 1e-8)
                neg_lpips = 0.5 * ((x_unit - s_unit) ** 2).sum(dim=1, keepdim=True)
                ref_lpips = 0.5 * ((s_unit - y_unit) ** 2).sum(dim=1, keepdim=True)
                neg_dist = neg_lpips.mean(dim=(1, 2, 3))
                ref_dist = ref_lpips.mean(dim=(1, 2, 3))
                cr_sample = pos_dist / (neg_dist + 0.50 * ref_dist.detach() + 1e-3)
                if active_change_gate is not None:
                    cr_sample = cr_sample * active_change_gate.float()
                total_contrast = total_contrast + sw * cr_sample.mean()

        if return_contrast:
            return total_pos, total_contrast
        return total_pos


class VisionAidedDiscriminator(nn.Module):
    """Spectral-Norm Multi-Scale Vision Discriminator supporting both Hinge & HYPIR Relativistic RpGAN."""

    def __init__(self, sem_dim: int, num_actions: int, base_ch: int = 64):
        super().__init__()
        rgb_in = 3 + num_actions
        self.rgb_net = nn.Sequential(
            _spectral_norm(nn.Conv2d(rgb_in, base_ch, kernel_size=3, stride=1, padding=1)),
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
    def _act_strength(action_map: torch.Tensor, struct_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if struct_indices is not None and struct_indices.numel() > 0:
            s_idx = struct_indices.to(action_map.device)
            return action_map.float().index_select(1, s_idx).abs().amax(dim=(1, 2, 3)).clamp(0.0, 1.0)
        return action_map.float().abs().amax(dim=(1, 2, 3)).clamp(0.0, 1.0)

    @classmethod
    def generator_hinge_loss(
        cls,
        logits_rgb: torch.Tensor,
        logits_sem: torch.Tensor,
        action_map: torch.Tensor,
        struct_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        act_strength = cls._act_strength(action_map, struct_indices)
        g_rgb = (-logits_rgb.float()).mean(dim=(1, 2, 3))
        g_sem = (-logits_sem.float()).mean(dim=(1, 2, 3))
        return (0.5 * (g_rgb + g_sem) * act_strength).mean()

    @classmethod
    def generator_rpgan_loss(
        cls,
        fake_rgb_logits: torch.Tensor,
        fake_sem_logits: torch.Tensor,
        real_rgb_logits: torch.Tensor,
        real_sem_logits: torch.Tensor,
        action_map: torch.Tensor,
        struct_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """HYPIR Relativistic Pairing Generator Loss: `softplus(-(D(pred) - D(gt)))`."""
        act_strength = cls._act_strength(action_map, struct_indices)
        rel_rgb = F.softplus(-(fake_rgb_logits.float() - real_rgb_logits.detach().float())).mean(dim=(1, 2, 3))
        rel_sem = F.softplus(-(fake_sem_logits.float() - real_sem_logits.detach().float())).mean(dim=(1, 2, 3))
        return (0.5 * (rel_rgb + rel_sem) * act_strength).mean()

    @staticmethod
    def discriminator_hinge_loss(
        disc_module: nn.Module,
        real_rgb: torch.Tensor,
        fake_rgb: torch.Tensor,
        real_grid: torch.Tensor,
        fake_grid: torch.Tensor,
        action_map: torch.Tensor,
        relativistic: bool = False,
    ) -> torch.Tensor:
        cat_rgb = torch.cat([real_rgb.detach(), fake_rgb.detach()], dim=0)
        cat_grid = torch.cat([real_grid.detach(), fake_grid.detach()], dim=0)
        cat_act = torch.cat([action_map.detach(), action_map.detach()], dim=0)
        logits_rgb, logits_sem = disc_module(cat_rgb, cat_grid, cat_act)

        real_r, fake_r = logits_rgb.float().chunk(2, dim=0)
        real_s, fake_s = logits_sem.float().chunk(2, dim=0)
        if relativistic:
            loss_r = F.softplus(-(real_r - fake_r)).mean()
            loss_s = F.softplus(-(real_s - fake_s)).mean()
            return 0.5 * (loss_r + loss_s)
        loss_r = 0.5 * (F.relu(1.0 - real_r).mean() + F.relu(1.0 + fake_r).mean())
        loss_s = 0.5 * (F.relu(1.0 - real_s).mean() + F.relu(1.0 + fake_s).mean())
        return 0.5 * (loss_r + loss_s)


class ControllableRestorationLoss(nn.Module):
    """V5 Latent + Universal eVAE + Laplacian Pyramid + Target-Anchored CSD Objective."""

    def __init__(self, config, use_pretrained_vgg: bool = True):
        super().__init__()
        self.config = config
        self.eps = 1e-3
        use_vgg = use_pretrained_vgg and (config.GENERATOR_BACKBONE.lower() != "mock_tiny")
        self.perceptual = MultiLayerVGGPerceptual(
            layers=getattr(config, "PERCEPTUAL_LAYERS", ["relu2_2", "relu3_4", "relu4_4"]),
            use_pretrained=use_vgg,
        )

        deg_names = list(getattr(config, "DEGRADATION_TYPES", []))
        signed_set = set(getattr(config, "SIGNED_ACTION_TYPES", ["exposure", "warmth"]))
        local_clean_set = {"denoise", "deartifact"}
        signed_idx = [i for i, n in enumerate(deg_names) if n in signed_set]
        struct_idx = [i for i, n in enumerate(deg_names) if n not in signed_set]
        deconv_idx = [i for i, n in enumerate(deg_names) if (n not in signed_set) and (n not in local_clean_set)]
        self.register_buffer("signed_indices", torch.tensor(signed_idx, dtype=torch.long), persistent=False)
        self.register_buffer("struct_indices", torch.tensor(struct_idx, dtype=torch.long), persistent=False)
        self.register_buffer("deconv_indices", torch.tensor(deconv_idx, dtype=torch.long), persistent=False)

        b1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32) / 16.0
        b2d = (b1d[:, None] * b1d[None, :]).view(1, 1, 5, 5).repeat(3, 1, 1, 1)
        self.register_buffer("gauss5_kernel", b2d, persistent=False)

    def _gaussian_blur5(self, rgb: torch.Tensor) -> torch.Tensor:
        padded = F.pad(rgb, (2, 2, 2, 2), mode="reflect")
        return F.conv2d(padded, self.gauss5_kernel, groups=3)

    def _build_spatial_focus_weight(
        self,
        target_01: torch.Tensor,
        action_map: torch.Tensor,
        src_01: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        a_f = action_map.float()
        max_act = a_f.abs().amax(dim=1, keepdim=True)
        if src_01 is not None:
            deg_mag = (target_01.float() - src_01.float()).abs().mean(dim=1, keepdim=True).detach()
            raw_w = 0.60 + 0.85 * max_act + 4.5 * deg_mag
            deg_sample = deg_mag.mean(dim=(1, 2, 3))
            active_change_gate = ((deg_sample - 0.008) / 0.020).clamp(0.0, 1.0)
        else:
            raw_w = 0.60 + 0.85 * max_act
            active_change_gate = None

        # Focus deconvolution supervision directly onto sharp GT edges, fur strands & object contours!
        if self.deconv_indices.numel() > 0 and target_01.shape[-2] >= 8 and target_01.shape[-1] >= 8:
            d_idx = self.deconv_indices.to(a_f.device)
            deconv_act = a_f.index_select(1, d_idx).abs().amax(dim=1, keepdim=True).clamp(0.0, 1.0)
            tgt_f = target_01.float().detach()
            tgt_edge = (tgt_f - self._gaussian_blur5(tgt_f)).abs().mean(dim=1, keepdim=True)
            raw_w = raw_w + 7.5 * deconv_act * tgt_edge

        spatial_w = raw_w / raw_w.mean(dim=(2, 3), keepdim=True).clamp(min=1e-4)
        return spatial_w.detach(), active_change_gate

    def charbonnier_with_preservation_weight(
        self, pred: torch.Tensor, target: torch.Tensor, spatial_weight: torch.Tensor
    ) -> torch.Tensor:
        diff_sq = (pred.float() - target.float()) ** 2
        charb = torch.sqrt(diff_sq + self.eps * self.eps)
        return (charb * spatial_weight).mean()

    def _lowpass_4x(self, rgb_01: torch.Tensor) -> torch.Tensor:
        """Anti-aliased 4x low-pass structural filter (locks shapes & color while freeing 1-3px pores!)."""
        h = self._gaussian_blur5(rgb_01.float())[:, :, ::2, ::2]
        return self._gaussian_blur5(h)[:, :, ::2, ::2]

    def latent_reconstruction_loss(
        self,
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
            l_val = (charb * w_z).mean()
        else:
            l_val = charb.mean()

        if lowpass_only:
            return l_val
        dx = ((zp[:, :, :, 1:] - zp[:, :, :, :-1]) - (zt[:, :, :, 1:] - zt[:, :, :, :-1])).abs().mean()
        dy = ((zp[:, :, 1:, :] - zp[:, :, :-1, :]) - (zt[:, :, 1:, :] - zt[:, :, :-1, :])).abs().mean()
        return l_val + 0.25 * 0.5 * (dx + dy)

    def laplacian_pyramid_loss(
        self,
        pred_01: torch.Tensor,
        target_01: torch.Tensor,
        spatial_weight: Optional[torch.Tensor] = None,
        detail_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cur_p = pred_01.float()
        cur_t = target_01.float()
        cur_w = spatial_weight.float() if spatial_weight is not None else None

        pyr_loss = cur_p.new_zeros((), dtype=torch.float32)
        hf_guard = cur_p.new_zeros((), dtype=torch.float32)
        # In Stage 3 (`detail_mode=True`), zero out finest L1 bands (lvl 0 & 1) so shifted 1-3px pores
        # aren't smoothed away by the double-penalty effect; keep coarse structural octaves (lvl 2 & 3)!
        level_weights = (0.0, 0.0, 0.55, 0.45) if detail_mode else (0.35, 0.30, 0.20, 0.15)
        w_sum = 0.0

        for lvl, lw in enumerate(level_weights):
            if cur_p.shape[-2] < 8 or cur_p.shape[-1] < 8:
                break
            blur_p = self._gaussian_blur5(cur_p)
            blur_t = self._gaussian_blur5(cur_t)
            band_p = cur_p - blur_p
            band_t = cur_t - blur_t

            if lw > 0.0:
                band_charb = torch.sqrt((band_p - band_t) ** 2 + self.eps * self.eps)
                if cur_w is not None:
                    pyr_loss = pyr_loss + lw * (band_charb * cur_w).mean()
                else:
                    pyr_loss = pyr_loss + lw * band_charb.mean()
                w_sum += lw

            if lvl <= 1:
                p_energy = band_p.abs().mean(dim=1, keepdim=True)
                t_energy = band_t.abs().mean(dim=1, keepdim=True).detach()
                # Allow natural GT-level crispness in detail_mode while still blocking extreme spike artifacts
                if detail_mode:
                    excess_hf = F.relu(p_energy - 1.18 * t_energy - 0.035)
                else:
                    excess_hf = F.relu(p_energy - t_energy - 0.010)
                hf_guard = hf_guard + (0.65 if lvl == 0 else 0.35) * excess_hf.mean()

            cur_p = blur_p[:, :, ::2, ::2]
            cur_t = blur_t[:, :, ::2, ::2]
            if cur_w is not None:
                cur_w = F.avg_pool2d(cur_w, kernel_size=2, stride=2)

        return pyr_loss / max(1e-6, w_sum), hf_guard

    @staticmethod
    def _opponent_chroma(rgb: torch.Tensor) -> torch.Tensor:
        return torch.cat([rgb[:, 0:1] - rgb[:, 1:2], rgb[:, 2:3] - 0.5 * (rgb[:, 0:1] + rgb[:, 1:2])], dim=1)

    @classmethod
    def multiscale_color_tone_loss(cls, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f, tgt_f = pred.float(), target.float()

        p_mean, t_mean = pred_f.mean(dim=(2, 3)), tgt_f.mean(dim=(2, 3))
        p_std = pred_f.std(dim=(2, 3), unbiased=False)
        t_std = tgt_f.std(dim=(2, 3), unbiased=False)
        global_tone = F.l1_loss(p_mean, t_mean) + 0.5 * F.l1_loss(p_std, t_std)

        p_chr_glob = cls._opponent_chroma(p_mean.unsqueeze(-1).unsqueeze(-1))
        t_chr_glob = cls._opponent_chroma(t_mean.unsqueeze(-1).unsqueeze(-1))
        global_chroma = F.l1_loss(p_chr_glob, t_chr_glob)

        spatial_total = pred_f.new_zeros((), dtype=torch.float32)
        scales = 0
        for pool_k in (4, 16, 32):
            if pred_f.shape[-2] >= pool_k and pred_f.shape[-1] >= pool_k:
                p_low = F.avg_pool2d(pred_f, kernel_size=pool_k, stride=pool_k)
                t_low = F.avg_pool2d(tgt_f, kernel_size=pool_k, stride=pool_k)
                l_rgb = F.l1_loss(p_low, t_low)
                l_chr = F.l1_loss(cls._opponent_chroma(p_low), cls._opponent_chroma(t_low))
                spatial_total = spatial_total + 0.5 * (l_rgb + l_chr)
                scales += 1

        local_part = spatial_total / max(1, scales)
        return 0.45 * (global_tone + global_chroma) + 0.55 * local_part

    @staticmethod
    def _single_scale_grad_loss(
        p: torch.Tensor, t: torch.Tensor, w: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dx = ((p[:, :, :, 1:] - p[:, :, :, :-1]) - (t[:, :, :, 1:] - t[:, :, :, :-1])).abs()
        dy = ((p[:, :, 1:, :] - p[:, :, :-1, :]) - (t[:, :, 1:, :] - t[:, :, :-1, :])).abs()
        dd1 = ((p[:, :, 1:, 1:] - p[:, :, :-1, :-1]) - (t[:, :, 1:, 1:] - t[:, :, :-1, :-1])).abs()
        dd2 = ((p[:, :, 1:, :-1] - p[:, :, :-1, 1:]) - (t[:, :, 1:, :-1] - t[:, :, :-1, 1:])).abs()
        if w is not None:
            wx = 0.5 * (w[:, :, :, 1:] + w[:, :, :, :-1])
            wy = 0.5 * (w[:, :, 1:, :] + w[:, :, :-1, :])
            wd = 0.5 * (w[:, :, 1:, 1:] + w[:, :, :-1, :-1])
            hv = 0.5 * ((dx * wx).mean() + (dy * wy).mean())
            diag = 0.5 * ((dd1 * wd).mean() + (dd2 * wd).mean())
            return 0.70 * hv + 0.30 * diag
        hv = 0.5 * (dx.mean() + dy.mean())
        diag = 0.5 * (dd1.mean() + dd2.mean())
        return 0.70 * hv + 0.30 * diag

    @classmethod
    def spatial_gradient_loss(
        cls, pred: torch.Tensor, target: torch.Tensor, spatial_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        p, t = pred.float(), target.float()
        w = spatial_weight.float() if spatial_weight is not None else None
        scale_weights = (0.50, 0.30, 0.20)
        total = p.new_zeros((), dtype=torch.float32)
        w_sum = 0.0
        for idx, sw in enumerate(scale_weights):
            if idx > 0:
                if p.shape[-2] < 8 or p.shape[-1] < 8:
                    break
                p = F.avg_pool2d(p, kernel_size=2, stride=2)
                t = F.avg_pool2d(t, kernel_size=2, stride=2)
                if w is not None:
                    w = F.avg_pool2d(w, kernel_size=2, stride=2)
            total = total + sw * cls._single_scale_grad_loss(p, t, w)
            w_sum += sw
        return total / max(1e-6, w_sum)

    @staticmethod
    def fft_frequency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fp = torch.fft.rfft2(pred.float(), norm="ortho")
        ft = torch.fft.rfft2(target.float(), norm="ortho")
        return torch.view_as_real(fp - ft).abs().mean()

    def selective_preservation_loss(
        self,
        pred_01: torch.Tensor,
        src_01: torch.Tensor,
        action_map: torch.Tensor,
    ) -> torch.Tensor:
        p_f, s_f, a_f = pred_01.float(), src_01.float(), action_map.float()

        max_abs_act = a_f.abs().amax(dim=1, keepdim=True)
        idle_mask = (1.0 - (max_abs_act / 0.12).clamp(0.0, 1.0)).square()
        idle_loss = ((p_f - s_f).abs() * idle_mask).mean()

        tone_preserve_loss = p_f.new_zeros((), dtype=torch.float32)
        if self.signed_indices.numel() > 0 and p_f.shape[-2] >= 16 and p_f.shape[-1] >= 16:
            s_idx = self.signed_indices.to(a_f.device)
            tone_act = a_f.index_select(1, s_idx).abs().amax(dim=1, keepdim=True)
            tone_act_16 = F.avg_pool2d(tone_act, kernel_size=16, stride=16)
            lock_tone = (1.0 - (tone_act_16 / 0.06).clamp(0.0, 1.0)).square()

            p_16 = F.avg_pool2d(p_f, kernel_size=16, stride=16)
            s_16 = F.avg_pool2d(s_f, kernel_size=16, stride=16)
            rgb_drift = ((p_16 - s_16).abs() * lock_tone).mean()
            chr_drift = ((self._opponent_chroma(p_16) - self._opponent_chroma(s_16)).abs() * lock_tone).mean()
            tone_preserve_loss = 0.5 * (rgb_drift + chr_drift)

        return idle_loss + tone_preserve_loss

    def forward(
        self,
        pred_01: torch.Tensor,
        target_01: torch.Tensor,
        action_map: torch.Tensor,
        align_loss: torch.Tensor,
        stage: str = "pixel_stage",
        t_vec: Optional[torch.Tensor] = None,
        src_01: Optional[torch.Tensor] = None,
        hat_z_0: Optional[torch.Tensor] = None,
        z_tgt: Optional[torch.Tensor] = None,
        evae_loss: Optional[torch.Tensor] = None,
        restorer_model: Optional[nn.Module] = None,
        semantic_encoder: Optional[nn.Module] = None,
        discriminator: Optional[nn.Module] = None,
        return_aux: bool = False,
    ):
        align_scalar = align_loss.float().mean()
        evae_scalar = (
            evae_loss.float().mean()
            if evae_loss is not None
            else pred_01.new_zeros((), dtype=torch.float32)
        )
        stage = stage.lower()

        is_detail_stage = stage == "semantic_stage"
        spatial_w, active_change_gate = self._build_spatial_focus_weight(target_01, action_map, src_01=src_01)

        if is_detail_stage and pred_01.shape[-2] >= 16 and pred_01.shape[-1] >= 16:
            # PiSA-SR Detail Stage Principle: supervise pixel L1 & gradients ONLY on 4x low-pass structure
            # so 1-3px micro-textures (pores, feather barbs, leaf veins) are free from MMSE blur averaging!
            pred_lp4 = self._lowpass_4x(pred_01)
            tgt_lp4 = self._lowpass_4x(target_01)
            w_lp4 = F.adaptive_avg_pool2d(spatial_w, output_size=pred_lp4.shape[-2:])
            w_lp4 = w_lp4 / w_lp4.mean(dim=(2, 3), keepdim=True).clamp(min=1e-4)
            loss_l1 = self.charbonnier_with_preservation_weight(pred_lp4, tgt_lp4, w_lp4)
            loss_grad = self.spatial_gradient_loss(pred_lp4, tgt_lp4, spatial_weight=w_lp4)
            loss_freq = pred_01.new_zeros((), dtype=torch.float32)
        else:
            loss_l1 = self.charbonnier_with_preservation_weight(pred_01, target_01, spatial_w)
            loss_grad = self.spatial_gradient_loss(pred_01, target_01, spatial_weight=spatial_w)
            loss_freq = self.fft_frequency_loss(pred_01, target_01)

        loss_pyr, loss_hf_guard = self.laplacian_pyramid_loss(
            pred_01, target_01, spatial_weight=spatial_w, detail_mode=is_detail_stage
        )
        loss_color = self.multiscale_color_tone_loss(pred_01, target_01)

        loss_latent = (
            self.latent_reconstruction_loss(
                hat_z_0, z_tgt, spatial_weight=spatial_w, lowpass_only=is_detail_stage
            )
            if (hat_z_0 is not None and z_tgt is not None)
            else pred_01.new_zeros((), dtype=torch.float32)
        )
        loss_preserve = (
            self.selective_preservation_loss(pred_01, src_01, action_map)
            if src_01 is not None
            else pred_01.new_zeros((), dtype=torch.float32)
        )

        l1_weight = (
            float(getattr(self.config, "LOSS_SEMANTIC_STAGE_LP_L1_WEIGHT", 0.75))
            if is_detail_stage
            else float(self.config.LOSS_L1_WEIGHT)
        )
        lat_weight = (
            float(getattr(self.config, "LOSS_SEMANTIC_STAGE_LATENT_LP_WEIGHT", 0.30))
            if is_detail_stage
            else float(getattr(self.config, "LOSS_LATENT_WEIGHT", 0.85))
        )
        evae_weight = float(getattr(self.config, "LOSS_EVAE_SKIP_WEIGHT", 0.35))
        if is_detail_stage:
            evae_weight = 0.15 * evae_weight
        pyr_weight = (0.30 if is_detail_stage else 1.0) * float(getattr(self.config, "LOSS_PYRAMID_WEIGHT", 0.60))
        grad_weight = (0.50 if is_detail_stage else 1.0) * float(self.config.LOSS_GRADIENT_WEIGHT)
        freq_weight = 0.0 if is_detail_stage else float(self.config.LOSS_FREQUENCY_WEIGHT)
        hf_guard_weight = float(getattr(self.config, "LOSS_HF_GUARD_WEIGHT", 0.50))
        color_weight = float(getattr(self.config, "LOSS_COLOR_TONE_WEIGHT", 0.65))
        preserve_weight = float(getattr(self.config, "LOSS_PRESERVE_WEIGHT", 0.50))

        total = (
            l1_weight * loss_l1
            + lat_weight * loss_latent
            + evae_weight * evae_scalar
            + pyr_weight * loss_pyr
            + grad_weight * loss_grad
            + hf_guard_weight * loss_hf_guard
            + freq_weight * loss_freq
            + color_weight * loss_color
            + preserve_weight * loss_preserve
        )

        if stage in ("pixel_stage", "joint"):
            total = total + self.config.LOSS_ALIGN_WEIGHT * align_scalar

        loss_perc = pred_01.new_zeros((), dtype=torch.float32)
        loss_contrast = pred_01.new_zeros((), dtype=torch.float32)
        loss_sem_cons = pred_01.new_zeros((), dtype=torch.float32)
        loss_csd = pred_01.new_zeros((), dtype=torch.float32)
        loss_gan = pred_01.new_zeros((), dtype=torch.float32)
        aux: Dict[str, torch.Tensor] = {}

        if t_vec is not None:
            t_weights = (1.15 - 0.30 * t_vec.float().view(-1)).clamp(0.6, 1.0)
        else:
            t_weights = None

        perc_w = (
            float(getattr(self.config, "LOSS_PIXEL_STAGE_PERCEPTUAL_WEIGHT", 0.08))
            if stage == "pixel_stage"
            else float(self.config.LOSS_PERCEPTUAL_WEIGHT)
        )
        contrast_w = float(getattr(self.config, "LOSS_CONTRAST_WEIGHT", 0.0))
        sem_w = (
            float(getattr(self.config, "LOSS_PIXEL_STAGE_SEM_WEIGHT", 0.10))
            if stage == "pixel_stage"
            else float(self.config.LOSS_SEM_CONSISTENCY_WEIGHT)
        )

        csd_w = float(getattr(self.config, "LOSS_CSD_WEIGHT", 0.0))
        if (
            stage in ("semantic_stage", "joint")
            and csd_w > 0.0
            and restorer_model is not None
            and hat_z_0 is not None
            and z_tgt is not None
            and not bool(getattr(self.config, "HYPIR_MODE", False))
        ):
            loss_csd = restorer_model.compute_target_anchored_csd_loss(hat_z_0, z_tgt, action_map)
            total = total + csd_w * loss_csd

        rpgan_w = float(getattr(self.config, "LOSS_RPGAN_WEIGHT", 0.0))
        hinge_gan_w = float(getattr(self.config, "LOSS_GAN_WEIGHT", 0.0))
        active_gan_w = rpgan_w if rpgan_w > 0.0 else hinge_gan_w
        use_rpgan = rpgan_w > 0.0
        use_gan = stage in ("semantic_stage", "joint") and discriminator is not None and active_gan_w > 0.0

        use_dists = stage in ("semantic_stage", "joint")
        if perc_w > 0.0 or (contrast_w > 0.0 and src_01 is not None):
            need_cr = contrast_w > 0.0 and src_01 is not None
            if need_cr:
                loss_perc, loss_contrast = self.perceptual(
                    pred_01,
                    target_01,
                    src_01=src_01,
                    spatial_weight=spatial_w,
                    sample_weights=t_weights,
                    active_change_gate=active_change_gate,
                    return_contrast=True,
                    use_dists_texture=use_dists,
                )
                total = total + perc_w * loss_perc + contrast_w * loss_contrast
            else:
                loss_perc = self.perceptual(
                    pred_01,
                    target_01,
                    spatial_weight=spatial_w,
                    sample_weights=t_weights,
                    return_contrast=False,
                    use_dists_texture=use_dists,
                )
                total = total + perc_w * loss_perc

        need_sem = semantic_encoder is not None and (sem_w > 0.0 or use_gan)
        if need_sem:
            pred_sem = semantic_encoder(pred_01, use_lora=False)
            with torch.no_grad():
                tgt_sem = semantic_encoder(target_01, use_lora=False)

            if sem_w > 0.0:
                p_grid = F.normalize(pred_sem["grid"].float(), dim=1, eps=1e-6)
                g_grid = F.normalize(tgt_sem["grid"].float().detach(), dim=1, eps=1e-6)
                grid_cons = (1.0 - (p_grid * g_grid).sum(dim=1)).mean()
                p_tok = F.normalize(pred_sem["tokens"].float(), dim=-1, eps=1e-6)
                g_tok = F.normalize(tgt_sem["tokens"].float().detach(), dim=-1, eps=1e-6)
                tok_cons = (1.0 - (p_tok * g_tok).sum(dim=-1)).mean()
                loss_sem_cons = 0.60 * grid_cons + 0.40 * tok_cons
                total = total + sem_w * loss_sem_cons

            if use_gan:
                logits_rgb, logits_sem = discriminator(pred_01, pred_sem["grid"], action_map)
                if use_rpgan:
                    with torch.no_grad():
                        real_rgb_l, real_sem_l = discriminator(target_01, tgt_sem["grid"], action_map)
                    loss_gan = VisionAidedDiscriminator.generator_rpgan_loss(
                        logits_rgb,
                        logits_sem,
                        real_rgb_l,
                        real_sem_l,
                        action_map,
                        struct_indices=self.struct_indices,
                    )
                else:
                    loss_gan = VisionAidedDiscriminator.generator_hinge_loss(
                        logits_rgb, logits_sem, action_map, struct_indices=self.struct_indices
                    )
                total = total + active_gan_w * loss_gan
                aux["pred_grid"] = pred_sem["grid"].detach()
                aux["tgt_grid"] = tgt_sem["grid"].detach()
                aux["use_rpgan"] = use_rpgan

        metrics = {
            "l1": float(loss_l1.detach().item()),
            "lat": float(loss_latent.detach().item()),
            "evae": float(evae_scalar.detach().item()),
            "pyr": float(loss_pyr.detach().item()),
            "grad": float(loss_grad.detach().item()),
            "hf_g": float(loss_hf_guard.detach().item()),
            "freq": float(loss_freq.detach().item()),
            "color": float(loss_color.detach().item()),
            "pres": float(loss_preserve.detach().item()),
            "align": float(align_scalar.detach().item()),
            "perc": float(loss_perc.detach().item()),
            "csd": float(loss_csd.detach().item()),
            "contrast": float(loss_contrast.detach().item()),
            "sem_cons": float(loss_sem_cons.detach().item()),
            "gan": float(loss_gan.detach().item()),
            "total": float(total.detach().item()),
        }
        if return_aux:
            return total, metrics, aux
        return total, metrics
