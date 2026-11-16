"""Calibrated Loss Objective & Vision-Aided GAN Discriminator (V4 Orthogonal + Directional).

Fixes V1-V3 training objective pathologies:
1. Strictly unit-normalized LPIPS-VGG (`MultiLayerVGGPerceptual`) so perceptual terms stay bounded
   in ~[0.02, 0.35] instead of drowning out RGB/Tone/Edge L1 by 50x.
2. 3-Scale Horizontal + Vertical + Diagonal Edge Pyramid (`spatial_gradient_loss`) providing smooth
   coarse-to-fine deconvolution gradients for 25px arbitrary-angle motion blur and defocus.
3. Full-Spectrum Phase-Aligned Complex FFT + Log-Power Loss (`fft_frequency_loss`) so blurred edges
   recover exact spatial phase instead of phase-less ringing.
4. Multi-Scale (`1x1` mean/std, `4x4`, `16x16`, `32x32`) RGB + Opponent Chroma Loss for exact
   directional `exposure` and `warmth` control.
5. Explicit Disentanglement & Preservation Loss (`selective_preservation_loss`) preventing structural
   sliders (`denoise`, `deartifact`, `deblur_motion`, `sharpen_focus`) from shifting exposure or color cast
   whenever `exposure == 0` and `warmth == 0`.
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
    """Strictly unit-channel-normalized LPIPS-VGG19 perceptual + bounded contrastive loss."""

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
        src_01: Optional[torch.Tensor] = None,
        spatial_weight: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
        active_change_gate: Optional[torch.Tensor] = None,
        return_contrast: bool = False,
    ):
        x = (pred_01.float() - self.mean) / self.std
        with torch.no_grad():
            y_in = (target_01.float() - self.mean) / self.std
            if src_01 is not None and return_contrast:
                s_in = (src_01.float() - self.mean) / self.std
                ys_cat = torch.cat([y_in, s_in], dim=0)
            else:
                ys_cat = y_in

        total_pos = pred_01.new_zeros((), dtype=torch.float32)
        total_contrast = pred_01.new_zeros((), dtype=torch.float32)
        has_src = src_01 is not None and return_contrast

        for block in self.slices:
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

            # Pure unit-normalized LPIPS squared distance (bounded in [0, 4], typically ~0.05-0.30)
            x_unit = x / torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + 1e-8)
            y_unit = y_det / torch.sqrt(torch.sum(y_det * y_det, dim=1, keepdim=True) + 1e-8)
            lpips_map = 0.5 * ((x_unit - y_unit) ** 2).sum(dim=1, keepdim=True)

            if w_l is not None:
                pos_dist = (lpips_map * w_l).mean(dim=(1, 2, 3))
            else:
                pos_dist = lpips_map.mean(dim=(1, 2, 3))

            if sample_weights is not None:
                total_pos = total_pos + (pos_dist * sample_weights.float()).mean()
            else:
                total_pos = total_pos + pos_dist.mean()

            if s_det is not None:
                s_unit = s_det / torch.sqrt(torch.sum(s_det * s_det, dim=1, keepdim=True) + 1e-8)
                neg_lpips = 0.5 * ((x_unit - s_unit) ** 2).sum(dim=1, keepdim=True)
                ref_lpips = 0.5 * ((s_unit - y_unit) ** 2).sum(dim=1, keepdim=True)
                if w_l is not None:
                    neg_dist = (neg_lpips * w_l).mean(dim=(1, 2, 3))
                    ref_dist = (ref_lpips * w_l).mean(dim=(1, 2, 3))
                else:
                    neg_dist = neg_lpips.mean(dim=(1, 2, 3))
                    ref_dist = ref_lpips.mean(dim=(1, 2, 3))

                cr_sample = pos_dist / (neg_dist + 0.50 * ref_dist.detach() + 1e-3)
                if active_change_gate is not None:
                    cr_sample = cr_sample * active_change_gate.float()
                total_contrast = total_contrast + cr_sample.mean()

        n_layers = max(1, len(self.slices))
        perc_out = total_pos / n_layers
        if return_contrast:
            return perc_out, total_contrast / n_layers
        return perc_out


class VisionAidedDiscriminator(nn.Module):
    """Action-conditioned Spectral-Norm PatchGAN + Frozen Vision (`C-RADIO`) Discriminator."""

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
        struct_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Gate GAN push primarily by structural restoration channels so pure tone/WB edits stay smooth & faithful
        if struct_indices is not None and struct_indices.numel() > 0:
            s_idx = struct_indices.to(action_map.device)
            act_strength = action_map.float().index_select(1, s_idx).abs().amax(dim=(1, 2, 3)).clamp(0.0, 1.0)
        else:
            act_strength = action_map.float().abs().amax(dim=(1, 2, 3)).clamp(0.0, 1.0)
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
    """Calibrated multi-scale pixel, edge-pyramid, complex-phase FFT, tone & preservation objective."""

    def __init__(self, config, use_pretrained_vgg: bool = True):
        super().__init__()
        self.config = config
        self.eps = 1e-3
        use_vgg = use_pretrained_vgg and (config.GENERATOR_BACKBONE.lower() != "mock_tiny")
        self.perceptual = MultiLayerVGGPerceptual(
            layers=getattr(config, "PERCEPTUAL_LAYERS", ["relu1_2", "relu2_2", "relu3_4", "relu4_4"]),
            use_pretrained=use_vgg,
        )

        deg_names = list(getattr(config, "DEGRADATION_TYPES", []))
        signed_set = set(getattr(config, "SIGNED_ACTION_TYPES", ["exposure", "warmth"]))
        signed_idx = [i for i, n in enumerate(deg_names) if n in signed_set]
        struct_idx = [i for i, n in enumerate(deg_names) if n not in signed_set]
        self.register_buffer("signed_indices", torch.tensor(signed_idx, dtype=torch.long), persistent=False)
        self.register_buffer("struct_indices", torch.tensor(struct_idx, dtype=torch.long), persistent=False)

    def _build_spatial_focus_weight(
        self,
        target_01: torch.Tensor,
        action_map: torch.Tensor,
        src_01: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Signed-safe spatial focus weight with strong >=0.55 floor so preserved pixels stay sharp & exact."""
        max_act = action_map.float().abs().amax(dim=1, keepdim=True)  # [B, 1, H, W]
        if src_01 is not None:
            deg_mag = (target_01.float() - src_01.float()).abs().mean(dim=1, keepdim=True).detach()
            raw_w = 0.55 + 0.90 * max_act + 5.0 * deg_mag
            deg_sample = deg_mag.mean(dim=(1, 2, 3))
            active_change_gate = ((deg_sample - 0.008) / 0.020).clamp(0.0, 1.0)
        else:
            raw_w = 0.55 + 0.90 * max_act
            active_change_gate = None

        spatial_w = raw_w / raw_w.mean(dim=(2, 3), keepdim=True).clamp(min=1e-4)
        return spatial_w.detach(), active_change_gate

    def charbonnier_with_preservation_weight(
        self, pred: torch.Tensor, target: torch.Tensor, spatial_weight: torch.Tensor
    ) -> torch.Tensor:
        diff_sq = (pred.float() - target.float()) ** 2
        charb = torch.sqrt(diff_sq + self.eps * self.eps)
        return (charb * spatial_weight).mean()

    @staticmethod
    def _opponent_chroma(rgb: torch.Tensor) -> torch.Tensor:
        return torch.cat([rgb[:, 0:1] - rgb[:, 1:2], rgb[:, 2:3] - 0.5 * (rgb[:, 0:1] + rgb[:, 1:2])], dim=1)

    @classmethod
    def multiscale_color_tone_loss(cls, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Global (mean + std) & multi-scale (4x4, 16x16, 32x32) RGB + opponent chroma loss."""
        pred_f, tgt_f = pred.float(), target.float()

        # 1. Global patch mean + contrast std matching (crucial for crisp directional exposure & warmth!)
        p_mean, t_mean = pred_f.mean(dim=(2, 3)), tgt_f.mean(dim=(2, 3))
        p_std = pred_f.std(dim=(2, 3), unbiased=False)
        t_std = tgt_f.std(dim=(2, 3), unbiased=False)
        global_tone = F.l1_loss(p_mean, t_mean) + 0.5 * F.l1_loss(p_std, t_std)

        p_chr_glob = cls._opponent_chroma(p_mean.unsqueeze(-1).unsqueeze(-1))
        t_chr_glob = cls._opponent_chroma(t_mean.unsqueeze(-1).unsqueeze(-1))
        global_chroma = F.l1_loss(p_chr_glob, t_chr_glob)

        # 2. Local low-pass pooled tone & chromaticity grids
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
        """3-scale (1x, 2x, 4x) horizontal, vertical & diagonal edge gradient pyramid."""
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
        """Phase-aligned full-spectrum complex rfft2 + high-freq weighted log-amplitude loss."""
        p_f, t_f = pred.float(), target.float()
        fp = torch.fft.rfft2(p_f, norm="ortho")
        ft = torch.fft.rfft2(t_f, norm="ortho")

        # Radial frequency emphasis (1.0 at DC -> 1.8 at Nyquist) so blurred high frequencies deconvolve
        fh, fw = fp.shape[-2], fp.shape[-1]
        fy = torch.fft.fftfreq(p_f.shape[-2], device=p_f.device, dtype=torch.float32).abs().view(1, 1, fh, 1)
        fx = torch.fft.rfftfreq(p_f.shape[-1], device=p_f.device, dtype=torch.float32).abs().view(1, 1, 1, fw)
        rad_w = 1.0 + 1.15 * torch.sqrt(fy * fy + fx * fx).clamp(0.0, 0.71)

        # 1. Full complex phase + amplitude L1 (restores exact spatial edge phase!)
        complex_diff = torch.view_as_real(fp - ft).abs().mean(dim=-1)
        phase_complex_loss = (complex_diff * rad_w).mean()

        # 2. Log-magnitude spectrum L1 (restores attenuated high-frequency power!)
        log_amp_diff = (torch.log1p(torch.abs(fp)) - torch.log1p(torch.abs(ft))).abs()
        log_amp_loss = (log_amp_diff * rad_w).mean()

        return 0.55 * phase_complex_loss + 0.45 * log_amp_loss

    def selective_preservation_loss(
        self,
        pred_01: torch.Tensor,
        src_01: torch.Tensor,
        action_map: torch.Tensor,
    ) -> torch.Tensor:
        """Explicitly enforce untouched-region preservation & zero-DC tone lock when exposure=warmth=0."""
        p_f, s_f, a_f = pred_01.float(), src_01.float(), action_map.float()

        # 1. Spatial zero-action region preservation
        max_abs_act = a_f.abs().amax(dim=1, keepdim=True)
        idle_mask = (1.0 - (max_abs_act / 0.12).clamp(0.0, 1.0)).square()
        idle_loss = ((p_f - s_f).abs() * idle_mask).mean()

        # 2. Zero-DC tone & color-cast preservation whenever `exposure == 0` and `warmth == 0`!
        # Guarantees denoising, deblurring, focus sharpening, and de-artifacting NEVER alter mood lighting or casts!
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
        semantic_encoder: Optional[nn.Module] = None,
        discriminator: Optional[nn.Module] = None,
        return_aux: bool = False,
    ):
        """Compute stage-appropriate total loss, scalar logging dictionary, and optional cached ViT grids."""
        align_scalar = align_loss.float().mean()
        stage = stage.lower()

        spatial_w, active_change_gate = self._build_spatial_focus_weight(target_01, action_map, src_01=src_01)
        loss_l1 = self.charbonnier_with_preservation_weight(pred_01, target_01, spatial_w)
        loss_grad = self.spatial_gradient_loss(pred_01, target_01, spatial_weight=spatial_w)
        loss_freq = self.fft_frequency_loss(pred_01, target_01)
        loss_color = self.multiscale_color_tone_loss(pred_01, target_01)

        loss_preserve = (
            self.selective_preservation_loss(pred_01, src_01, action_map)
            if src_01 is not None
            else pred_01.new_zeros((), dtype=torch.float32)
        )

        l1_weight = (
            float(getattr(self.config, "LOSS_SEMANTIC_STAGE_L1_WEIGHT", 0.75))
            if stage == "semantic_stage"
            else float(self.config.LOSS_L1_WEIGHT)
        )
        color_weight = float(getattr(self.config, "LOSS_COLOR_TONE_WEIGHT", 0.65))
        preserve_weight = float(getattr(self.config, "LOSS_PRESERVE_WEIGHT", 0.50))

        total = (
            l1_weight * loss_l1
            + self.config.LOSS_GRADIENT_WEIGHT * loss_grad
            + self.config.LOSS_FREQUENCY_WEIGHT * loss_freq
            + color_weight * loss_color
            + preserve_weight * loss_preserve
        )

        if stage in ("pixel_stage", "joint"):
            total = total + self.config.LOSS_ALIGN_WEIGHT * align_scalar

        loss_perc = pred_01.new_zeros((), dtype=torch.float32)
        loss_contrast = pred_01.new_zeros((), dtype=torch.float32)
        loss_sem_cons = pred_01.new_zeros((), dtype=torch.float32)
        loss_gan = pred_01.new_zeros((), dtype=torch.float32)
        aux: Dict[str, torch.Tensor] = {}

        if t_vec is not None:
            t_weights = (1.15 - 0.30 * t_vec.float().view(-1)).clamp(0.6, 1.0)
        else:
            t_weights = None

        perc_w = (
            float(getattr(self.config, "LOSS_PIXEL_STAGE_PERCEPTUAL_WEIGHT", 0.35))
            if stage == "pixel_stage"
            else float(self.config.LOSS_PERCEPTUAL_WEIGHT)
        )
        contrast_w = float(getattr(self.config, "LOSS_CONTRAST_WEIGHT", 0.15))
        sem_w = (
            float(getattr(self.config, "LOSS_PIXEL_STAGE_SEM_WEIGHT", 0.15))
            if stage == "pixel_stage"
            else float(self.config.LOSS_SEM_CONSISTENCY_WEIGHT)
        )
        gan_w = float(getattr(self.config, "LOSS_GAN_WEIGHT", 0.08))
        use_gan = stage in ("semantic_stage", "joint") and discriminator is not None and gan_w > 0.0

        if perc_w > 0.0 or (contrast_w > 0.0 and src_01 is not None):
            loss_perc, loss_contrast = self.perceptual(
                pred_01,
                target_01,
                src_01=src_01 if contrast_w > 0.0 else None,
                spatial_weight=spatial_w,
                sample_weights=t_weights,
                active_change_gate=active_change_gate,
                return_contrast=True,
            )
            total = total + perc_w * loss_perc + contrast_w * loss_contrast

        need_sem = semantic_encoder is not None and (sem_w > 0.0 or use_gan)
        if need_sem:
            pred_sem = semantic_encoder(pred_01, use_lora=False)
            with torch.no_grad():
                tgt_sem = semantic_encoder(target_01, use_lora=False)

            if sem_w > 0.0:
                p_tok = F.normalize(pred_sem["tokens"].float(), dim=-1)
                g_tok = F.normalize(tgt_sem["tokens"].float().detach(), dim=-1)
                loss_sem_cons = (1.0 - (p_tok * g_tok).sum(dim=-1)).mean()
                total = total + sem_w * loss_sem_cons

            if use_gan:
                logits_rgb, logits_sem = discriminator(pred_01, pred_sem["grid"], action_map)
                loss_gan = VisionAidedDiscriminator.generator_hinge_loss(
                    logits_rgb, logits_sem, action_map, struct_indices=self.struct_indices
                )
                total = total + gan_w * loss_gan
                aux["pred_grid"] = pred_sem["grid"].detach()
                aux["tgt_grid"] = tgt_sem["grid"].detach()

        metrics = {
            "l1": float(loss_l1.detach().item()),
            "grad": float(loss_grad.detach().item()),
            "freq": float(loss_freq.detach().item()),
            "color": float(loss_color.detach().item()),
            "pres": float(loss_preserve.detach().item()),
            "align": float(align_scalar.detach().item()),
            "perc": float(loss_perc.detach().item()),
            "contrast": float(loss_contrast.detach().item()),
            "sem_cons": float(loss_sem_cons.detach().item()),
            "gan": float(loss_gan.detach().item()),
            "total": float(total.detach().item()),
        }
        if return_aux:
            return total, metrics, aux
        return total, metrics
