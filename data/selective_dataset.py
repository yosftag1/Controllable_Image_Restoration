"""Seed-synchronized selective & spatially localized degradation synthesis (CPU-accelerated)."""

from dataclasses import dataclass, field
from pathlib import Path
import os
import random
from typing import Optional

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover - allows standalone NumPy/OpenCV pipeline tests
    torch = None
    _TorchDataset = object


@dataclass
class DegradationParams:
    """Frozen random realizations for a single image patch so source and partial targets align exactly."""

    height: int
    width: int
    noise_seed: int                 # Deterministic seed for lazy N(0, 1) realization (avoids RNG cost when unused!)
    motion_angle: float             # Degrees in [0, 180)
    color_gains_unit: np.ndarray    # (3,) float32 in [-1, 1]
    hue_shift_unit: float           # Float in [-1, 1]
    sat_loss_frac: float            # Float in [0.15, 1.0]
    underexpose: bool               # True = dark gamma (>1), False = bright/washed gamma (<1)
    brightness_frac: float          # Float in [0.15, 1.0]
    contrast_frac: float            # Float in [0.15, 1.0]
    _noise_cache: Optional[np.ndarray] = field(default=None, repr=False)

    def get_noise_unit(self) -> np.ndarray:
        """Lazily materialize the N(0, 1) sensor noise field only if gaussian_noise is active."""
        if self._noise_cache is None:
            nrng = np.random.default_rng(self.noise_seed)
            self._noise_cache = nrng.standard_normal((self.height, self.width, 3), dtype=np.float32)
        return self._noise_cache


class SelectiveDegradationPipeline:
    """Synthesizes (x_src, A_spatial, x_tgt) pairs with exact seed synchronization."""

    # Physical camera pipeline order:
    # Optics -> Radiometric/ISP -> Spatial Sampling -> Sensor Noise -> Quantization/Compression
    PHYSICAL_ORDER = [
        "defocus_blur",
        "motion_blur",
        "exposure_issue",
        "color_degradation",
        "downsample_upsample",
        "gaussian_noise",
        "color_banding",
        "jpeg_compression",
    ]

    def __init__(self, config):
        self.config = config
        self.degradation_types = list(config.DEGRADATION_TYPES)
        self.type_to_idx = {name: i for i, name in enumerate(self.degradation_types)}
        self.num_types = len(self.degradation_types)
        self.min_severity = getattr(config, "MIN_SEVERITY", 0.20)
        self.max_compound = getattr(config, "MAX_COMPOUND_DEGRADATIONS", 3)
        self.compound_prob = getattr(config, "COMPOUND_PROB", 0.65)
        self.spatial_mask_prob = getattr(config, "SPATIAL_MASK_PROB", 0.40)
        self._lut_grid = np.linspace(0.0, 1.0, 1025, dtype=np.float32)

    def sample_params(self, height: int, width: int, rng: np.random.Generator = None) -> DegradationParams:
        """Sample all stochastic physical realizations upfront, independent of severity."""
        rng = rng or np.random.default_rng()
        min_angle, max_angle = self.config.MOTION_BLUR_ANGLE_RANGE
        return DegradationParams(
            height=height,
            width=width,
            noise_seed=int(rng.integers(0, 2**31 - 1)),
            motion_angle=float(rng.uniform(min_angle, max_angle)),
            color_gains_unit=rng.uniform(-1.0, 1.0, size=(3,)).astype(np.float32),
            hue_shift_unit=float(rng.uniform(-1.0, 1.0)),
            sat_loss_frac=float(rng.uniform(0.25, 1.0)),
            underexpose=bool(rng.random() < 0.5),
            brightness_frac=float(rng.uniform(0.20, 1.0)),
            contrast_frac=float(rng.uniform(0.20, 1.0)),
        )

    @staticmethod
    def _fast_directional_motion_blur(img_float: np.ndarray, length_px: float, angle_deg: float) -> np.ndarray:
        """Continuous subpixel 1D shutter-trajectory blur via warpAffine (~25x faster than dense 2D filter2D)."""
        if length_px <= 0.05:
            return img_float
        h, w = img_float.shape[:2]
        rad = np.deg2rad(angle_deg)
        dx = float(np.cos(rad)) * 0.5 * length_px
        dy = -float(np.sin(rad)) * 0.5 * length_px

        # Adaptive tap count: smooth continuous trajectory without zero-padded 2D matrix waste
        num_taps = max(5, min(9, int(np.ceil(length_px * 0.35)) | 1))
        offsets = np.linspace(-1.0, 1.0, num_taps, dtype=np.float32)
        acc = np.zeros_like(img_float, dtype=np.float32)
        mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        for t in offsets:
            mat[0, 2] = t * dx
            mat[1, 2] = t * dy
            acc += cv2.warpAffine(
                img_float,
                mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        acc *= np.float32(1.0 / float(num_taps))
        return acc

    def _apply_single(self, img_float: np.ndarray, deg_name: str, severity: float, params: DegradationParams) -> np.ndarray:
        """Apply one degradation smoothly in strict float32 [0, 255] space (zero float64 upcasting)."""
        if severity <= 1e-4:
            return img_float

        severity = float(np.clip(severity, 0.0, 1.0))

        if deg_name == "defocus_blur":
            _, max_sigma = getattr(self.config, "DEFOCUS_SIGMA_RANGE", (0.8, 5.5))
            sigma = severity * max_sigma
            if sigma < 0.15:
                return img_float
            return cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma, sigmaY=sigma)

        if deg_name == "motion_blur":
            _, max_k = self.config.MOTION_BLUR_KERNEL_RANGE
            length_px = severity * (float(max_k) - 1.0)
            return self._fast_directional_motion_blur(img_float, length_px, params.motion_angle)

        if deg_name == "exposure_issue":
            min_g, max_g = self.config.GAMMA_RANGE
            if params.underexpose:
                gamma = 1.0 + severity * (max_g - 1.0)
                b_delta = -params.brightness_frac * self.config.BRIGHTNESS_MAX * severity
            else:
                gamma = 1.0 - severity * (1.0 - min_g)
                b_delta = params.brightness_frac * self.config.BRIGHTNESS_MAX * severity

            c_scale = 1.0 - params.contrast_frac * self.config.CONTRAST_MAX * severity
            x256 = np.linspace(0.0, 1.0, 256, dtype=np.float32)
            shifted = np.clip((x256 - 0.5) * c_scale + 0.5 + b_delta, 1e-4, 1.0)
            lut_u8 = np.clip(np.round((shifted ** gamma) * 255.0), 0.0, 255.0).astype(np.uint8)
            u8 = np.clip(np.round(img_float), 0.0, 255.0).astype(np.uint8)
            curved = cv2.LUT(u8, lut_u8).astype(np.float32)
            if severity >= 0.08:
                return curved
            blend = np.float32(severity / 0.08)
            return img_float * (np.float32(1.0) - blend) + curved * blend

        if deg_name == "color_degradation":
            max_shift = self.config.COLOR_SHIFT_MAX * severity
            scales = np.array([1.0, 0.6, 1.0], dtype=np.float32)
            gains = 1.0 + params.color_gains_unit * scales * max_shift
            gains = gains / max(1e-4, float(np.mean(gains)))

            x256 = np.arange(256, dtype=np.float32)
            lut_rgb = np.empty((1, 256, 3), dtype=np.uint8)
            for ch in range(3):
                lut_rgb[0, :, ch] = np.clip(np.round(x256 * float(gains[ch])), 0.0, 255.0).astype(np.uint8)

            u8 = np.clip(np.round(img_float), 0.0, 255.0).astype(np.uint8)
            cast_u8 = cv2.LUT(u8, lut_rgb)
            hsv = cv2.cvtColor(cast_u8, cv2.COLOR_RGB2HSV)

            hue_delta = params.hue_shift_unit * self.config.HUE_SHIFT_MAX * severity
            sat_scale = 1.0 - params.sat_loss_frac * self.config.SATURATION_LOSS_MAX * severity
            lut_hsv = np.empty((1, 256, 3), dtype=np.uint8)
            lut_hsv[0, :, 0] = (np.round(x256 + hue_delta).astype(np.int32) % 180).astype(np.uint8)
            lut_hsv[0, :, 1] = np.clip(np.round(x256 * sat_scale), 0.0, 255.0).astype(np.uint8)
            lut_hsv[0, :, 2] = np.arange(256, dtype=np.uint8)
            hsv_mod = cv2.LUT(hsv, lut_hsv)
            rgb = cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2RGB).astype(np.float32)
            if severity >= 0.10:
                return rgb
            blend = np.float32(severity / 0.10)
            return img_float * (np.float32(1.0) - blend) + rgb * blend

        if deg_name == "downsample_upsample":
            h, w = img_float.shape[:2]
            min_scale, _ = self.config.DOWNSAMPLE_SCALE_RANGE
            scale = 1.0 - severity * (1.0 - min_scale)
            if scale >= 0.98:
                return img_float
            new_w = max(8, int(round(w * scale)))
            new_h = max(8, int(round(h * scale)))
            down = cv2.resize(img_float, (new_w, new_h), interpolation=cv2.INTER_AREA)
            up = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
            return np.clip(up, np.float32(0.0), np.float32(255.0))

        if deg_name == "gaussian_noise":
            _, max_std = self.config.GAUSSIAN_NOISE_STD_RANGE
            noise_std = np.float32(severity * max_std)
            return np.clip(img_float + params.get_noise_unit() * noise_std, np.float32(0.0), np.float32(255.0))

        if deg_name == "color_banding":
            min_levels, max_levels = self.config.BANDING_LEVELS_RANGE
            min_bits = np.log2(float(min_levels))
            max_bits = np.log2(float(max_levels))
            bits = max_bits - severity * (max_bits - min_bits)
            levels = max(4, int(round(2.0 ** bits)))

            h, w = img_float.shape[:2]
            u8 = np.clip(img_float, np.float32(0.0), np.float32(255.0)).astype(np.uint8)
            # Half-res edge-preserving bilateral pre-smoothing (~8x faster than full-res d=5 bilateral)
            sigma_color = 6.0 + 18.0 * severity
            half_u8 = cv2.resize(u8, (max(16, w // 2), max(16, h // 2)), interpolation=cv2.INTER_AREA)
            smooth_half = cv2.bilateralFilter(half_u8, d=5, sigmaColor=sigma_color, sigmaSpace=3)
            smoothed = cv2.resize(smooth_half, (w, h), interpolation=cv2.INTER_LINEAR)

            lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB)
            l_step = 255.0 / float(levels - 1)
            ab_step = 255.0 / float(max(6, levels * 2) - 1)
            x256 = np.arange(256, dtype=np.float32)
            lut_lab = np.empty((1, 256, 3), dtype=np.uint8)
            lut_lab[0, :, 0] = np.clip(np.round(x256 / l_step) * l_step, 0.0, 255.0).astype(np.uint8)
            ab_q = np.clip(np.round((x256 - 128.0) / ab_step) * ab_step + 128.0, 0.0, 255.0).astype(np.uint8)
            lut_lab[0, :, 1] = ab_q
            lut_lab[0, :, 2] = ab_q
            banded = cv2.cvtColor(cv2.LUT(lab, lut_lab), cv2.COLOR_LAB2RGB).astype(np.float32)
            if severity >= 0.12:
                return banded
            blend = np.float32(severity / 0.12)
            return img_float * (np.float32(1.0) - blend) + banded * blend

        if deg_name == "jpeg_compression":
            min_q, max_q = self.config.JPEG_QUALITY_RANGE
            quality = int(round(max_q - severity * (max_q - min_q)))
            quality = int(np.clip(quality, 5, 95))
            u8_bgr = cv2.cvtColor(np.clip(img_float, np.float32(0.0), np.float32(255.0)).astype(np.uint8), cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".jpg", u8_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                return img_float
            dec_rgb = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
            if severity >= 0.12:
                return dec_rgb
            blend = np.float32(severity / 0.12)
            return img_float * (np.float32(1.0) - blend) + dec_rgb * blend

        raise ValueError(f"Unknown degradation type: {deg_name}")

    def render_with_params(self, clean_rgb_u8: np.ndarray, severity_vec: np.ndarray, params: DegradationParams) -> np.ndarray:
        """Render a degraded uint8 RGB image from clean_rgb_u8 using frozen DegradationParams."""
        cur = clean_rgb_u8.astype(np.float32)
        for deg_name in self.PHYSICAL_ORDER:
            idx = self.type_to_idx[deg_name]
            sev = float(severity_vec[idx])
            if sev > 1e-4:
                cur = self._apply_single(cur, deg_name, sev, params)
        return np.clip(np.round(cur), np.float32(0.0), np.float32(255.0)).astype(np.uint8)

    def sample_source_severity(self, rng: np.random.Generator = None) -> np.ndarray:
        """Sample 1 to 3 active degradations in [min_severity, 1.0]."""
        rng = rng or np.random.default_rng()
        if self.max_compound > 1 and rng.random() < self.compound_prob:
            k = 2 if (self.max_compound == 2 or rng.random() < 0.70) else min(3, self.max_compound)
        else:
            k = 1

        chosen = list(rng.choice(self.num_types, size=min(k, self.num_types), replace=False))
        chosen_names = [self.degradation_types[i] for i in chosen]

        if "motion_blur" in chosen_names and "defocus_blur" in chosen_names and len(chosen) > 1:
            drop_name = "motion_blur" if rng.random() < 0.5 else "defocus_blur"
            chosen.remove(self.type_to_idx[drop_name])

        damping = 1.0 if len(chosen) == 1 else (0.88 if len(chosen) == 2 else 0.78)
        s_src = np.zeros(self.num_types, dtype=np.float32)
        for idx in chosen:
            raw = float(rng.uniform(self.min_severity, 1.0)) * damping
            s_src[idx] = float(np.clip(raw, self.min_severity, 1.0))
        return s_src

    def _enforce_physical_action_consistency(self, s_src: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Prevent physically contradictory targets when strong radiometric edits precede quantization."""
        action = action.copy()
        exp_idx = self.type_to_idx["exposure_issue"]
        col_idx = self.type_to_idx["color_degradation"]
        rad_edit = max(
            action[exp_idx] if s_src[exp_idx] > 0 else 0.0,
            action[col_idx] if s_src[col_idx] > 0 else 0.0,
        )
        if rad_edit > 0.35:
            for quant_name in ("jpeg_compression", "color_banding"):
                q_idx = self.type_to_idx[quant_name]
                if s_src[q_idx] > 0:
                    action[q_idx] = max(action[q_idx], 0.55 * rad_edit)
        return np.clip(action, np.float32(0.0), np.float32(1.0))

    def sample_action_vector(self, s_src: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        """Sample diverse controllable user action vectors a in [0, 1]^8."""
        rng = rng or np.random.default_rng()
        active_indices = [i for i in range(self.num_types) if s_src[i] > 1e-4]
        inactive_indices = [i for i in range(self.num_types) if s_src[i] <= 1e-4]

        p_sel = self.config.ACTION_DIST_SELECTIVE
        p_part = self.config.ACTION_DIST_PARTIAL
        p_full = self.config.ACTION_DIST_FULL
        p_fp = self.config.ACTION_DIST_FALSE_POS
        u = float(rng.random())

        action = np.zeros(self.num_types, dtype=np.float32)

        if u < p_sel and len(active_indices) >= 2:
            num_fix = int(rng.integers(1, len(active_indices)))
            fix_ids = rng.choice(active_indices, size=num_fix, replace=False)
            for idx in fix_ids:
                action[idx] = 1.0 if rng.random() < 0.7 else float(rng.uniform(0.5, 1.0))

        elif u < p_sel + p_part:
            for idx in active_indices:
                if rng.random() < 0.80:
                    action[idx] = float(rng.uniform(0.15, 0.90))

        elif u < p_sel + p_part + p_full:
            for idx in active_indices:
                action[idx] = 1.0

        elif u < p_sel + p_part + p_full + p_fp:
            for idx in active_indices:
                if rng.random() < 0.65:
                    action[idx] = float(rng.choice([0.0, 0.5, 1.0]))
            if inactive_indices:
                n_fp = int(rng.integers(1, min(3, len(inactive_indices)) + 1))
                for idx in rng.choice(inactive_indices, size=n_fp, replace=False):
                    action[idx] = float(rng.uniform(0.4, 1.0))

        else:
            action[:] = 0.0

        return self._enforce_physical_action_consistency(s_src, action)

    def sample_spatial_mask(self, clean_rgb_u8: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        """Generate a smooth feathered spatial region mask M in [0, 1]^(H, W) (~15x faster via coarse grid)."""
        rng = rng or np.random.default_rng()
        h, w = clean_rgb_u8.shape[:2]
        gh, gw = max(16, h // 4), max(16, w // 4)
        mode = float(rng.random())

        if mode < 0.45:
            # Coarse 64x64 LAB region clustering + full-res smooth feathering (~0.5 ms vs 9.5 ms!)
            small_rgb = cv2.resize(clean_rgb_u8, (gw, gh), interpolation=cv2.INTER_AREA)
            lab = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            cy = int(rng.integers(gh // 6, 5 * gh // 6))
            cx = int(rng.integers(gw // 6, 5 * gw // 6))
            diff = lab - lab[cy, cx].reshape(1, 1, 3)
            dist_sq = np.sum(diff * diff, axis=-1)
            thresh = float(np.percentile(dist_sq, rng.uniform(25.0, 55.0)))
            raw_small = (dist_sq <= thresh).astype(np.float32)
            raw_mask = cv2.resize(raw_small, (w, h), interpolation=cv2.INTER_LINEAR)
            sigma = float(rng.uniform(4.0, 10.0))
            mask = cv2.GaussianBlur(raw_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
        elif mode < 0.80:
            mask_small = np.zeros((gh, gw), dtype=np.float32)
            num_blobs = int(rng.integers(1, 3))
            for _ in range(num_blobs):
                cx = int(rng.integers(gw // 5, 4 * gw // 5))
                cy = int(rng.integers(gh // 5, 4 * gh // 5))
                rx = int(rng.integers(gw // 6, gw // 2))
                ry = int(rng.integers(gh // 6, gh // 2))
                angle = float(rng.uniform(0.0, 180.0))
                cv2.ellipse(mask_small, (cx, cy), (rx, ry), angle, 0, 360, 1.0, -1)
            mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
            sigma = float(rng.uniform(6.0, 18.0))
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
        else:
            yy, xx = np.meshgrid(
                np.linspace(-1.0, 1.0, gh, dtype=np.float32),
                np.linspace(-1.0, 1.0, gw, dtype=np.float32),
                indexing="ij",
            )
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            proj = np.float32(np.cos(theta)) * xx + np.float32(np.sin(theta)) * yy
            offset = np.float32(rng.uniform(-0.35, 0.35))
            sharpness = np.float32(rng.uniform(4.0, 10.0))
            mask_small = np.float32(1.0) / (np.float32(1.0) + np.exp(-sharpness * (proj - offset)))
            mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)

        m_max = float(mask.max())
        if m_max > 1e-5:
            mask = mask * np.float32(1.0 / m_max)
        return np.clip(mask, np.float32(0.0), np.float32(1.0))

    def synthesize_components(
        self,
        clean_rgb_u8: np.ndarray,
        rng: np.random.Generator = None,
        align_only: bool = False,
    ):
        """Fast synthesis returning compact components (x_src, x_tgt, a_fg, a_bg, mask_2d, s_src)."""
        rng = rng or np.random.default_rng()
        h, w = clean_rgb_u8.shape[:2]
        params = self.sample_params(h, w, rng=rng)
        s_src = self.sample_source_severity(rng=rng)
        x_src = self.render_with_params(clean_rgb_u8, s_src, params)

        if align_only:
            zeros_a = np.zeros(self.num_types, dtype=np.float32)
            ones_mask = np.ones((h, w), dtype=np.float32)
            return x_src, x_src, zeros_a, zeros_a, ones_mask, s_src

        a_fg = self.sample_action_vector(s_src, rng=rng)
        s_tgt_fg = np.clip((1.0 - a_fg) * s_src, 0.0, 1.0)
        if np.allclose(s_tgt_fg, s_src, atol=1e-4):
            x_tgt_fg = x_src.copy()
        elif np.allclose(s_tgt_fg, 0.0, atol=1e-4):
            # Shortcut: full restoration target is literally clean_rgb_u8 (saves an entire render pass!)
            x_tgt_fg = clean_rgb_u8.copy()
        else:
            x_tgt_fg = self.render_with_params(clean_rgb_u8, s_tgt_fg, params)

        if rng.random() < self.spatial_mask_prob:
            if rng.random() < 0.65:
                a_bg = np.zeros(self.num_types, dtype=np.float32)
            else:
                a_bg = self.sample_action_vector(s_src, rng=rng)

            s_tgt_bg = np.clip((1.0 - a_bg) * s_src, 0.0, 1.0)
            if np.allclose(s_tgt_bg, s_src, atol=1e-4):
                x_tgt_bg = x_src
            elif np.allclose(s_tgt_bg, 0.0, atol=1e-4):
                x_tgt_bg = clean_rgb_u8
            elif np.allclose(s_tgt_bg, s_tgt_fg, atol=1e-4):
                x_tgt_bg = x_tgt_fg
            else:
                x_tgt_bg = self.render_with_params(clean_rgb_u8, s_tgt_bg, params)

            mask = self.sample_spatial_mask(clean_rgb_u8, rng=rng)
            mask_3d = mask[:, :, None]
            x_tgt = np.clip(
                np.round(mask_3d * x_tgt_fg.astype(np.float32) + (np.float32(1.0) - mask_3d) * x_tgt_bg.astype(np.float32)),
                np.float32(0.0),
                np.float32(255.0),
            ).astype(np.uint8)
        else:
            a_bg = a_fg
            mask = np.ones((h, w), dtype=np.float32)
            x_tgt = x_tgt_fg

        return x_src, x_tgt, a_fg, a_bg, mask, s_src

    def synthesize_pair(self, clean_rgb_u8: np.ndarray, rng: np.random.Generator = None):
        """Synthesize seed-synchronized (x_src, action_map [8, H, W], x_tgt, s_src)."""
        x_src, x_tgt, a_fg, a_bg, mask, s_src = self.synthesize_components(clean_rgb_u8, rng=rng, align_only=False)
        h, w = clean_rgb_u8.shape[:2]
        if a_bg is a_fg or np.array_equal(a_fg, a_bg):
            action_map = np.broadcast_to(a_fg[:, None, None], (self.num_types, h, w)).copy().astype(np.float32)
        else:
            action_map = (
                mask[None, :, :] * a_fg[:, None, None] + (np.float32(1.0) - mask[None, :, :]) * a_bg[:, None, None]
            ).astype(np.float32)
        return x_src, action_map, x_tgt, s_src


class ControllableRestorationDataset(_TorchDataset):
    """CPU-optimized PyTorch Dataset with staggered 2K-decode ring cache & compact IPC masks."""

    def __init__(self, image_paths, config, is_train=True, align_only: bool = False, compact_ipc: bool = False):
        self.image_paths = list(image_paths)
        self.config = config
        self.is_train = is_train
        self.align_only = align_only
        self.compact_ipc = compact_ipc
        self.patch_size = config.PATCH_SIZE
        self.pipeline = SelectiveDegradationPipeline(config)

        # Per-worker staggered image cache: cuts 2K PNG FUSE disk reads & libpng CPU decodes by ~6x!
        self._cache_capacity = int(getattr(config, "DECODE_CACHE_SIZE", 16)) if is_train else 0
        self._max_crops_per_decode = int(getattr(config, "CROPS_PER_DECODE", 6)) if is_train else 1
        self._decoded_ring = []  # List of [rgb_full_u8, remaining_uses]

    def __len__(self):
        return len(self.image_paths)

    def _crop_and_augment(self, img_rgb: np.ndarray) -> np.ndarray:
        h, w = img_rgb.shape[:2]
        ps = self.patch_size

        if h >= ps and w >= ps:
            if self.is_train:
                y = random.randint(0, h - ps)
                x = random.randint(0, w - ps)
            else:
                y = (h - ps) // 2
                x = (w - ps) // 2
            crop = img_rgb[y : y + ps, x : x + ps]
        else:
            crop = cv2.resize(img_rgb, (ps, ps), interpolation=cv2.INTER_CUBIC)

        if self.is_train:
            if random.random() < 0.5:
                crop = np.fliplr(crop)
            if random.random() < 0.5:
                crop = np.flipud(crop)
            k = random.randint(0, 3)
            if k > 0:
                crop = np.rot90(crop, k)
            return np.ascontiguousarray(crop)
        return crop.copy()

    def _decode_rgb(self, image_path: str) -> np.ndarray:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_crop(self, image_path: str) -> np.ndarray:
        return self._crop_and_augment(self._decode_rgb(image_path))

    def _load_crop_with_retry(self, idx: int, max_attempts: int = 5) -> np.ndarray:
        # Staggered reuse from per-worker decoded image ring buffer (only when cache is full so batches stay diverse)
        if self.is_train and self._cache_capacity > 0 and len(self._decoded_ring) >= self._cache_capacity:
            slot = random.randrange(len(self._decoded_ring))
            rgb_cached, rem = self._decoded_ring[slot]
            if rem > 1:
                self._decoded_ring[slot][1] = rem - 1
            else:
                self._decoded_ring.pop(slot)
            return self._crop_and_augment(rgb_cached)

        n = len(self.image_paths)
        last_err = None
        for attempt in range(max_attempts):
            cand_idx = (idx + attempt) % n
            try:
                rgb_full = self._decode_rgb(self.image_paths[cand_idx])
                if (
                    self.is_train
                    and self._cache_capacity > 0
                    and self._max_crops_per_decode > 1
                    and rgb_full.shape[0] >= self.patch_size * 2
                ):
                    self._decoded_ring.append([rgb_full, self._max_crops_per_decode - 1])
                return self._crop_and_augment(rgb_full)
            except Exception as err:
                last_err = err
        raise RuntimeError(f"Failed to read valid image after {max_attempts} attempts: {last_err}")

    def __getitem__(self, idx):
        clean_u8 = self._load_crop_with_retry(idx)
        rng = np.random.default_rng() if self.is_train else np.random.default_rng(self.config.SEED + idx)

        x_src_u8, x_tgt_u8, a_fg, a_bg, mask_2d, s_src = self.pipeline.synthesize_components(
            clean_u8, rng=rng, align_only=self.align_only
        )

        inv_255 = np.float32(1.0 / 255.0)
        clean_t = torch.from_numpy(clean_u8).permute(2, 0, 1).float().mul_(inv_255)
        src_t = torch.from_numpy(x_src_u8).permute(2, 0, 1).float().mul_(inv_255)

        if self.align_only:
            return {"clean": clean_t, "source": src_t}

        tgt_t = torch.from_numpy(x_tgt_u8).permute(2, 0, 1).float().mul_(inv_255)
        s_src_t = torch.from_numpy(s_src)

        if self.compact_ipc:
            # 16x smaller multiprocessing shared-memory transfer; expanded in 0.03 ms on GPU!
            return {
                "clean": clean_t,
                "source": src_t,
                "target": tgt_t,
                "a_fg": torch.from_numpy(a_fg),
                "a_bg": torch.from_numpy(a_bg),
                "spatial_mask": torch.from_numpy(mask_2d).unsqueeze(0).half(),
                "source_severity": s_src_t,
            }

        h, w = clean_u8.shape[:2]
        if a_bg is a_fg or np.array_equal(a_fg, a_bg):
            action_map = np.broadcast_to(a_fg[:, None, None], (self.pipeline.num_types, h, w)).copy().astype(np.float32)
        else:
            action_map = (
                mask_2d[None, :, :] * a_fg[:, None, None] + (np.float32(1.0) - mask_2d[None, :, :]) * a_bg[:, None, None]
            ).astype(np.float32)
        action_map_t = torch.from_numpy(action_map)
        return {
            "clean": clean_t,
            "source": src_t,
            "target": tgt_t,
            "action_map": action_map_t,
            "action_global": action_map_t.mean(dim=(1, 2)),
            "source_severity": s_src_t,
        }


def dataloader_worker_init_fn(worker_id: int):
    """Disable OpenCV internal OpenMP thread contention across PyTorch DataLoader workers."""
    cv2.setNumThreads(0)
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    base_seed = torch.initial_seed() % (2**31)
    random.seed(base_seed + worker_id)
    np.random.seed(base_seed + worker_id)


def unpack_batch_action_map(batch: dict, device) -> "torch.Tensor":
    """Reconstruct [B, K, H, W] action_map directly on GPU from compact IPC components."""
    if "action_map" in batch:
        return batch["action_map"].to(device, non_blocking=True)
    a_fg = batch["a_fg"].to(device, non_blocking=True).float()[:, :, None, None]
    a_bg = batch["a_bg"].to(device, non_blocking=True).float()[:, :, None, None]
    mask = batch["spatial_mask"].to(device, non_blocking=True).float()
    return mask * a_fg + (1.0 - mask) * a_bg


def _collect_images(folder: Path):
    """Recursively collect image files (following symlinks for Kaggle /kaggle/input mounts)."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    folder_str = str(folder)
    if not os.path.exists(folder_str):
        return []
    found = []
    for root_dir, _dirs, files in os.walk(folder_str, followlinks=True):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in exts:
                found.append(Path(os.path.join(root_dir, fname)))
    return sorted(set(found))


def create_controllable_dataset(config, is_train=True, dataset_path=None, align_only: bool = False, compact_ipc: bool = False):
    """Discover DIV2K, DF2K-OST, or arbitrary recursive image folders (matching Kaggle workflow)."""
    split_name = "train" if is_train else "val"
    explicit_dir = dataset_path or (config.TRAIN_DATASET_PATH if is_train else config.VAL_DATASET_PATH)

    if explicit_dir:
        target_root = Path(explicit_dir)
        div2k_sub = target_root / ("DIV2K_train_HR" if is_train else "DIV2K_valid_HR")
        all_paths = _collect_images(target_root)
        if not all_paths:
            raise FileNotFoundError(
                f"No images found in {target_root}. Verify --{'train' if is_train else 'val'}-root path."
            )
        same_shared_root = (
            config.TRAIN_DATASET_PATH
            and config.VAL_DATASET_PATH
            and os.path.abspath(str(config.TRAIN_DATASET_PATH)) == os.path.abspath(str(config.VAL_DATASET_PATH))
        )
        if same_shared_root and div2k_sub.exists():
            paths = _collect_images(div2k_sub)
        elif same_shared_root:
            rng = random.Random(config.SEED)
            shuffled = list(all_paths)
            rng.shuffle(shuffled)
            split_idx = max(1, int(len(shuffled) * 0.9))
            paths = shuffled[:split_idx] if is_train else shuffled[split_idx:]
        else:
            paths = all_paths
        source_desc = str(target_root)
    else:
        root = Path(config.DATASET_PATH)
        sub = root / ("DIV2K_train_HR" if is_train else "DIV2K_valid_HR")
        if sub.exists():
            paths = _collect_images(sub)
            source_desc = str(sub)
        else:
            print(f"No DIV2K folder found at {sub}, falling back to recursive image search in {root}.")
            all_paths = _collect_images(root)
            if not all_paths:
                raise FileNotFoundError(
                    f"No images found in {root}. Pass --train-root / --val-root or --data-root."
                )
            rng = random.Random(config.SEED)
            shuffled = list(all_paths)
            rng.shuffle(shuffled)
            split_idx = max(1, int(len(shuffled) * 0.9))
            paths = shuffled[:split_idx] if is_train else shuffled[split_idx:]
            source_desc = str(root)

    limit = config.TRAIN_IMAGES_COUNT if is_train else config.VAL_IMAGES_COUNT
    if limit and 0 < limit < len(paths):
        rng = random.Random(config.SEED + (0 if is_train else 1))
        shuffled = list(paths)
        rng.shuffle(shuffled)
        paths = sorted(shuffled[:limit])

    print(f"[{split_name}] Using {len(paths)} clean reference images from {source_desc}")
    return ControllableRestorationDataset(
        paths, config, is_train=is_train, align_only=align_only, compact_ipc=compact_ipc
    )
