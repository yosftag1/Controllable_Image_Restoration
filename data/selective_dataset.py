"""Seed-synchronized selective, localized & directional degradation + Real-Pair Hybrid Dataset (V5)."""

from dataclasses import dataclass, field
from pathlib import Path
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

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
    """Frozen random realizations for a single image patch so source and partial targets align."""

    height: int
    width: int
    noise_seed: int                  # Deterministic seed for lazy sensor noise realization
    motion_angle: float              # Primary trajectory angle in [0, 180)
    motion_curve_bend: float = 0.0   # Quadratic Bezier camera-shake curvature in [-max_bend, +max_bend]
    motion_dwell_end: float = 0.0    # Shutter turnaround endpoint dwell in [0, 0.35]
    motion_shake_jitter: float = 0.0 # S-curve / handheld tremor amplitude in [-0.35, +0.35]
    motion_pattern: int = 0          # 0 = Bezier arc, 1 = S-curve tremor, 2 = micro-shake jitter
    use_bokeh_disc: bool = True      # True = optical circle-of-confusion disc PSF, False = Gaussian softness
    focus_mode: int = 0              # 0 = optical defocus/bokeh, 1 = low-res bicubic, 2 = hybrid lens+resample
    use_chromatic_aberration: bool = False  # Subtle lateral R/B lens fringing in optical focus mode
    chroma_shift_px: float = 0.0     # Lateral R/B chromatic shift amplitude in pixels
    artifact_mode: int = 0           # 0 = JPEG, 1 = 8-bit banding, 2 = JPEG+banding, 3 = 4:2:0 chroma+ringing+JPEG
    jpeg_quality: int = 20           # Frozen base JPEG quality in [10, 32] for monotonic severity blending
    banding_levels: int = 12         # Frozen base quantization levels in [7, 16]
    heteroscedastic_noise: bool = False  # Signal-dependent CMOS shot+read noise modulation
    flare_style: int = 0             # 0=Wave-Optics FFT, 1=Anamorphic, 2=Off-frame Wash, 3=Ghost Chain, 4=Rainbow Arc
    flare_cx: float = 0.80           # Flare light center X in normalized coords (can be off-frame)
    flare_cy: float = 0.16           # Flare light center Y in normalized coords
    flare_seed: int = 7              # Deterministic seed for FFT dust/scratches & ghosts
    flare_tint: Tuple[float, float, float] = (1.0, 0.88, 0.68)
    refl_sigma: float = 2.6          # Glass reflection optical defocus sigma
    refl_dx: float = 5.0             # Double-pane horizontal ghost offset (px)
    refl_dy: float = 3.0             # Double-pane vertical ghost offset (px)
    refl_angle: float = 0.6          # Planar Fresnel gradient angle
    ca_max_scale: float = 0.016      # Radial R/B magnification delta at severity=1.0
    haze_seed: int = 11              # Seed for smooth spatial atmospheric depth map
    haze_airlight: Tuple[float, float, float] = (215.0, 224.0, 236.0)
    exposure_sign: float = 1.0       # -1.0 (dark/underexposed) or +1.0 (bright/washed-out)
    warmth_sign: float = 1.0         # -1.0 (cold blue cast) or +1.0 (warm amber/yellow cast)
    _noise_cache: Optional[np.ndarray] = field(default=None, repr=False)
    _flare_cache: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None, repr=False)
    _refl_cache: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None, repr=False)
    _haze_depth_cache: Optional[np.ndarray] = field(default=None, repr=False)

    def get_noise_unit(self) -> np.ndarray:
        """Lazily materialize realistic sensor noise (80% RGB independent + 20% correlated luminance grain)."""
        if self._noise_cache is None:
            nrng = np.random.default_rng(self.noise_seed)
            rgb_n = nrng.standard_normal((self.height, self.width, 3), dtype=np.float32)
            luma_n = nrng.standard_normal((self.height, self.width, 1), dtype=np.float32)
            self._noise_cache = np.float32(0.82) * rgb_n + np.float32(0.35) * luma_n
        return self._noise_cache


class SelectiveDegradationPipeline:
    """Synthesizes (x_src, A_spatial, x_tgt) pairs from clean GT or Compositional Real Camera Blur Pairs."""

    PHYSICAL_ORDER = [
        "dehaze",
        "dereflection",
        "sharpen_focus",
        "dechromatic",
        "deblur_motion",
        "deflare",
        "exposure",
        "warmth",
        "deartifact",
        "denoise",
    ]

    def __init__(self, config):
        self.config = config
        self.degradation_types = list(config.DEGRADATION_TYPES)
        self.signed_types = set(getattr(config, "SIGNED_ACTION_TYPES", ["exposure", "warmth"]))
        self.type_to_idx = {name: i for i, name in enumerate(self.degradation_types)}
        self.signed_indices = [i for i, n in enumerate(self.degradation_types) if n in self.signed_types]
        self.unsigned_indices = [i for i, n in enumerate(self.degradation_types) if n not in self.signed_types]
        self.num_types = len(self.degradation_types)

        self.min_severity = float(getattr(config, "MIN_SEVERITY", 0.25))
        self.max_compound = int(getattr(config, "MAX_COMPOUND_DEGRADATIONS", 3))
        self.compound_prob = float(getattr(config, "COMPOUND_PROB", 0.70))
        self.spatial_mask_prob = float(getattr(config, "SPATIAL_MASK_PROB", 0.25))

        self._curriculum_boost_val: float = 0.0
        self._curriculum_shared = (
            torch.zeros(1, dtype=torch.float32).share_memory_() if torch is not None else None
        )

        self.exp_kappa = float(getattr(config, "EXPOSURE_STRENGTH_KAPPA", 2.15))
        self.warm_r_kappa = float(getattr(config, "WARMTH_R_KAPPA", 1.15))
        self.warm_g_kappa = float(getattr(config, "WARMTH_G_KAPPA", 0.12))
        self.warm_b_kappa = float(getattr(config, "WARMTH_B_KAPPA", 1.30))

    @property
    def curriculum_single_boost(self) -> float:
        if self._curriculum_shared is not None:
            return float(self._curriculum_shared[0].item())
        return self._curriculum_boost_val

    @curriculum_single_boost.setter
    def curriculum_single_boost(self, val: float) -> None:
        clamped = float(np.clip(val, 0.0, 1.0))
        self._curriculum_boost_val = clamped
        if self._curriculum_shared is not None:
            self._curriculum_shared[0] = clamped

    def is_signed(self, deg_name_or_idx) -> bool:
        if isinstance(deg_name_or_idx, int):
            return deg_name_or_idx in self.signed_indices
        return str(deg_name_or_idx) in self.signed_types

    def sample_params(self, height: int, width: int, rng: np.random.Generator = None) -> DegradationParams:
        """Sample all stochastic physical realizations upfront, independent of severity."""
        rng = rng or np.random.default_rng()
        min_angle, max_angle = self.config.MOTION_BLUR_ANGLE_RANGE
        max_bend = float(getattr(self.config, "MOTION_CURVE_MAX_BEND", 0.38))
        min_q, max_q = self.config.JPEG_QUALITY_RANGE
        min_lvl, max_lvl = self.config.BANDING_LEVELS_RANGE
        f_style = int(rng.choice([0, 1, 2, 3, 4], p=[0.24, 0.18, 0.22, 0.22, 0.14]))
        if f_style == 2:
            # Off-frame light source just outside border
            side = int(rng.integers(0, 4))
            if side == 0:
                fcx, fcy = float(rng.uniform(-0.22, -0.04)), float(rng.uniform(0.05, 0.90))
            elif side == 1:
                fcx, fcy = float(rng.uniform(1.04, 1.22)), float(rng.uniform(0.05, 0.90))
            elif side == 2:
                fcx, fcy = float(rng.uniform(0.10, 0.90)), float(rng.uniform(-0.22, -0.04))
            else:
                fcx, fcy = float(rng.uniform(0.10, 0.90)), float(rng.uniform(1.04, 1.22))
        else:
            fcx = float(rng.uniform(0.08, 0.92))
            fcy = float(rng.uniform(0.08, 0.42) if rng.random() < 0.7 else rng.uniform(0.55, 0.92))
        tint_presets = [
            (1.0, 0.88, 0.66),
            (1.0, 0.94, 0.82),
            (0.92, 0.96, 1.00),
            (1.0, 0.82, 0.56),
        ]
        f_tint = tint_presets[int(rng.integers(0, len(tint_presets)))]
        air_base = float(rng.uniform(195.0, 238.0))
        air_col = (
            float(np.clip(air_base + rng.uniform(-10.0, 8.0), 175.0, 248.0)),
            float(np.clip(air_base + rng.uniform(-4.0, 8.0), 180.0, 248.0)),
            float(np.clip(air_base + rng.uniform(2.0, 18.0), 185.0, 250.0)),
        )
        return DegradationParams(
            height=height,
            width=width,
            noise_seed=int(rng.integers(0, 2**31 - 1)),
            motion_angle=float(rng.uniform(min_angle, max_angle)),
            motion_curve_bend=float(rng.uniform(-max_bend, max_bend)),
            motion_dwell_end=float(rng.uniform(0.0, 0.35)),
            motion_shake_jitter=float(rng.uniform(-0.35, 0.35)),
            motion_pattern=int(rng.choice([0, 1, 2], p=[0.45, 0.35, 0.20])),
            use_bokeh_disc=bool(rng.random() < 0.70),
            focus_mode=int(rng.choice([0, 1, 2], p=[0.46, 0.32, 0.22])),
            use_chromatic_aberration=bool(rng.random() < 0.20),
            chroma_shift_px=float(rng.uniform(0.8, 2.0)),
            artifact_mode=int(rng.choice([0, 1, 2, 3], p=[0.38, 0.20, 0.20, 0.22])),
            jpeg_quality=int(rng.integers(min_q, max(min_q + 1, (min_q + max_q) // 2))),
            banding_levels=int(rng.integers(min_lvl, max(min_lvl + 1, (min_lvl + max_lvl) // 2))),
            heteroscedastic_noise=bool(rng.random() < 0.40),
            flare_style=f_style,
            flare_cx=fcx,
            flare_cy=fcy,
            flare_seed=int(rng.integers(0, 2**31 - 1)),
            flare_tint=f_tint,
            refl_sigma=float(rng.uniform(1.8, 4.0)),
            refl_dx=float(rng.uniform(3.0, 7.0)) * (-1.0 if rng.random() < 0.5 else 1.0),
            refl_dy=float(rng.uniform(2.0, 5.0)) * (-1.0 if rng.random() < 0.5 else 1.0),
            refl_angle=float(rng.uniform(0.0, 2.0 * np.pi)),
            ca_max_scale=float(rng.uniform(0.010, 0.022)),
            haze_seed=int(rng.integers(0, 2**31 - 1)),
            haze_airlight=air_col,
            exposure_sign=-1.0 if rng.random() < 0.5 else 1.0,
            warmth_sign=-1.0 if rng.random() < 0.5 else 1.0,
        )

    def compute_target_state(self, s_src: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Map source state `s_src` + user control `action` to exact target state `s_tgt`."""
        s_tgt = np.array(s_src, dtype=np.float32, copy=True)
        act = np.asarray(action, dtype=np.float32)
        for k in self.unsigned_indices:
            a_k = float(np.clip(act[k], 0.0, 1.0))
            s_tgt[k] = float(np.clip((1.0 - a_k) * float(s_src[k]), 0.0, 1.0))
        for k in self.signed_indices:
            a_k = float(np.clip(act[k], -1.0, 1.0))
            s_tgt[k] = float(np.clip(float(s_src[k]) + a_k, -1.25, 1.25))
        return s_tgt

    def compute_full_restore_action(self, s_src: np.ndarray) -> np.ndarray:
        """Return the exact user action vector that restores `s_src` completely back to clean `0.0`."""
        act = np.zeros(self.num_types, dtype=np.float32)
        for k in self.unsigned_indices:
            if float(s_src[k]) > 1e-4:
                act[k] = 1.0
        for k in self.signed_indices:
            if abs(float(s_src[k])) > 1e-4:
                act[k] = float(np.clip(-float(s_src[k]), -1.0, 1.0))
        return act

    def compute_partial_restore_action(self, s_src: np.ndarray, fraction: float = 0.5) -> np.ndarray:
        """Return the user action vector that removes `fraction` of every active degradation."""
        frac = float(np.clip(fraction, 0.0, 1.0))
        act = np.zeros(self.num_types, dtype=np.float32)
        for k in self.unsigned_indices:
            if float(s_src[k]) > 1e-4:
                act[k] = frac
        for k in self.signed_indices:
            if abs(float(s_src[k])) > 1e-4:
                act[k] = float(np.clip(-frac * float(s_src[k]), -1.0, 1.0))
        return act

    @staticmethod
    def _rational_tone_curve(x01: np.ndarray, gain: np.ndarray) -> np.ndarray:
        """Exact Lie-group smooth bijection T(x; g) = (g * x) / (1 + (g - 1) * x) on [0, 1]."""
        denom = np.float32(1.0) + (gain - np.float32(1.0)) * x01
        return np.clip((gain * x01) / np.maximum(denom, np.float32(1e-6)), np.float32(0.0), np.float32(1.0))

    @staticmethod
    def _continuous_curved_motion_blur(
        img_float: np.ndarray,
        length_px: float,
        angle_deg: float,
        curve_bend: float = 0.0,
        dwell_end: float = 0.0,
        shake_jitter: float = 0.0,
        pattern: int = 0,
    ) -> np.ndarray:
        """Single-pass 2D Bezier / S-curve / handheld tremor PSF kernel convolution."""
        if length_px <= 0.05:
            return img_float
        rad = np.deg2rad(angle_deg)
        tx = float(np.cos(rad)) * 0.5 * length_px
        ty = -float(np.sin(rad)) * 0.5 * length_px
        nx = -ty
        ny = tx

        num_taps = max(7, min(33, int(np.ceil(length_px * 1.4)) | 1))
        t_vals = np.linspace(-1.0, 1.0, num_taps, dtype=np.float32)
        weights = np.float32(1.0) + np.float32(dwell_end) * (t_vals * t_vals)
        weights /= np.sum(weights)

        quad_zero_mean = (t_vals * t_vals) - np.float32(1.0 / 3.0)
        dx_arr = t_vals * np.float32(tx) + np.float32(curve_bend) * quad_zero_mean * np.float32(nx)
        dy_arr = t_vals * np.float32(ty) + np.float32(curve_bend) * quad_zero_mean * np.float32(ny)

        if pattern == 1 and abs(shake_jitter) > 1e-4:
            # S-curve handheld trajectory wobble
            s_wave = np.sin(np.float32(np.pi) * t_vals) * np.float32(shake_jitter)
            dx_arr = dx_arr + s_wave * np.float32(nx)
            dy_arr = dy_arr + s_wave * np.float32(ny)
        elif pattern == 2 and abs(shake_jitter) > 1e-4:
            # Multi-frequency handheld micro-shake tremor
            wobble_n = np.sin(np.float32(2.5 * np.pi) * t_vals) * np.float32(0.75 * shake_jitter)
            wobble_t = np.cos(np.float32(2.0 * np.pi) * t_vals) * np.float32(0.18 * abs(shake_jitter))
            dx_arr = dx_arr + wobble_n * np.float32(nx) + wobble_t * np.float32(tx)
            dy_arr = dy_arr + wobble_n * np.float32(ny) + wobble_t * np.float32(ty)

        max_reach = int(np.ceil(max(float(np.max(np.abs(dx_arr))), float(np.max(np.abs(dy_arr)))) + 1.5))
        half_k = max(1, max_reach)
        k_size = 2 * half_k + 1
        psf = np.zeros((k_size, k_size), dtype=np.float32)

        # Subpixel bilinear splatting of shift `(dx, dy)` onto compact 2D correlation kernel
        fx = float(half_k) - dx_arr
        fy = float(half_k) - dy_arr
        x0 = np.floor(fx).astype(np.int32)
        y0 = np.floor(fy).astype(np.int32)
        wx1 = fx - x0.astype(np.float32)
        wy1 = fy - y0.astype(np.float32)
        wx0 = np.float32(1.0) - wx1
        wy0 = np.float32(1.0) - wy1

        np.add.at(psf, (y0, x0), weights * wy0 * wx0)
        np.add.at(psf, (y0, x0 + 1), weights * wy0 * wx1)
        np.add.at(psf, (y0 + 1, x0), weights * wy1 * wx0)
        np.add.at(psf, (y0 + 1, x0 + 1), weights * wy1 * wx1)
        psf /= np.maximum(np.sum(psf), np.float32(1e-6))

        return cv2.filter2D(img_float, cv2.CV_32F, psf, borderType=cv2.BORDER_REFLECT_101)

    @classmethod
    def render_custom_motion_blur(
        cls,
        img_rgb_u8: np.ndarray,
        length_px: float,
        angle_deg: float,
        curve_bend: float = 0.0,
        shake_jitter: float = 0.0,
        dwell_end: float = 0.0,
        pattern: int = 0,
    ) -> np.ndarray:
        """Synthesize custom linear-light motion blur with explicit length, angle, curvature & shake."""
        if length_px <= 0.05:
            return img_rgb_u8.copy()
        inv255 = np.float32(1.0 / 255.0)
        lin = np.square(np.clip(img_rgb_u8.astype(np.float32) * inv255, np.float32(0.0), np.float32(1.0))) * np.float32(255.0)
        blurred = cls._continuous_curved_motion_blur(
            lin,
            length_px=float(length_px),
            angle_deg=float(angle_deg),
            curve_bend=float(curve_bend),
            dwell_end=float(dwell_end),
            shake_jitter=float(shake_jitter),
            pattern=int(pattern),
        )
        out = np.sqrt(np.clip(blurred * inv255, np.float32(0.0), np.float32(1.0))) * np.float32(255.0)
        return np.clip(np.round(out), np.float32(0.0), np.float32(255.0)).astype(np.uint8)

    @staticmethod
    def _build_optical_bokeh_disc_kernel(radius: float) -> np.ndarray:
        """Anti-aliased circular lens aperture circle-of-confusion PSF (matches real DSLR defocus!)."""
        rad = max(0.15, float(radius))
        half_k = max(1, int(np.ceil(rad + 0.75)))
        coords = np.arange(-half_k, half_k + 1, dtype=np.float32)
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        dist = np.sqrt(xx * xx + yy * yy)
        # Subpixel anti-aliased aperture rim + subtle realistic spherical-aberration ring profile
        disc = np.clip(rad + 0.5 - dist, 0.0, 1.0)
        rim_boost = 1.0 + 0.12 * np.square(np.minimum(dist / max(rad, 0.5), 1.0))
        kern = (disc * rim_boost).astype(np.float32)
        k_sum = float(kern.sum())
        if k_sum <= 1e-6:
            kern[half_k, half_k] = 1.0
            return kern
        return kern * np.float32(1.0 / k_sum)

    def _apply_focus_softness(self, img_float: np.ndarray, severity: float, params: DegradationParams) -> np.ndarray:
        """Apply real-style optical bokeh disc PSF, Gaussian defocus, and/or bicubic resampling softness."""
        if severity <= 1e-4:
            return img_float
        sev = float(np.clip(severity, 0.0, 1.0))
        h, w = img_float.shape[:2]
        out = img_float

        if params.focus_mode in (0, 2):
            scale_f = 1.0 if params.focus_mode == 0 else 0.58
            if params.use_bokeh_disc:
                _, max_rad = getattr(self.config, "BOKEH_DISC_RADIUS_RANGE", (1.2, 6.5))
                rad = sev * float(max_rad) * scale_f
                if rad >= 0.20:
                    kern = self._build_optical_bokeh_disc_kernel(rad)
                    out = cv2.filter2D(out, cv2.CV_32F, kern, borderType=cv2.BORDER_REFLECT_101)
            else:
                _, max_sigma = getattr(self.config, "DEFOCUS_SIGMA_RANGE", (0.8, 4.4))
                sigma = sev * float(max_sigma) * scale_f
                if sigma >= 0.15:
                    out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)

        if params.focus_mode in (1, 2):
            min_scale, _ = self.config.DOWNSAMPLE_SCALE_RANGE
            eff_sev = sev if params.focus_mode == 1 else (0.75 * sev)
            scale = 1.0 - eff_sev * (1.0 - float(min_scale))
            if scale < 0.985:
                new_w = max(8, int(round(w * scale)))
                new_h = max(8, int(round(h * scale)))
                down = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_AREA)
                out = np.clip(
                    cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC),
                    np.float32(0.0),
                    np.float32(255.0),
                )

        if params.focus_mode in (0, 2) and params.use_chromatic_aberration and params.chroma_shift_px > 0.05:
            shift = float(sev * params.chroma_shift_px)
            if shift >= 0.05:
                m_r = np.array([[1.0, 0.0, shift], [0.0, 1.0, 0.0]], dtype=np.float32)
                m_b = np.array([[1.0, 0.0, -shift], [0.0, 1.0, 0.0]], dtype=np.float32)
                out = out.copy()
                out[:, :, 0] = cv2.warpAffine(
                    out[:, :, 0], m_r, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                )
                out[:, :, 2] = cv2.warpAffine(
                    out[:, :, 2], m_b, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                )

        return out

    @staticmethod
    def _render_base_banding(img_float: np.ndarray, levels: int) -> np.ndarray:
        h, w = img_float.shape[:2]
        u8 = np.clip(np.round(img_float), 0.0, 255.0).astype(np.uint8)
        half_u8 = cv2.resize(u8, (max(16, w // 2), max(16, h // 2)), interpolation=cv2.INTER_AREA)
        smooth_half = cv2.bilateralFilter(half_u8, d=5, sigmaColor=18.0, sigmaSpace=3)
        smoothed = cv2.resize(smooth_half, (w, h), interpolation=cv2.INTER_LINEAR)

        lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB)
        l_step = 255.0 / float(max(4, levels) - 1)
        ab_step = 255.0 / float(max(6, levels * 2) - 1)
        x256 = np.arange(256, dtype=np.float32)
        lut_lab = np.empty((1, 256, 3), dtype=np.uint8)
        lut_lab[0, :, 0] = np.clip(np.round(x256 / l_step) * l_step, 0.0, 255.0).astype(np.uint8)
        ab_q = np.clip(np.round((x256 - 128.0) / ab_step) * ab_step + 128.0, 0.0, 255.0).astype(np.uint8)
        lut_lab[0, :, 1] = ab_q
        lut_lab[0, :, 2] = ab_q
        return cv2.cvtColor(cv2.LUT(lab, lut_lab), cv2.COLOR_LAB2RGB).astype(np.float32)

    @staticmethod
    def _render_base_jpeg(img_float: np.ndarray, quality: int) -> np.ndarray:
        q = int(np.clip(quality, 5, 95))
        u8_bgr = cv2.cvtColor(np.clip(np.round(img_float), 0.0, 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", u8_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return img_float
        return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)

    @classmethod
    def _render_base_chroma_ringing(cls, img_float: np.ndarray, quality: int) -> np.ndarray:
        """4:2:0 YCrCb chroma subsampling + subtle unsharp ringing + JPEG compression."""
        h, w = img_float.shape[:2]
        u8 = np.clip(np.round(img_float), 0.0, 255.0).astype(np.uint8)
        ycc = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
        y_ch, cr, cb = cv2.split(ycc)
        half_w, half_h = max(8, w // 2), max(8, h // 2)
        cr_sub = cv2.resize(cv2.resize(cr, (half_w, half_h), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
        cb_sub = cv2.resize(cv2.resize(cb, (half_w, half_h), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
        sub_rgb = cv2.cvtColor(cv2.merge([y_ch, cr_sub, cb_sub]), cv2.COLOR_YCrCb2RGB).astype(np.float32)
        # Subtle Gibbs ringing overshoot around high-contrast edges before JPEG
        blur_l = cv2.GaussianBlur(sub_rgb, (0, 0), sigmaX=1.0, sigmaY=1.0)
        ringed = np.clip(sub_rgb + np.float32(0.35) * (sub_rgb - blur_l), np.float32(0.0), np.float32(255.0))
        return cls._render_base_jpeg(ringed, quality)

    def _apply_deartifact_forward(self, img_float: np.ndarray, severity: float, params: DegradationParams) -> np.ndarray:
        """Apply compression/banding/chroma-ringing artifacts with monotonic linear blending."""
        if severity <= 1e-4:
            return img_float
        sev = float(np.clip(severity, 0.0, 1.0))
        cur = img_float
        if params.artifact_mode in (1, 2):
            cur = self._render_base_banding(cur, params.banding_levels)
        if params.artifact_mode in (0, 2):
            cur = self._render_base_jpeg(cur, params.jpeg_quality)
        if params.artifact_mode == 3:
            cur = self._render_base_chroma_ringing(cur, params.jpeg_quality)
        if sev >= 0.999:
            return cur
        return img_float * np.float32(1.0 - sev) + cur * np.float32(sev)

    @staticmethod
    def _apply_haze(img_float: np.ndarray, severity: float, params: DegradationParams) -> np.ndarray:
        """Spatially varying atmospheric scattering veil I = J*t + A*(1 - t)."""
        if severity <= 1e-4:
            return img_float
        sev = float(np.clip(severity, 0.0, 1.0))
        h, w = img_float.shape[:2]
        if params._haze_depth_cache is None or params._haze_depth_cache.shape[:2] != (h, w):
            hrng = np.random.default_rng(params.haze_seed)
            coarse = hrng.uniform(0.50, 1.0, size=(6, 6)).astype(np.float32)
            smooth = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
            vert = np.linspace(0.95, 0.45, h, dtype=np.float32)[:, None]
            params._haze_depth_cache = np.clip(0.55 * smooth + 0.45 * vert, 0.25, 1.0)[:, :, None]
        beta = np.float32(sev * 1.40)
        trans = np.exp(-beta * params._haze_depth_cache)
        air = np.array(params.haze_airlight, dtype=np.float32).reshape(1, 1, 3)
        return np.clip(img_float * trans + air * (np.float32(1.0) - trans), np.float32(0.0), np.float32(255.0))

    @staticmethod
    def _apply_reflection(
        lin_rgb: np.ndarray,
        severity: float,
        params: DegradationParams,
        refl_donor_u8: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Highlight-dominant double-pane glass reflection with planar Fresnel gradient."""
        if severity <= 1e-4:
            return lin_rgb
        sev = float(np.clip(severity, 0.0, 1.0))
        h, w = lin_rgb.shape[:2]
        if params._refl_cache is None or params._refl_cache[0].shape[:2] != (h, w):
            if refl_donor_u8 is not None:
                d_rs = cv2.resize(refl_donor_u8, (w, h), interpolation=cv2.INTER_AREA)
                inv255 = np.float32(1.0 / 255.0)
                donor_lin = np.square(np.clip(d_rs.astype(np.float32) * inv255, 0.0, 1.0)) * np.float32(255.0)
            else:
                donor_lin = np.roll(np.fliplr(lin_rgb), shift=(h // 5, w // 4), axis=(0, 1)).astype(np.float32)
            refl_blur = cv2.GaussianBlur(donor_lin, (0, 0), sigmaX=params.refl_sigma, sigmaY=params.refl_sigma)
            refl_hi = np.power(np.clip(refl_blur / np.float32(255.0), 0.0, 1.0), np.float32(1.75)) * np.float32(255.0)
            m_shift = np.array([[1.0, 0.0, params.refl_dx], [0.0, 1.0, params.refl_dy]], dtype=np.float32)
            refl_ghost = cv2.warpAffine(
                refl_hi, m_shift, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )
            refl_double = (np.float32(0.72) * refl_hi + np.float32(0.28) * refl_ghost) * np.array(
                [0.94, 1.00, 0.98], dtype=np.float32
            ).reshape(1, 1, 3)

            yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, h), np.linspace(-1.0, 1.0, w), indexing="ij")
            proj = np.cos(params.refl_angle) * xx + np.sin(params.refl_angle) * yy
            fresnel = np.clip(0.58 + 0.38 * proj, 0.18, 1.0).astype(np.float32)[:, :, None]
            params._refl_cache = (refl_double, fresnel)

        refl_double, fresnel = params._refl_cache
        w_refl = np.float32(sev * 0.72) * fresnel
        w_trans = np.float32(1.0) - np.float32(0.18) * w_refl
        return np.clip(lin_rgb * w_trans + refl_double * w_refl, np.float32(0.0), np.float32(255.0))

    @staticmethod
    def _apply_radial_chromatic_aberration(
        lin_rgb: np.ndarray, severity: float, params: DegradationParams
    ) -> np.ndarray:
        """Radial R/B magnification scaling: center stays crisp, outer edges fringe purple/cyan."""
        if severity <= 1e-4:
            return lin_rgb
        sev = float(np.clip(severity, 0.0, 1.0))
        h, w = lin_rgb.shape[:2]
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        k = float(sev * params.ca_max_scale)
        map_r_x = (cx + (xx - cx) * (1.0 - k)).astype(np.float32)
        map_r_y = (cy + (yy - cy) * (1.0 - k)).astype(np.float32)
        map_b_x = (cx + (xx - cx) * (1.0 + k)).astype(np.float32)
        map_b_y = (cy + (yy - cy) * (1.0 + k)).astype(np.float32)
        out = lin_rgb.copy()
        out[:, :, 0] = cv2.remap(
            lin_rgb[:, :, 0], map_r_x, map_r_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        out[:, :, 2] = cv2.remap(
            lin_rgb[:, :, 2], map_b_x, map_b_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        return out

    @staticmethod
    def _build_flare_fields(h: int, w: int, params: DegradationParams) -> Tuple[np.ndarray, np.ndarray]:
        """Lazy-compute unit-severity (flare_rgb [0..1+], veil_atten_strength [0..0.35]) for all 5 flare styles."""
        frng = np.random.default_rng(params.flare_seed)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = float(params.flare_cx) * w, float(params.flare_cy) * h
        dx, dy = (xx - cx) / max(h, w), (yy - cy) / max(h, w)
        dist = np.sqrt(dx * dx + dy * dy) + np.float32(1e-4)
        tint_arr = np.array(params.flare_tint, dtype=np.float32).reshape(1, 1, 3)

        style = int(params.flare_style) % 5
        if style == 0:
            # 0: Wave-optics FFT Fraunhofer diffraction corona (computed fast at 128x128)
            n = 128
            py, px = np.mgrid[0:n, 0:n].astype(np.float32) - n * 0.5
            rr = np.sqrt(px * px + py * py) / (n * 0.11)
            th = np.arctan2(py, px)
            sector = (2.0 * np.pi) / 6.0
            ang_m = np.abs(((th + 0.18) % sector) - 0.5 * sector)
            ap = (rr <= (np.cos(0.5 * sector) / np.maximum(np.cos(ang_m), 1e-4))).astype(np.float32)
            dust = cv2.GaussianBlur(frng.normal(0.0, 1.2, size=(n, n)).astype(np.float32), (0, 0), sigmaX=1.2)
            pupil = ap * np.exp(1j * (0.45 * dust))
            pwr = np.power(np.abs(np.fft.fftshift(np.fft.fft2(pupil))) ** 2, 0.38).astype(np.float32)
            pwr /= max(1e-8, float(pwr.max()))
            c_n = n * 0.5
            psf_s = np.zeros((n, n, 3), dtype=np.float32)
            for ch, wl in enumerate((1.07, 1.00, 0.93)):
                m = np.array([[wl, 0.0, c_n * (1.0 - wl)], [0.0, wl, c_n * (1.0 - wl)]], dtype=np.float32)
                psf_s[:, :, ch] = cv2.warpAffine(pwr, m, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            psf_full = cv2.resize(psf_s, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
            x0 = max(0, min(w, w - int(round(cx))))
            y0 = max(0, min(h, h - int(round(cy))))
            crop_psf = psf_full[y0 : y0 + h, x0 : x0 + w]
            veil = (np.exp(-dist * 2.2) * 0.24)[:, :, None] * tint_arr
            flare_rgb = crop_psf * np.float32(1.25) + veil
            veil_atten = np.clip(0.16 * veil, 0.0, 0.30)
            return flare_rgb, veil_atten

        if style == 1:
            # 1: Anamorphic / Smartphone smudge streak
            dx_w, dy_h = (xx - cx) / w, (yy - cy) / h
            s_main = np.exp(-np.abs(dy_h) * 32.0) * np.exp(-np.abs(dx_w) * 2.2) * 0.80
            s_wide = np.exp(-np.abs(dy_h) * 9.5) * np.exp(-np.abs(dx_w) * 1.4) * 0.24
            core = np.exp(-(dx_w * dx_w + dy_h * dy_h) * 52.0) * 0.85
            ana_t = np.array([0.52, 0.78, 1.00], dtype=np.float32).reshape(1, 1, 3)
            core_t = np.array([0.92, 0.97, 1.00], dtype=np.float32).reshape(1, 1, 3)
            smudge = np.exp(-((dx_w * 2.4) ** 2 + (dy_h * 1.2) ** 2)) * 0.14
            flare_rgb = (s_main + s_wide)[:, :, None] * ana_t + core[:, :, None] * core_t + smudge[:, :, None] * core_t
            veil_atten = np.clip(0.10 * smudge[:, :, None], 0.0, 0.20)
            return flare_rgb, veil_atten

        if style == 2:
            # 2: Off-frame golden backlit wash
            wash = np.exp(-dist * 1.75) * 0.46 + np.exp(-dist * 4.5) * 0.40
            flare_rgb = wash[:, :, None] * tint_arr
            veil_atten = np.clip(0.26 * wash[:, :, None], 0.0, 0.35)
            return flare_rgb, veil_atten

        if style == 3:
            # 3: Multi-element DSLR hexagonal/ringed ghost chain
            ox, oy = w * 0.5, h * 0.5
            ghost_layer = np.zeros((h, w, 3), dtype=np.float32)
            specs = [
                (0.28, 11, (0.95, 0.88, 0.72), 0.13, False),
                (0.56, 20, (0.55, 0.92, 0.68), 0.11, True),
                (0.84, 10, (0.98, 0.68, 0.42), 0.14, False),
                (1.12, 28, (0.48, 0.74, 0.98), 0.09, True),
                (1.48, 18, (0.85, 0.58, 0.95), 0.10, False),
                (1.78, 34, (0.45, 0.88, 0.80), 0.08, True),
            ]
            for pos, rad, col, alpha, ring_edge in specs:
                gx = cx + pos * (ox - cx)
                gy = cy + pos * (oy - cy)
                angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False) + float(frng.uniform(0.0, 0.5))
                pts = np.round(np.stack([gx + rad * np.cos(angles), gy + rad * np.sin(angles)], axis=-1)).astype(np.int32)
                mask = np.zeros((h, w), dtype=np.float32)
                cv2.fillConvexPoly(mask, pts, 1.0)
                if ring_edge:
                    in_pts = np.round(
                        np.stack([gx + rad * 0.72 * np.cos(angles), gy + rad * 0.72 * np.sin(angles)], axis=-1)
                    ).astype(np.int32)
                    inner = np.zeros((h, w), dtype=np.float32)
                    cv2.fillConvexPoly(inner, in_pts, 1.0)
                    mask = np.clip(0.42 * mask + 0.58 * (mask - inner), 0.0, 1.0)
                soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2.0, rad * 0.18))
                ghost_layer += (soft * np.float32(alpha))[:, :, None] * np.array(col, dtype=np.float32).reshape(1, 1, 3)
            src_glow = (np.exp(-dist * 4.0) * 0.24)[:, :, None] * tint_arr
            flare_rgb = ghost_layer + src_glow
            veil_atten = np.clip(0.08 * src_glow, 0.0, 0.15)
            return flare_rgb, veil_atten

        # 4: Lens-barrel curved chromatic rainbow crescent arc
        angle = np.arctan2(dy, dx)
        to_c = np.arctan2(0.5 * h - cy, 0.5 * w - cx)
        ang_d = np.arctan2(np.sin(angle - to_c), np.cos(angle - to_c))
        crescent = np.exp(-np.square(ang_d / 0.85))
        r_c = 0.62
        arc = np.stack(
            [
                np.exp(-np.square((dist - (r_c + 0.040)) / 0.042)),
                np.exp(-np.square((dist - r_c) / 0.042)) * 0.90,
                np.exp(-np.square((dist - (r_c - 0.040)) / 0.042)),
            ],
            axis=-1,
        ) * crescent[:, :, None] * np.float32(0.14)
        src_veil = (np.exp(-dist * 2.2) * 0.26)[:, :, None] * tint_arr
        flare_rgb = arc + src_veil
        veil_atten = np.clip(0.14 * src_veil, 0.0, 0.25)
        return flare_rgb, veil_atten

    @classmethod
    def _apply_lens_flare(cls, lin_rgb: np.ndarray, severity: float, params: DegradationParams) -> np.ndarray:
        """Apply realistic optical lens flare in linear light with strict severity monotonicity."""
        if severity <= 1e-4:
            return lin_rgb
        sev = float(np.clip(severity, 0.0, 1.0))
        h, w = lin_rgb.shape[:2]
        if params._flare_cache is None or params._flare_cache[0].shape[:2] != (h, w):
            params._flare_cache = cls._build_flare_fields(h, w, params)
        flare_rgb, veil_atten = params._flare_cache
        atten = np.clip(1.0 - np.float32(sev) * veil_atten, 0.60, 1.0)
        return np.clip(lin_rgb * atten + np.float32(sev * 255.0) * flare_rgb, np.float32(0.0), np.float32(255.0))

    @staticmethod
    def render_lens_distortion(rgb_u8: np.ndarray, k1: float = 0.0, k2: float = 0.0) -> np.ndarray:
        """Full-frame Brown-Conrady radial distortion/correction (k1>0 barrel, k1<0 pincushion)."""
        if abs(k1) < 1e-5 and abs(k2) < 1e-5:
            return rgb_u8.copy()
        h, w = rgb_u8.shape[:2]
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xn = (xx - cx) / max(1.0, cx)
        yn = (yy - cy) / max(1.0, cy)
        r2 = xn * xn + yn * yn
        factor = 1.0 + np.float32(k1) * r2 + np.float32(k2) * (r2 * r2)
        xs, ys = xn * factor, yn * factor
        max_reach = max(1.0, float(np.max(np.abs(xs))), float(np.max(np.abs(ys))))
        map_x = ((xs / np.float32(max_reach)) * cx + cx).astype(np.float32)
        map_y = ((ys / np.float32(max_reach)) * cy + cy).astype(np.float32)
        return cv2.remap(rgb_u8, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)

    def render_with_params(
        self,
        clean_rgb_u8: np.ndarray,
        state_vec: np.ndarray,
        params: DegradationParams,
        refl_donor_u8: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render an RGB uint8 image from `clean_rgb_u8` at signed/unsigned state `state_vec`."""
        cur = clean_rgb_u8.astype(np.float32)

        # 1. Atmospheric haze (scene level)
        if "dehaze" in self.type_to_idx:
            s_haze = float(state_vec[self.type_to_idx["dehaze"]])
            if s_haze > 1e-4:
                cur = self._apply_haze(cur, s_haze, params)

        # 2. Linear-light optical pipeline (reflection -> focus -> radial CA -> motion blur -> lens flare)
        s_refl = float(state_vec[self.type_to_idx["dereflection"]]) if "dereflection" in self.type_to_idx else 0.0
        s_foc = float(state_vec[self.type_to_idx["sharpen_focus"]]) if "sharpen_focus" in self.type_to_idx else 0.0
        s_ca = float(state_vec[self.type_to_idx["dechromatic"]]) if "dechromatic" in self.type_to_idx else 0.0
        s_mot = float(state_vec[self.type_to_idx["deblur_motion"]]) if "deblur_motion" in self.type_to_idx else 0.0
        s_flr = float(state_vec[self.type_to_idx["deflare"]]) if "deflare" in self.type_to_idx else 0.0

        if s_refl > 1e-4 or s_foc > 1e-4 or s_ca > 1e-4 or s_mot > 1e-4 or s_flr > 1e-4:
            inv255 = np.float32(1.0 / 255.0)
            lin = np.square(np.clip(cur * inv255, np.float32(0.0), np.float32(1.0))) * np.float32(255.0)
            if s_refl > 1e-4:
                lin = self._apply_reflection(lin, s_refl, params, refl_donor_u8=refl_donor_u8)
            if s_foc > 1e-4:
                lin = self._apply_focus_softness(lin, s_foc, params)
            if s_ca > 1e-4:
                lin = self._apply_radial_chromatic_aberration(lin, s_ca, params)
            if s_mot > 1e-4:
                _, max_k = self.config.MOTION_BLUR_KERNEL_RANGE
                length_px = float(np.clip(s_mot, 0.0, 1.0)) * (float(max_k) - 1.0)
                lin = self._continuous_curved_motion_blur(
                    lin,
                    length_px,
                    params.motion_angle,
                    curve_bend=params.motion_curve_bend,
                    dwell_end=params.motion_dwell_end,
                    shake_jitter=params.motion_shake_jitter,
                    pattern=params.motion_pattern,
                )
            if s_flr > 1e-4:
                lin = self._apply_lens_flare(lin, s_flr, params)
            cur = np.sqrt(np.clip(lin * inv255, np.float32(0.0), np.float32(1.0))) * np.float32(255.0)

        # 3 & 4. Exact Commutative Lie-Group Exposure + Warmth in float32 [0, 1]
        exp_val = float(state_vec[self.type_to_idx["exposure"]]) if "exposure" in self.type_to_idx else 0.0
        warm_val = float(state_vec[self.type_to_idx["warmth"]]) if "warmth" in self.type_to_idx else 0.0
        if abs(exp_val) > 1e-4 or abs(warm_val) > 1e-4:
            x01 = np.clip(cur * np.float32(1.0 / 255.0), np.float32(0.0), np.float32(1.0))
            log2_r = self.exp_kappa * exp_val + self.warm_r_kappa * warm_val
            log2_g = self.exp_kappa * exp_val + self.warm_g_kappa * warm_val
            log2_b = self.exp_kappa * exp_val - self.warm_b_kappa * warm_val
            gains = np.array([2.0**log2_r, 2.0**log2_g, 2.0**log2_b], dtype=np.float32).reshape(1, 1, 3)
            cur = self._rational_tone_curve(x01, gains) * np.float32(255.0)

        # 5. Compression / banding quantization
        if "deartifact" in self.type_to_idx:
            s_art = float(state_vec[self.type_to_idx["deartifact"]])
            if s_art > 1e-4:
                cur = self._apply_deartifact_forward(cur, s_art, params)

        # 6. Sensor / ISO grain (applied last so grain realization stays intact across deartifact residuals)
        if "denoise" in self.type_to_idx:
            s_noi = float(state_vec[self.type_to_idx["denoise"]])
            if s_noi > 1e-4:
                _, max_std = self.config.GAUSSIAN_NOISE_STD_RANGE
                noise_std = np.float32(float(np.clip(s_noi, 0.0, 1.0)) * float(max_std))
                if params.heteroscedastic_noise:
                    lum_mod = np.float32(0.70) + np.float32(0.60) * np.sqrt(
                        np.clip(cur * np.float32(1.0 / 255.0), np.float32(0.0), np.float32(1.0))
                    )
                    noise_delta = params.get_noise_unit() * (noise_std * lum_mod)
                else:
                    noise_delta = params.get_noise_unit() * noise_std
                cur = np.clip(cur + noise_delta, np.float32(0.0), np.float32(255.0))

        return np.clip(np.round(cur), np.float32(0.0), np.float32(255.0)).astype(np.uint8)

    def sample_source_severity(
        self, rng: np.random.Generator = None, params: Optional[DegradationParams] = None
    ) -> np.ndarray:
        """Sample 1 to 4 active degradations (unsigned in [min_sev, 1], signed in +/-[min_sev, 0.95])."""
        rng = rng or np.random.default_rng()
        eff_compound_prob = max(0.15, self.compound_prob * (1.0 - 0.70 * self.curriculum_single_boost))

        if self.max_compound > 1 and rng.random() < eff_compound_prob:
            u_k = float(rng.random())
            if self.max_compound >= 4 and u_k < 0.22:
                k = 4
            elif self.max_compound >= 3 and u_k < 0.54:
                k = 3
            else:
                k = 2
        else:
            k = 1

        # Up-weight structural deconvolution (`deblur_motion`, `sharpen_focus`) so motion/focus experts never starve!
        type_weights = np.array(
            [
                1.55 if name == "deblur_motion" else (1.40 if name == "sharpen_focus" else 1.0)
                for name in self.degradation_types
            ],
            dtype=np.float64,
        )
        type_probs = type_weights / np.maximum(type_weights.sum(), 1e-8)

        chosen = list(rng.choice(self.num_types, size=min(k, self.num_types), replace=False, p=type_probs))
        chosen_names = [self.degradation_types[i] for i in chosen]

        if "deblur_motion" in chosen_names and "sharpen_focus" in chosen_names and len(chosen) > 1 and rng.random() < 0.35:
            drop_name = "deblur_motion" if rng.random() < 0.4 else "sharpen_focus"
            chosen.remove(self.type_to_idx[drop_name])

        damping = 1.0 if len(chosen) == 1 else (0.90 if len(chosen) == 2 else (0.82 if len(chosen) == 3 else 0.76))
        s_src = np.zeros(self.num_types, dtype=np.float32)
        for idx in chosen:
            mag = float(np.clip(float(rng.uniform(self.min_severity, 0.95)) * damping, self.min_severity, 0.95))
            if self.is_signed(idx):
                if params is not None:
                    sign = params.exposure_sign if self.degradation_types[idx] == "exposure" else params.warmth_sign
                else:
                    sign = -1.0 if rng.random() < 0.5 else 1.0
                s_src[idx] = float(sign * mag)
            else:
                s_src[idx] = mag
        return s_src

    def sample_action_vector(self, s_src: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        """Sample diverse controllable user action vectors (4 unsigned [0, 1] + 2 signed [-1, +1])."""
        rng = rng or np.random.default_rng()
        active_indices = [i for i in range(self.num_types) if abs(float(s_src[i])) > 1e-4]
        inactive_unsigned = [i for i in self.unsigned_indices if abs(float(s_src[i])) <= 1e-4]

        p_sel = self.config.ACTION_DIST_SELECTIVE * (1.0 - 0.5 * self.curriculum_single_boost)
        p_part = self.config.ACTION_DIST_PARTIAL
        p_full = self.config.ACTION_DIST_FULL + (self.config.ACTION_DIST_SELECTIVE - p_sel)
        p_fp = self.config.ACTION_DIST_FALSE_POS
        u = float(rng.random())

        action = np.zeros(self.num_types, dtype=np.float32)
        full_act = self.compute_full_restore_action(s_src)

        if u < p_sel and len(active_indices) >= 2:
            num_fix = int(rng.integers(1, len(active_indices)))
            fix_ids = rng.choice(active_indices, size=num_fix, replace=False)
            for idx in fix_ids:
                frac = 1.0 if rng.random() < 0.75 else float(rng.uniform(0.5, 1.0))
                action[idx] = full_act[idx] * frac

        elif u < p_sel + p_part:
            for idx in active_indices:
                if rng.random() < 0.85:
                    frac = float(rng.uniform(0.25, 0.85))
                    action[idx] = full_act[idx] * frac

        elif u < p_sel + p_part + p_full:
            action = full_act.copy()
            # Keep exact active sliders 95% of the time so inactive `deblur_motion` isn't flooded with no-op `1.0`s!
            if rng.random() < 0.05:
                for idx in self.unsigned_indices:
                    action[idx] = 1.0

        elif u < p_sel + p_part + p_full + p_fp:
            if rng.random() < 0.50:
                for idx in active_indices:
                    if rng.random() < 0.70:
                        action[idx] = full_act[idx]
                if inactive_unsigned:
                    n_fp = int(rng.integers(1, min(3, len(inactive_unsigned)) + 1))
                    for idx in rng.choice(inactive_unsigned, size=n_fp, replace=False):
                        action[idx] = float(rng.uniform(0.4, 1.0))
            else:
                for idx in active_indices:
                    if not self.is_signed(idx) and rng.random() < 0.75:
                        action[idx] = 1.0
                for idx in self.signed_indices:
                    if rng.random() < 0.75:
                        art_delta = float(rng.uniform(0.25, 0.80)) * (-1.0 if rng.random() < 0.5 else 1.0)
                        lo_a = max(-1.0, -0.95 - float(s_src[idx]))
                        hi_a = min(1.0, 0.95 - float(s_src[idx]))
                        if hi_a > lo_a + 0.10:
                            action[idx] = float(np.clip(art_delta, lo_a, hi_a))

        else:
            action[:] = 0.0

        return action

    def sample_spatial_mask(self, clean_rgb_u8: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        """Generate a smooth feathered spatial region mask M in [0, 1]^(H, W) via fast coarse grid."""
        rng = rng or np.random.default_rng()
        h, w = clean_rgb_u8.shape[:2]
        gh, gw = max(16, h // 4), max(16, w // 4)
        mode = float(rng.random())

        if mode < 0.45:
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
        refl_donor_u8: Optional[np.ndarray] = None,
    ):
        """Fast synthesis returning compact components (x_src, x_tgt, a_fg, a_bg, mask_2d, s_src)."""
        rng = rng or np.random.default_rng()
        h, w = clean_rgb_u8.shape[:2]
        params = self.sample_params(h, w, rng=rng)
        s_src = self.sample_source_severity(rng=rng, params=params)
        x_src = self.render_with_params(clean_rgb_u8, s_src, params, refl_donor_u8=refl_donor_u8)

        if align_only:
            zeros_a = np.zeros(self.num_types, dtype=np.float32)
            ones_mask = np.ones((h, w), dtype=np.float32)
            return x_src, x_src, zeros_a, zeros_a, ones_mask, s_src

        a_fg = self.sample_action_vector(s_src, rng=rng)
        s_tgt_fg = self.compute_target_state(s_src, a_fg)
        if np.allclose(s_tgt_fg, s_src, atol=1e-4):
            x_tgt_fg = x_src.copy()
        elif np.allclose(s_tgt_fg, 0.0, atol=1e-4):
            x_tgt_fg = clean_rgb_u8.copy()
        else:
            x_tgt_fg = self.render_with_params(clean_rgb_u8, s_tgt_fg, params, refl_donor_u8=refl_donor_u8)

        eff_mask_prob = self.spatial_mask_prob * (1.0 - 0.65 * self.curriculum_single_boost)
        if rng.random() < eff_mask_prob:
            if rng.random() < 0.68:
                a_bg = np.zeros(self.num_types, dtype=np.float32)
            else:
                a_bg = self.sample_action_vector(s_src, rng=rng)

            s_tgt_bg = self.compute_target_state(s_src, a_bg)
            if np.allclose(s_tgt_bg, s_src, atol=1e-4):
                x_tgt_bg = x_src
            elif np.allclose(s_tgt_bg, 0.0, atol=1e-4):
                x_tgt_bg = clean_rgb_u8
            elif np.allclose(s_tgt_bg, s_tgt_fg, atol=1e-4):
                x_tgt_bg = x_tgt_fg
            else:
                x_tgt_bg = self.render_with_params(clean_rgb_u8, s_tgt_bg, params, refl_donor_u8=refl_donor_u8)

            mask = self.sample_spatial_mask(clean_rgb_u8, rng=rng)
            mask_3d = mask[:, :, None]
            x_tgt = np.clip(
                np.round(
                    mask_3d * x_tgt_fg.astype(np.float32)
                    + (np.float32(1.0) - mask_3d) * x_tgt_bg.astype(np.float32)
                ),
                np.float32(0.0),
                np.float32(255.0),
            ).astype(np.uint8)
        else:
            a_bg = a_fg
            mask = np.ones((h, w), dtype=np.float32)
            x_tgt = x_tgt_fg

        return x_src, x_tgt, a_fg, a_bg, mask, s_src

    def synthesize_real_pair_components(
        self,
        blur_rgb_u8: np.ndarray,
        sharp_rgb_u8: np.ndarray,
        blur_channel_name: str = "deblur_motion",
        rng: np.random.Generator = None,
        align_only: bool = False,
    ):
        """Compositional Real-Pair Wrapper for GoPro, HIDE, RealBlur, and DPDD pairs!

        Layers synchronized invertible downstream operators (`exposure`, `warmth`, `deartifact`, `denoise`)
        onto real camera `(blur_rgb_u8, sharp_rgb_u8)` pairs so the model trains all 6 selective sliders
        directly on authentic real-world camera shake and DSLR lens defocus!
        """
        rng = rng or np.random.default_rng()
        h, w = sharp_rgb_u8.shape[:2]
        params = self.sample_params(h, w, rng=rng)

        blur_idx = self.type_to_idx.get(blur_channel_name, self.type_to_idx.get("deblur_motion", 0))
        downstream_candidates = [
            i
            for i, name in enumerate(self.degradation_types)
            if i != blur_idx and name not in ("deblur_motion", "sharpen_focus")
        ]

        # Sample 0, 1, or 2 downstream orthogonal degradations (exposure, warmth, deartifact, denoise)
        s_down = np.zeros(self.num_types, dtype=np.float32)
        u_mode = float(rng.random())
        need_downstream = (0.06 <= u_mode < 0.22)  # selective blur preservation needs >= 1 downstream edit
        if downstream_candidates and (need_downstream or rng.random() < 0.58):
            n_extra = 1 if rng.random() < 0.72 else min(2, len(downstream_candidates))
            for idx in rng.choice(downstream_candidates, size=n_extra, replace=False):
                mag = float(rng.uniform(self.min_severity, 0.85))
                if self.is_signed(idx):
                    sign = params.exposure_sign if self.degradation_types[idx] == "exposure" else params.warmth_sign
                    s_down[idx] = float(sign * mag)
                else:
                    s_down[idx] = mag

        if align_only:
            x_src = self.render_with_params(blur_rgb_u8, s_down, params)
            s_src = s_down.copy()
            s_src[blur_idx] = 1.0
            zeros_a = np.zeros(self.num_types, dtype=np.float32)
            ones_mask = np.ones((h, w), dtype=np.float32)
            return x_src, sharp_rgb_u8, zeros_a, zeros_a, ones_mask, s_src

        if u_mode < 0.06:
            # 6%: False-Positive Safety on Real Clean Sharp Photo (prevents oversharpening clean photos!)
            base_src = sharp_rgb_u8
            base_tgt_fg = sharp_rgb_u8
            s_src = s_down.copy()
            a_fg = self.compute_full_restore_action(s_down)
            a_fg[blur_idx] = float(rng.uniform(0.55, 1.0))
            s_tgt_down = self.compute_target_state(s_down, a_fg)
        elif u_mode < 0.22:
            # 16%: Selective Real-Blur Preservation (`a_blur = 0.0`: clean noise/tone, KEEP real camera blur!)
            base_src = blur_rgb_u8
            base_tgt_fg = blur_rgb_u8
            s_src = s_down.copy()
            s_src[blur_idx] = 1.0
            a_fg = self.compute_full_restore_action(s_down)
            a_fg[blur_idx] = 0.0
            s_tgt_down = self.compute_target_state(s_down, a_fg)
        else:
            # 78%: Active Real Camera Deblurring (`a_blur = 1.0`) + full or selective downstream fixes!
            base_src = blur_rgb_u8
            base_tgt_fg = sharp_rgb_u8
            s_src = s_down.copy()
            s_src[blur_idx] = 1.0
            a_fg = np.zeros(self.num_types, dtype=np.float32)
            a_fg[blur_idx] = 1.0
            full_down = self.compute_full_restore_action(s_down)
            for idx in downstream_candidates:
                if abs(float(s_down[idx])) > 1e-4:
                    if rng.random() < 0.72:
                        a_fg[idx] = full_down[idx]
            if rng.random() < 0.20:
                for idx in self.unsigned_indices:
                    if idx == blur_idx or abs(float(s_down[idx])) > 1e-4:
                        a_fg[idx] = 1.0
            s_tgt_down = self.compute_target_state(s_down, a_fg)

        x_src = (
            base_src.copy()
            if np.allclose(s_down, 0.0, atol=1e-4)
            else self.render_with_params(base_src, s_down, params)
        )
        x_tgt_fg = (
            base_tgt_fg.copy()
            if np.allclose(s_tgt_down, 0.0, atol=1e-4)
            else self.render_with_params(base_tgt_fg, s_tgt_down, params)
        )

        eff_mask_prob = 0.55 * self.spatial_mask_prob * (1.0 - 0.65 * self.curriculum_single_boost)
        if rng.random() < eff_mask_prob:
            a_bg = np.zeros(self.num_types, dtype=np.float32)
            mask = self.sample_spatial_mask(sharp_rgb_u8, rng=rng)
            mask_3d = mask[:, :, None]
            x_tgt = np.clip(
                np.round(
                    mask_3d * x_tgt_fg.astype(np.float32)
                    + (np.float32(1.0) - mask_3d) * x_src.astype(np.float32)
                ),
                np.float32(0.0),
                np.float32(255.0),
            ).astype(np.uint8)
        else:
            a_bg = a_fg
            mask = np.ones((h, w), dtype=np.float32)
            x_tgt = x_tgt_fg

        return x_src, x_tgt, a_fg, a_bg, mask, s_src

    def synthesize_pair(self, clean_rgb_u8: np.ndarray, rng: np.random.Generator = None):
        """Synthesize seed-synchronized (x_src, action_map [K, H, W], x_tgt, s_src)."""
        x_src, x_tgt, a_fg, a_bg, mask, s_src = self.synthesize_components(clean_rgb_u8, rng=rng, align_only=False)
        h, w = clean_rgb_u8.shape[:2]
        if a_bg is a_fg or np.array_equal(a_fg, a_bg):
            action_map = np.broadcast_to(a_fg[:, None, None], (self.num_types, h, w)).copy().astype(np.float32)
        else:
            action_map = (
                mask[None, :, :] * a_fg[:, None, None]
                + (np.float32(1.0) - mask[None, :, :]) * a_bg[:, None, None]
            ).astype(np.float32)
        return x_src, action_map, x_tgt, s_src


class ControllableRestorationDataset(_TorchDataset):
    """Hybrid Clean-Reference + Real-Paired Deblurring Dataset with Sharpness-Gated Crops."""

    def __init__(
        self,
        image_paths,
        config,
        is_train: bool = True,
        align_only: bool = False,
        compact_ipc: bool = False,
        paired_records: Optional[Sequence[Tuple[Path, Path, str]]] = None,
    ):
        self.image_paths = list(image_paths)
        self.paired_records = list(paired_records or [])
        self.config = config
        self.is_train = is_train
        self.align_only = align_only
        self.compact_ipc = compact_ipc
        self.patch_size = config.PATCH_SIZE
        self.pipeline = SelectiveDegradationPipeline(config)
        self.paired_prob = float(getattr(config, "PAIRED_REAL_PROB", 0.35)) if self.paired_records else 0.0

        self._sharpness_candidates = max(1, int(getattr(config, "CLEAN_PATCH_SHARPNESS_CANDIDATES", 3)))
        self._min_lap_var = float(getattr(config, "CLEAN_PATCH_MIN_LAPLACIAN_VAR", 18.0))
        self._smooth_allow_prob = float(getattr(config, "CLEAN_PATCH_SMOOTH_ALLOW_PROB", 0.15))

        self._cache_capacity = int(getattr(config, "DECODE_CACHE_SIZE", 16)) if is_train else 0
        self._paired_cache_capacity = max(4, self._cache_capacity // 2) if (is_train and self.paired_records) else 0
        self._max_crops_per_decode = int(getattr(config, "CROPS_PER_DECODE", 6)) if is_train else 1
        self._decoded_ring = []
        self._paired_ring = []

    def __len__(self):
        return len(self.image_paths) if self.image_paths else len(self.paired_records)

    def set_curriculum_boost(self, boost: float) -> None:
        self.pipeline.curriculum_single_boost = float(np.clip(boost, 0.0, 1.0))

    @staticmethod
    def _estimate_patch_laplacian_var(crop_rgb: np.ndarray) -> float:
        """Fast 2x-subsampled grayscale Laplacian variance to screen out blurry bokeh patches."""
        small = crop_rgb[::2, ::2]
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        return float(lap.var())

    @staticmethod
    def _augment_crop_pair(crop_a: np.ndarray, crop_b: Optional[np.ndarray] = None):
        if random.random() < 0.5:
            crop_a = np.fliplr(crop_a)
            if crop_b is not None:
                crop_b = np.fliplr(crop_b)
        if random.random() < 0.5:
            crop_a = np.flipud(crop_a)
            if crop_b is not None:
                crop_b = np.flipud(crop_b)
        k = random.randint(0, 3)
        if k > 0:
            crop_a = np.rot90(crop_a, k)
            if crop_b is not None:
                crop_b = np.rot90(crop_b, k)
        if crop_b is not None:
            return np.ascontiguousarray(crop_a), np.ascontiguousarray(crop_b)
        return np.ascontiguousarray(crop_a)

    def _crop_and_augment(self, img_rgb: np.ndarray) -> np.ndarray:
        h, w = img_rgb.shape[:2]
        ps = self.patch_size

        if h >= ps and w >= ps:
            if self.is_train:
                # Laplacian Sharpness Gate: reject out-of-focus background bokeh in OST/Flickr2K!
                if (
                    self._sharpness_candidates > 1
                    and (h > ps or w > ps)
                    and random.random() >= self._smooth_allow_prob
                ):
                    best_crop = None
                    best_var = -1.0
                    for _ in range(self._sharpness_candidates):
                        y = random.randint(0, h - ps)
                        x = random.randint(0, w - ps)
                        cand = img_rgb[y : y + ps, x : x + ps]
                        v = self._estimate_patch_laplacian_var(cand)
                        if v > best_var:
                            best_var = v
                            best_crop = cand
                        if v >= self._min_lap_var:
                            break
                    crop = best_crop
                else:
                    y = random.randint(0, h - ps)
                    x = random.randint(0, w - ps)
                    crop = img_rgb[y : y + ps, x : x + ps]
            else:
                y = (h - ps) // 2
                x = (w - ps) // 2
                crop = img_rgb[y : y + ps, x : x + ps]
        else:
            crop = cv2.resize(img_rgb, (ps, ps), interpolation=cv2.INTER_CUBIC)

        if self.is_train:
            return self._augment_crop_pair(crop)
        return crop.copy()

    def _decode_rgb(self, image_path: str) -> np.ndarray:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_crop_with_retry(self, idx: int, max_attempts: int = 6) -> np.ndarray:
        if self.is_train and self._cache_capacity > 0 and len(self._decoded_ring) >= self._cache_capacity:
            slot = random.randrange(len(self._decoded_ring))
            rgb_cached, rem = self._decoded_ring[slot]
            if rem > 1:
                self._decoded_ring[slot][1] = rem - 1
            else:
                self._decoded_ring.pop(slot)
            return self._crop_and_augment(rgb_cached)

        n = len(self.image_paths)
        ps = self.patch_size
        last_err = None
        fallback_small = None
        for attempt in range(max_attempts):
            cand_idx = (idx + attempt) % n
            try:
                rgb_full = self._decode_rgb(self.image_paths[cand_idx])
                # Skip sub-patch low-res thumbnails during training whenever full-res images exist!
                if self.is_train and n > 1 and (rgb_full.shape[0] < ps or rgb_full.shape[1] < ps):
                    fallback_small = rgb_full
                    continue
                if (
                    self.is_train
                    and self._cache_capacity > 0
                    and self._max_crops_per_decode > 1
                    and rgb_full.shape[0] >= ps * 2
                ):
                    self._decoded_ring.append([rgb_full, self._max_crops_per_decode - 1])
                return self._crop_and_augment(rgb_full)
            except Exception as err:
                last_err = err

        if fallback_small is not None:
            return self._crop_and_augment(fallback_small)
        raise RuntimeError(f"Failed to read valid image after {max_attempts} attempts: {last_err}")

    def _crop_paired_arrays(self, blur_rgb: np.ndarray, sharp_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ps = self.patch_size
        h, w = sharp_rgb.shape[:2]
        if h >= ps and w >= ps:
            if self.is_train:
                # Multi-scale window (`256..470 -> 256`) bridges 1280x720 GoPro blur across all real photo scales,
                # and 4-candidate Active Blur Gate rejects flat sky/road crops where `blur == sharp`!
                min_hw = min(h, w)
                if min_hw >= int(ps * 1.45) and random.random() < 0.45:
                    win = random.randint(int(ps * 1.15), min(min_hw, int(ps * 1.85)))
                else:
                    win = ps

                best_b, best_s = None, None
                best_score = -1.0
                n_cands = 4 if (h > win or w > win) else 1
                for _ in range(n_cands):
                    y = random.randint(0, h - win)
                    x = random.randint(0, w - win)
                    bc = blur_rgb[y : y + win, x : x + win]
                    sc = sharp_rgb[y : y + win, x : x + win]
                    b_sub = bc[::4, ::4].astype(np.float32)
                    s_sub = sc[::4, ::4].astype(np.float32)
                    motion_diff = float(np.mean(np.abs(s_sub - b_sub)))
                    edge_energy = float(np.mean(np.abs(s_sub[:, 1:] - s_sub[:, :-1])))
                    score = motion_diff + 0.35 * edge_energy
                    if score > best_score:
                        best_score = score
                        best_b, best_s = bc, sc
                    if motion_diff >= 4.5 and edge_energy >= 6.0:
                        break

                if win != ps:
                    b_crop = cv2.resize(best_b, (ps, ps), interpolation=cv2.INTER_AREA)
                    s_crop = cv2.resize(best_s, (ps, ps), interpolation=cv2.INTER_AREA)
                else:
                    b_crop, s_crop = best_b, best_s
            else:
                y = (h - ps) // 2
                x = (w - ps) // 2
                b_crop = blur_rgb[y : y + ps, x : x + ps]
                s_crop = sharp_rgb[y : y + ps, x : x + ps]
        else:
            b_crop = cv2.resize(blur_rgb, (ps, ps), interpolation=cv2.INTER_CUBIC)
            s_crop = cv2.resize(sharp_rgb, (ps, ps), interpolation=cv2.INTER_CUBIC)

        if self.is_train:
            return self._augment_crop_pair(b_crop, s_crop)
        return b_crop.copy(), s_crop.copy()

    def _load_paired_crop_with_retry(self, idx: int, max_attempts: int = 5) -> Tuple[np.ndarray, np.ndarray, str]:
        if (
            self.is_train
            and self._paired_cache_capacity > 0
            and len(self._paired_ring) >= self._paired_cache_capacity
        ):
            slot = random.randrange(len(self._paired_ring))
            b_cached, s_cached, ch_cached, rem = self._paired_ring[slot]
            if rem > 1:
                self._paired_ring[slot][3] = rem - 1
            else:
                self._paired_ring.pop(slot)
            b_crop, s_crop = self._crop_paired_arrays(b_cached, s_cached)
            return b_crop, s_crop, ch_cached

        m = len(self.paired_records)
        ps = self.patch_size
        last_err = None
        for attempt in range(max_attempts):
            p_idx = (idx + attempt) % m
            blur_path, sharp_path, blur_channel = self.paired_records[p_idx]
            try:
                blur_rgb = self._decode_rgb(blur_path)
                sharp_rgb = self._decode_rgb(sharp_path)
                if blur_rgb.shape[:2] != sharp_rgb.shape[:2]:
                    sh, sw = sharp_rgb.shape[:2]
                    blur_rgb = cv2.resize(blur_rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)

                if (
                    self.is_train
                    and self._paired_cache_capacity > 0
                    and self._max_crops_per_decode > 1
                    and sharp_rgb.shape[0] >= int(ps * 1.5)
                ):
                    self._paired_ring.append(
                        [blur_rgb, sharp_rgb, blur_channel, max(2, self._max_crops_per_decode - 1)]
                    )
                b_crop, s_crop = self._crop_paired_arrays(blur_rgb, sharp_rgb)
                return b_crop, s_crop, blur_channel
            except Exception as err:
                last_err = err
        raise RuntimeError(f"Failed to read valid paired deblur sample after {max_attempts} attempts: {last_err}")

    def __getitem__(self, idx):
        rng = np.random.default_rng() if self.is_train else np.random.default_rng(self.config.SEED + idx)

        use_paired = bool(
            self.paired_records
            and (not self.image_paths or (self.is_train and float(rng.random()) < self.paired_prob))
        )

        if use_paired:
            pair_idx = int(rng.integers(0, len(self.paired_records))) if self.is_train else idx
            blur_u8, sharp_u8, blur_ch = self._load_paired_crop_with_retry(pair_idx)
            clean_u8 = sharp_u8
            x_src_u8, x_tgt_u8, a_fg, a_bg, mask_2d, s_src = self.pipeline.synthesize_real_pair_components(
                blur_u8, sharp_u8, blur_channel_name=blur_ch, rng=rng, align_only=self.align_only
            )
        else:
            clean_u8 = self._load_crop_with_retry(idx % max(1, len(self.image_paths)))
            donor_u8 = None
            if self.is_train and len(self._decoded_ring) >= 2:
                d_slot = int(rng.integers(0, len(self._decoded_ring)))
                donor_u8 = self._crop_and_augment(self._decoded_ring[d_slot][0])
            x_src_u8, x_tgt_u8, a_fg, a_bg, mask_2d, s_src = self.pipeline.synthesize_components(
                clean_u8, rng=rng, align_only=self.align_only, refl_donor_u8=donor_u8
            )

        inv_255 = np.float32(1.0 / 255.0)
        clean_t = torch.from_numpy(clean_u8).permute(2, 0, 1).float().mul_(inv_255)
        src_t = torch.from_numpy(x_src_u8).permute(2, 0, 1).float().mul_(inv_255)

        if self.align_only:
            return {"clean": clean_t, "source": src_t}

        tgt_t = torch.from_numpy(x_tgt_u8).permute(2, 0, 1).float().mul_(inv_255)
        s_src_t = torch.from_numpy(s_src)

        if self.compact_ipc:
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
                mask_2d[None, :, :] * a_fg[:, None, None]
                + (np.float32(1.0) - mask_2d[None, :, :]) * a_bg[:, None, None]
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
    """Reconstruct [B, K, H, W] signed/unsigned action_map directly on GPU from compact IPC components."""
    if "action_map" in batch:
        return batch["action_map"].to(device, non_blocking=True)
    a_fg = batch["a_fg"].to(device, non_blocking=True).float()[:, :, None, None]
    a_bg = batch["a_bg"].to(device, non_blocking=True).float()[:, :, None, None]
    mask = batch["spatial_mask"].to(device, non_blocking=True).float()
    return mask * a_fg + (1.0 - mask) * a_bg


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
_BLUR_DIR_NAMES = {
    "blur",
    "blur_gamma",
    "blurred",
    "motion_blur",
    "motion_blurred",
    "defocus",
    "defocus_blur",
    "defocused_blurred",
    "input",
    "inputc",
    "source",
    "lq",
    "degraded",
}
_SHARP_DIR_NAMES = {"sharp", "gt", "groundtruth", "ground_truth", "target", "clean", "hq", "hr"}
_NON_CLEAN_DIR_NAMES = _BLUR_DIR_NAMES | {"noisy", "noise", "lr", "low", "corrupted"}
# Exclude synthetic low-res face/OCR subsets (`CelebA`, `Text`) & dark un-tonemapped linear RAW (`RealBlur_R`)
# inside curated collections so paired & clean training stays 100% high-res natural sRGB photography!
_EXCLUDED_PAIR_SUBSTRINGS = ("celeba", "text", "bmvc", "widerface", "realblur_r", "realblur-r")


def _is_excluded_pair_token(path_or_name: str, exclude_domain_subsets: bool = True) -> bool:
    if not exclude_domain_subsets:
        return False
    low = path_or_name.lower()
    return any(tok in low for tok in _EXCLUDED_PAIR_SUBSTRINGS)


def _is_degraded_or_excluded_for_clean_pool(path: Path, root: Optional[Path] = None) -> bool:
    """Quarantine pre-blurred/noisy folders & synthetic CelebA/Text/RAW subtrees from entering clean `image_paths`!"""
    try:
        rel_str = os.path.relpath(str(path), str(root)) if root is not None else str(path)
    except ValueError:
        rel_str = str(path)
    parts = [p.lower() for p in rel_str.replace("\\", "/").split("/")[:-1] if p and p != "."]
    if any(p in _NON_CLEAN_DIR_NAMES for p in parts):
        return True
    return any(_is_excluded_pair_token(p, exclude_domain_subsets=True) for p in parts)


def _collect_images(folder: Path) -> List[Path]:
    """Recursively collect image files (following symlinks for Kaggle /kaggle/input mounts)."""
    folder_str = str(folder)
    if not os.path.exists(folder_str):
        return []
    found = []
    for root_dir, _dirs, files in os.walk(folder_str, followlinks=True):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in _IMG_EXTS:
                found.append(Path(os.path.join(root_dir, fname)))
    return sorted(set(found))


def _normalize_pair_stem(rel_key: str) -> str:
    """Normalize pair filenames across prefix & suffix conventions (`blur_001` <-> `gt_001`, `01_M` <-> `01_S`)."""
    parent, base = os.path.split(rel_key.replace("\\", "/"))
    low = base.lower()
    for pfx in ("blur_", "blurred_", "sharp_", "gt_", "target_", "input_", "source_", "defocus_", "motion_"):
        if low.startswith(pfx) and len(base) > len(pfx):
            base = base[len(pfx) :]
            low = base.lower()
            break
    for sfx in ("_blurred", "_blur", "_sharp", "_gt", "_clean", "_target", "_input", "_defocus", "_motion"):
        if low.endswith(sfx) and len(base) > len(sfx):
            base = base[: -len(sfx)]
            low = base.lower()
            break
    if len(base) > 3 and low[-2:] in ("_s", "_m", "_f"):
        base = base[:-2]
    return f"{parent}/{base}" if parent else base


def _infer_pair_channel(path_str: str, default_channel: str) -> str:
    low = path_str.lower()
    if any(tok in low for tok in ("dpdd", "defocus", "realdof", "bokeh", "dof")):
        return "sharpen_focus"
    if any(tok in low for tok in ("motion", "gopro", "hide", "realblur")):
        return "deblur_motion"
    return default_channel


def discover_paired_blur_datasets(
    roots_spec,
    default_channel: str = "deblur_motion",
    is_train: bool = True,
    exclude_domain_subsets: bool = True,
) -> List[Tuple[Path, Path, str]]:
    """Auto-discover paired `(blur_path, sharp_path, blur_channel)` across GoPro, HIDE, RealBlur-J, BlurDataset & DPDD!"""
    if not roots_spec:
        return []
    if isinstance(roots_spec, (str, Path)):
        raw_roots = [s.strip() for s in str(roots_spec).split(",") if s.strip()]
    else:
        raw_roots = [str(x).strip() for x in roots_spec if str(x).strip()]

    seen = set()
    all_pairs: List[Tuple[Path, Path, str]] = []

    for r_str in raw_roots:
        if not os.path.exists(r_str):
            continue
        root_path = Path(r_str)
        root_pairs: List[Tuple[Path, Path, str]] = []

        # 1. Check RealBlur/HIDE-style `.txt` pair index files first (pruning CelebA/Text/RAW subtrees immediately)
        for cur_dir, dirs, files in os.walk(r_str, followlinks=True):
            dirs[:] = [d for d in dirs if not _is_excluded_pair_token(d, exclude_domain_subsets)]
            for fname in files:
                low_f = fname.lower()
                if not low_f.endswith(".txt") or _is_excluded_pair_token(low_f, exclude_domain_subsets):
                    continue
                if is_train and ("test" in low_f or "val" in low_f) and "train" not in low_f:
                    continue
                if (not is_train) and "train" in low_f and not ("test" in low_f or "val" in low_f):
                    continue
                txt_path = Path(cur_dir) / fname
                try:
                    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                except Exception:
                    continue
                for line in lines[:50000]:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    if _is_excluded_pair_token(parts[0], exclude_domain_subsets) or _is_excluded_pair_token(
                        parts[1], exclude_domain_subsets
                    ):
                        continue
                    p0, p1 = Path(cur_dir) / parts[0], Path(cur_dir) / parts[1]
                    if not (p0.exists() and p1.exists()):
                        p0, p1 = root_path / parts[0], root_path / parts[1]
                    if p0.exists() and p1.exists():
                        if any(k in parts[0].lower() for k in ("gt", "sharp", "target", "clean")):
                            sharp_p, blur_p = p0, p1
                        else:
                            blur_p, sharp_p = p0, p1
                        ch = _infer_pair_channel(str(blur_p), default_channel)
                        root_pairs.append((blur_p, sharp_p, ch))

        # 2. Discover sibling `(blur, sharp)` / `(motion_blurred, defocused_blurred, sharp)` / `(inputC, target)` folders
        for cur_dir, dirs, _files in os.walk(r_str, followlinks=True):
            dirs[:] = [d for d in dirs if not _is_excluded_pair_token(d, exclude_domain_subsets)]
            lower_map: Dict[str, str] = {d.lower(): d for d in dirs}
            blur_subs = [lower_map[k] for k in lower_map if k in _BLUR_DIR_NAMES]
            sharp_subs = [lower_map[k] for k in lower_map if k in _SHARP_DIR_NAMES]
            if not blur_subs or not sharp_subs:
                continue

            blur_subs.sort(key=lambda s: (0 if s.lower() != "blur_gamma" else 1, s))
            sharp_subs.sort(key=lambda s: (0 if s.lower() in ("sharp", "gt", "target", "clean") else 1, s))

            sharp_dir = Path(cur_dir) / sharp_subs[0]
            sharp_imgs = _collect_images(sharp_dir)
            if not sharp_imgs:
                continue
            sharp_index: Dict[str, Path] = {}
            for sp in sharp_imgs:
                rel = os.path.splitext(os.path.relpath(str(sp), str(sharp_dir)))[0]
                sharp_index[_normalize_pair_stem(rel)] = sp

            has_canonical_blur = "blur" in lower_map
            for b_sub in blur_subs:
                # Skip duplicate `blur_gamma` when canonical `blur` already exists in the same GoPro scene
                if b_sub.lower() == "blur_gamma" and has_canonical_blur:
                    continue
                b_dir = Path(cur_dir) / b_sub
                for bp in _collect_images(b_dir):
                    rel_b = os.path.splitext(os.path.relpath(str(bp), str(b_dir)))[0]
                    key_b = _normalize_pair_stem(rel_b)
                    if key_b in sharp_index:
                        ch = _infer_pair_channel(str(bp), default_channel)
                        root_pairs.append((bp, sharp_index[key_b], ch))

        # Filter train vs test/val PER top-level subdataset (so GoPro uses GoPro/train while HIDE — which only
        # has `HIDE/test/` in the Kaggle DBlur collection — is NOT discarded alongside GoPro/test!)
        if root_pairs:
            buckets: Dict[str, List[Tuple[Path, Path, str]]] = {}
            for rec in root_pairs:
                rel_parts = os.path.relpath(str(rec[0]), r_str).replace("\\", "/").split("/")
                top_key = rel_parts[0].lower() if len(rel_parts) > 1 else "."
                if top_key in ("train", "training", "test", "testing", "val", "valid", "validation"):
                    top_key = "."
                buckets.setdefault(top_key, []).append(rec)

            def _is_eval_pair(rec: Tuple[Path, Path, str]) -> bool:
                rel_tokens = set(os.path.relpath(str(rec[0]), r_str).replace("\\", "/").lower().split("/"))
                return bool(rel_tokens & {"test", "testing", "val", "valid", "validation"})

            for _b_key, sub_pairs in buckets.items():
                train_only = [rec for rec in sub_pairs if not _is_eval_pair(rec)]
                eval_only = [rec for rec in sub_pairs if _is_eval_pair(rec)]
                selected = (
                    (train_only if train_only else sub_pairs)
                    if is_train
                    else (eval_only if eval_only else sub_pairs)
                )
                for bp, sp, ch in selected:
                    sig = (str(bp), str(sp))
                    if sig not in seen:
                        seen.add(sig)
                        all_pairs.append((bp, sp, ch))

    return sorted(all_pairs, key=lambda x: (str(x[0]), str(x[1])))


def create_controllable_dataset(
    config, is_train=True, dataset_path=None, align_only: bool = False, compact_ipc: bool = False, verbose: bool = True
):
    """Discover clean 2K references + clean sharp targets from paired datasets + real paired camera blur."""
    split_name = "train" if is_train else "val"
    explicit_dir = dataset_path or (config.TRAIN_DATASET_PATH if is_train else config.VAL_DATASET_PATH)

    paired_records: List[Tuple[Path, Path, str]] = []
    if is_train:
        motion_root = getattr(config, "PAIRED_MOTION_DATASET_PATH", None)
        defocus_root = getattr(config, "PAIRED_DEFOCUS_DATASET_PATH", None)
        if motion_root:
            paired_records.extend(discover_paired_blur_datasets(motion_root, "deblur_motion", is_train=True))
        if defocus_root:
            paired_records.extend(discover_paired_blur_datasets(defocus_root, "sharpen_focus", is_train=True))

    paths: List[Path] = []
    source_desc = ""

    if explicit_dir:
        # Support comma-separated multiple clean/curated roots (e.g. `/kaggle/input/df2k,/kaggle/input/curated`)
        dir_parts = [p.strip() for p in str(explicit_dir).split(",") if p.strip()]
        auto_pairs: List[Tuple[Path, Path, str]] = []
        for d_str in dir_parts:
            if is_train:
                auto_pairs.extend(discover_paired_blur_datasets(d_str, "deblur_motion", is_train=True))
        if auto_pairs:
            existing_sigs = {(str(a), str(b)) for a, b, _ in paired_records}
            for rec in auto_pairs:
                if (str(rec[0]), str(rec[1])) not in existing_sigs:
                    paired_records.append(rec)

        blur_exclude = {str(bp) for bp, _sp, _ch in paired_records}
        if len(dir_parts) == 1:
            target_root = Path(dir_parts[0])
            div2k_sub = target_root / ("DIV2K_train_HR" if is_train else "DIV2K_valid_HR")
            all_paths = [
                p
                for p in _collect_images(target_root)
                if str(p) not in blur_exclude and not _is_degraded_or_excluded_for_clean_pool(p, target_root)
            ]
            if not all_paths and not paired_records:
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
            elif same_shared_root and all_paths:
                rng = random.Random(config.SEED)
                shuffled = list(all_paths)
                rng.shuffle(shuffled)
                split_idx = max(1, int(len(shuffled) * 0.9))
                paths = shuffled[:split_idx] if is_train else shuffled[split_idx:]
            else:
                paths = all_paths
            source_desc = str(target_root)
        else:
            collected = []
            for d_str in dir_parts:
                d_root = Path(d_str)
                collected.extend(
                    p
                    for p in _collect_images(d_root)
                    if str(p) not in blur_exclude and not _is_degraded_or_excluded_for_clean_pool(p, d_root)
                )
            paths = sorted(set(collected))
            if not paths and not paired_records:
                raise FileNotFoundError(f"No images found in {explicit_dir}.")
            source_desc = ", ".join(dir_parts)
    else:
        root = Path(config.DATASET_PATH)
        sub = root / ("DIV2K_train_HR" if is_train else "DIV2K_valid_HR")
        if sub.exists():
            paths = _collect_images(sub)
            source_desc = str(sub)
        elif root.exists():
            if verbose:
                print(f"No DIV2K folder found at {sub}, falling back to recursive image search in {root}.")
            all_paths = [
                p for p in _collect_images(root) if not _is_degraded_or_excluded_for_clean_pool(p, root)
            ]
            if all_paths:
                rng = random.Random(config.SEED)
                shuffled = list(all_paths)
                rng.shuffle(shuffled)
                split_idx = max(1, int(len(shuffled) * 0.9))
                paths = shuffled[:split_idx] if is_train else shuffled[split_idx:]
                source_desc = str(root)

        if not paths and paired_records:
            paths = sorted({sp for _bp, sp, _ch in paired_records})
            source_desc = "paired_sharp_targets"
        elif not paths:
            raise FileNotFoundError(
                f"No images found in {root}. Pass --train-root / --val-root or --paired-motion-root."
            )

    # Reuse all verified clean `sharp`/`gt` images from paired datasets inside `paths` so they ALSO train
    # all 6 main tasks (`denoise`, `deartifact`, `sharpen_focus`, `exposure`, `warmth`, `deblur_motion`)!
    added_paired_clean = 0
    if is_train and paired_records and bool(getattr(config, "REUSE_PAIRED_CLEAN_FOR_SYNTH", True)):
        existing_clean = {str(p) for p in paths}
        extra_clean = sorted(
            {
                sp
                for _bp, sp, _ch in paired_records
                if str(sp) not in existing_clean and not _is_excluded_pair_token(str(sp))
            },
            key=str,
        )
        if extra_clean:
            paths = list(paths) + extra_clean
            added_paired_clean = len(extra_clean)

    limit = config.TRAIN_IMAGES_COUNT if is_train else config.VAL_IMAGES_COUNT
    if limit and 0 < limit < len(paths):
        rng = random.Random(config.SEED + (0 if is_train else 1))
        shuffled = list(paths)
        rng.shuffle(shuffled)
        paths = sorted(shuffled[:limit])

    if verbose:
        clean_note = (
            f" (incl. +{added_paired_clean} clean sharp/GT images from paired datasets)"
            if added_paired_clean > 0
            else ""
        )
        pair_msg = ""
        if paired_records:
            n_mot = sum(1 for _b, _s, ch in paired_records if ch == "deblur_motion")
            n_foc = sum(1 for _b, _s, ch in paired_records if ch == "sharpen_focus")
            pair_msg = (
                f" + {len(paired_records)} real camera pairs "
                f"(motion={n_mot}, defocus={n_foc}, p_real={getattr(config, 'PAIRED_REAL_PROB', 0.42):.2f})"
            )
        print(f"[{split_name}] Using {len(paths)} clean reference images from {source_desc}{clean_note}{pair_msg}")
    return ControllableRestorationDataset(
        paths,
        config,
        is_train=is_train,
        align_only=align_only,
        compact_ipc=compact_ipc,
        paired_records=paired_records,
    )

