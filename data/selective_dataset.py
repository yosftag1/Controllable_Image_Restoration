"""Seed-synchronized selective, localized & directional degradation synthesis (CPU-accelerated)."""

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
    """Frozen random realizations for a single image patch so source and partial targets align."""

    height: int
    width: int
    noise_seed: int                 # Deterministic seed for lazy sensor noise realization
    motion_angle: float             # Degrees in [0, 180)
    focus_mode: int                 # 0 = optical Gaussian defocus, 1 = low-res bicubic, 2 = hybrid lens+resample
    artifact_mode: int              # 0 = JPEG compression, 1 = 8-bit color banding, 2 = JPEG + banding
    jpeg_quality: int               # Frozen base JPEG quality in [10, 32] for monotonic severity blending
    banding_levels: int             # Frozen base quantization levels in [7, 16]
    exposure_sign: float            # -1.0 (dark/underexposed) or +1.0 (bright/washed-out)
    warmth_sign: float              # -1.0 (cold blue cast) or +1.0 (warm amber/yellow cast)
    _noise_cache: Optional[np.ndarray] = field(default=None, repr=False)

    def get_noise_unit(self) -> np.ndarray:
        """Lazily materialize realistic sensor noise (80% RGB independent + 20% correlated luminance grain)."""
        if self._noise_cache is None:
            nrng = np.random.default_rng(self.noise_seed)
            rgb_n = nrng.standard_normal((self.height, self.width, 3), dtype=np.float32)
            luma_n = nrng.standard_normal((self.height, self.width, 1), dtype=np.float32)
            self._noise_cache = np.float32(0.82) * rgb_n + np.float32(0.35) * luma_n
        return self._noise_cache


class SelectiveDegradationPipeline:
    """Synthesizes (x_src, A_spatial, x_tgt) pairs for 4 unsigned fixes + 2 signed grades."""

    # Forward pipeline order designed for exact Lie-group radiometrics & well-posed selective residuals:
    # Optics/Resampling -> Motion -> Invertible Exposure & Warmth -> Compression/Banding -> Additive Grain
    PHYSICAL_ORDER = [
        "sharpen_focus",
        "deblur_motion",
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

        self.min_severity = float(getattr(config, "MIN_SEVERITY", 0.28))
        self.max_compound = int(getattr(config, "MAX_COMPOUND_DEGRADATIONS", 3))
        self.compound_prob = float(getattr(config, "COMPOUND_PROB", 0.72))
        self.spatial_mask_prob = float(getattr(config, "SPATIAL_MASK_PROB", 0.25))

        # Shared-memory scalar so persistent DataLoader workers immediately see per-epoch curriculum updates!
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
        min_q, max_q = self.config.JPEG_QUALITY_RANGE
        min_lvl, max_lvl = self.config.BANDING_LEVELS_RANGE
        return DegradationParams(
            height=height,
            width=width,
            noise_seed=int(rng.integers(0, 2**31 - 1)),
            motion_angle=float(rng.uniform(min_angle, max_angle)),
            focus_mode=int(rng.choice([0, 1, 2], p=[0.42, 0.36, 0.22])),
            artifact_mode=int(rng.choice([0, 1, 2], p=[0.50, 0.25, 0.25])),
            jpeg_quality=int(rng.integers(min_q, max(min_q + 1, (min_q + max_q) // 2))),
            banding_levels=int(rng.integers(min_lvl, max(min_lvl + 1, (min_lvl + max_lvl) // 2))),
            exposure_sign=-1.0 if rng.random() < 0.5 else 1.0,
            warmth_sign=-1.0 if rng.random() < 0.5 else 1.0,
        )

    def compute_target_state(self, s_src: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Map source state `s_src` + user control `action` to exact target state `s_tgt`.

        - Unsigned channels (`denoise`, `deartifact`, `deblur_motion`, `sharpen_focus` in [0, 1]):
          `s_tgt[k] = clip((1 - action[k]) * s_src[k], 0, 1)`
        - Signed Lie-group channels (`exposure`, `warmth` in [-1, 1]):
          `s_tgt[k] = clip(s_src[k] + action[k], -1.25, 1.25)`
        """
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
        """Return the user action vector that removes `fraction` (e.g. 50%) of every active degradation."""
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
    def _continuous_directional_motion_blur(img_float: np.ndarray, length_px: float, angle_deg: float) -> np.ndarray:
        """Subpixel-overlapping 1D shutter-trajectory blur (tap spacing <= 1.25px prevents ghost copies)."""
        if length_px <= 0.05:
            return img_float
        h, w = img_float.shape[:2]
        rad = np.deg2rad(angle_deg)
        dx = float(np.cos(rad)) * 0.5 * length_px
        dy = -float(np.sin(rad)) * 0.5 * length_px

        num_taps = max(5, min(21, int(np.ceil(length_px * 0.80)) | 1))
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

    def _apply_focus_softness(self, img_float: np.ndarray, severity: float, params: DegradationParams) -> np.ndarray:
        """Apply optical Gaussian defocus and/or bicubic downsample-upsample softness."""
        if severity <= 1e-4:
            return img_float
        sev = float(np.clip(severity, 0.0, 1.0))
        h, w = img_float.shape[:2]
        out = img_float

        if params.focus_mode in (0, 2):
            _, max_sigma = getattr(self.config, "DEFOCUS_SIGMA_RANGE", (0.8, 4.8))
            sig_scale = 1.0 if params.focus_mode == 0 else 0.55
            sigma = sev * float(max_sigma) * sig_scale
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
                out = np.clip(cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC), np.float32(0.0), np.float32(255.0))

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

    def _apply_deartifact_forward(self, img_float: np.ndarray, severity: float, params: DegradationParams) -> np.ndarray:
        """Apply compression/banding artifacts with monotonic linear blending for partial severities."""
        if severity <= 1e-4:
            return img_float
        sev = float(np.clip(severity, 0.0, 1.0))
        cur = img_float
        if params.artifact_mode in (1, 2):
            cur = self._render_base_banding(cur, params.banding_levels)
        if params.artifact_mode in (0, 2):
            cur = self._render_base_jpeg(cur, params.jpeg_quality)
        if sev >= 0.999:
            return cur
        return img_float * np.float32(1.0 - sev) + cur * np.float32(sev)

    def render_with_params(
        self, clean_rgb_u8: np.ndarray, state_vec: np.ndarray, params: DegradationParams
    ) -> np.ndarray:
        """Render an RGB uint8 image from `clean_rgb_u8` at signed/unsigned state `state_vec`."""
        cur = clean_rgb_u8.astype(np.float32)

        # 1. Optical defocus / resampling softness
        if "sharpen_focus" in self.type_to_idx:
            s_foc = float(state_vec[self.type_to_idx["sharpen_focus"]])
            if s_foc > 1e-4:
                cur = self._apply_focus_softness(cur, s_foc, params)

        # 2. Directional motion blur
        if "deblur_motion" in self.type_to_idx:
            s_mot = float(state_vec[self.type_to_idx["deblur_motion"]])
            if s_mot > 1e-4:
                _, max_k = self.config.MOTION_BLUR_KERNEL_RANGE
                length_px = float(np.clip(s_mot, 0.0, 1.0)) * (float(max_k) - 1.0)
                cur = self._continuous_directional_motion_blur(cur, length_px, params.motion_angle)

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
                cur = np.clip(cur + params.get_noise_unit() * noise_std, np.float32(0.0), np.float32(255.0))

        return np.clip(np.round(cur), np.float32(0.0), np.float32(255.0)).astype(np.uint8)

    def sample_source_severity(
        self, rng: np.random.Generator = None, params: Optional[DegradationParams] = None
    ) -> np.ndarray:
        """Sample 1 to 3 active degradations (unsigned in [min_sev, 1], signed in +/-[min_sev, 0.92])."""
        rng = rng or np.random.default_rng()
        eff_compound_prob = max(0.15, self.compound_prob * (1.0 - 0.70 * self.curriculum_single_boost))

        if self.max_compound > 1 and rng.random() < eff_compound_prob:
            k = 2 if (self.max_compound == 2 or rng.random() < 0.68) else min(3, self.max_compound)
        else:
            k = 1

        chosen = list(rng.choice(self.num_types, size=min(k, self.num_types), replace=False))
        chosen_names = [self.degradation_types[i] for i in chosen]

        # Avoid stacking both blurs simultaneously on the majority of samples so each blur kernel stays crisp
        if "deblur_motion" in chosen_names and "sharpen_focus" in chosen_names and len(chosen) > 1 and rng.random() < 0.75:
            drop_name = "deblur_motion" if rng.random() < 0.5 else "sharpen_focus"
            chosen.remove(self.type_to_idx[drop_name])

        damping = 1.0 if len(chosen) == 1 else (0.90 if len(chosen) == 2 else 0.80)
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
            # Keep CLI `action="all"` ([1, 1, 1, 1, 0, 0] + optional tone fix) well in-distribution!
            if rng.random() < 0.22:
                for idx in self.unsigned_indices:
                    action[idx] = 1.0

        elif u < p_sel + p_part + p_full + p_fp:
            if rng.random() < 0.50:
                # 50%: False-positive safety on clean objective sliders (must act as safe no-op without ringing)
                for idx in active_indices:
                    if rng.random() < 0.70:
                        action[idx] = full_act[idx]
                if inactive_unsigned:
                    n_fp = int(rng.integers(1, min(3, len(inactive_unsigned)) + 1))
                    for idx in rng.choice(inactive_unsigned, size=n_fp, replace=False):
                        action[idx] = float(rng.uniform(0.4, 1.0))
            else:
                # 50%: Pure artistic directional Exposure / Warmth grading (works on clean or partially fixed photos!)
                for idx in active_indices:
                    if not self.is_signed(idx) and rng.random() < 0.75:
                        action[idx] = 1.0
                for idx in self.signed_indices:
                    if rng.random() < 0.75:
                        art_delta = float(rng.uniform(0.25, 0.80)) * (-1.0 if rng.random() < 0.5 else 1.0)
                        # Keep net exposure/warmth within realistic [-1.0, +1.0] artistic envelope
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
                cy = int(rng.integers(gh // 5, 4 * gw // 5))
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
        s_src = self.sample_source_severity(rng=rng, params=params)
        x_src = self.render_with_params(clean_rgb_u8, s_src, params)

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
            x_tgt_fg = self.render_with_params(clean_rgb_u8, s_tgt_fg, params)

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
                x_tgt_bg = self.render_with_params(clean_rgb_u8, s_tgt_bg, params)

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
    """CPU-optimized PyTorch Dataset with staggered 2K-decode ring cache & compact IPC masks."""

    def __init__(self, image_paths, config, is_train=True, align_only: bool = False, compact_ipc: bool = False):
        self.image_paths = list(image_paths)
        self.config = config
        self.is_train = is_train
        self.align_only = align_only
        self.compact_ipc = compact_ipc
        self.patch_size = config.PATCH_SIZE
        self.pipeline = SelectiveDegradationPipeline(config)

        self._cache_capacity = int(getattr(config, "DECODE_CACHE_SIZE", 16)) if is_train else 0
        self._max_crops_per_decode = int(getattr(config, "CROPS_PER_DECODE", 6)) if is_train else 1
        self._decoded_ring = []

    def __len__(self):
        return len(self.image_paths)

    def set_curriculum_boost(self, boost: float) -> None:
        self.pipeline.curriculum_single_boost = float(np.clip(boost, 0.0, 1.0))

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


def create_controllable_dataset(
    config, is_train=True, dataset_path=None, align_only: bool = False, compact_ipc: bool = False, verbose: bool = True
):
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

    if verbose:
        print(f"[{split_name}] Using {len(paths)} clean reference images from {source_desc}")
    return ControllableRestorationDataset(
        paths, config, is_train=is_train, align_only=align_only, compact_ipc=compact_ipc
    )
