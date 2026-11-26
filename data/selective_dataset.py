"""Artifact-First, Dataset-Balanced Selective Restoration & Real-World SR Dataset (V7).

Key design principles:
1. Streamlined 7-control orthogonal schema:
   `denoise`, `deartifact`, `deblur_motion`, `super_resolution`, `dehaze`, `exposure`, `warmth`.
2. Balanced around ARTIFACTS first (`ARTIFACT_SAMPLING_WEIGHTS` prioritizing `deblur_motion`,
   `super_resolution`, `denoise`), and balanced across DATASETS inside each artifact:
   - Clean pool is grouped by dataset source (`DIV2K`, `Flickr2K`, `LSDIR`, `OST`, etc.) so 800
     razor-sharp DIV2K images aren't drowned out by 10k OST images, and soft 720p GoPro/HIDE/RealBlur
     frames are NEVER dumped into the clean pool (`REUSE_PAIRED_CLEAN_FOR_SYNTH = False`).
   - Within `deblur_motion`, real paired datasets are balanced per sub-dataset (`GoPro`, `HIDE`,
     `RealBlur`, `BlurDataset`) alongside crisp synthetic DIV2K/DF2K motion blur.
3. Single-artifact-first curriculum (~84% single-artifact early -> ~62% single / 38% compound),
   explicit `"all"` action vector coverage (`16%` on both synthetic and real pairs), and boosted
   `denoise + deblur_motion` co-occurrence with realistic demosaiced & noise-before-JPEG ISP noise.
4. Strict `gan_clean_eligible` flag per sample so Stage-2 DINOv2 Discriminator ONLY treats 100%
   restored (`s_tgt == 0`), crisp high-res clean photography as REAL.
"""

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
    noise_mode: int = 0              # 0=white/heteroscedastic, 1=demosaiced correlated ISP, 2=noise-before-JPEG
    heteroscedastic_noise: bool = False
    motion_angle: float = 0.0        # Primary trajectory angle in [0, 180)
    motion_curve_bend: float = 0.0   # Quadratic Bezier camera-shake curvature in [-max_bend, +max_bend]
    motion_dwell_end: float = 0.0    # Shutter turnaround endpoint dwell in [0, 0.35]
    motion_shake_jitter: float = 0.0 # S-curve / handheld tremor amplitude in [-0.35, +0.35]
    motion_pattern: int = 0          # 0 = Bezier arc, 1 = S-curve tremor, 2 = micro-shake jitter
    motion_spatial_mode: int = 0     # 0=uniform, 1=parallax, 2=moving FG, 3=smooth 3-basis field
    motion_hdr_boost: float = 0.0    # Linear-light specular highlight lift before blur
    motion_angle_delta: float = 0.0  # Spatial angle shift (deg) across smooth non-uniform blur field
    motion_bg_ratio: float = 0.10    # Relative blur length on sharp/minor layer in non-uniform modes
    motion_field_seed: int = 17      # Deterministic seed for spatial motion blur mask / depth field
    # Real-ESRGAN-style blind super-resolution / crisp detail parameters
    sr_mode: int = 0                 # 0=anisotropic blur+down-up, 1=2-stage blur+down-up+mild JPEG, 2=optical softness+mild down-up
    sr_blur_angle: float = 0.0       # Anisotropic optical PSF angle (deg)
    sr_blur_aniso: float = 1.0       # Aspect ratio between sigma_x and sigma_y in [0.55, 1.0]
    sr_down_interp: int = 0          # 0=INTER_AREA, 1=INTER_LINEAR, 2=INTER_CUBIC
    sr_up_interp: int = 0            # 0=INTER_CUBIC, 1=INTER_LINEAR
    sr_scale_factor: float = 0.35    # Base min scale in SR_DOWNSAMPLE_SCALE_RANGE at sev=1.0
    sr_sigma_max: float = 1.5        # Base max optical blur sigma in SR_BLUR_SIGMA_RANGE at sev=1.0
    # Compression & banding
    artifact_mode: int = 0           # 0=JPEG, 1=8-bit banding, 2=JPEG+banding, 3=4:2:0 chroma+ringing+JPEG
    jpeg_quality: int = 20           # Frozen base JPEG quality for monotonic severity blending
    banding_levels: int = 12         # Frozen base quantization levels
    # Atmospheric haze
    haze_seed: int = 11
    haze_airlight: Tuple[float, float, float] = (215.0, 224.0, 236.0)
    # Signed tone/WB signs
    exposure_sign: float = 1.0       # -1.0 (dark) or +1.0 (bright)
    warmth_sign: float = 1.0         # -1.0 (cool blue) or +1.0 (warm amber)
    _noise_cache: Optional[np.ndarray] = field(default=None, repr=False)
    _haze_depth_cache: Optional[np.ndarray] = field(default=None, repr=False)
    _motion_mask_cache: Optional[np.ndarray] = field(default=None, repr=False)

    def get_noise_unit(self) -> np.ndarray:
        """Lazily materialize realistic sensor noise (pixel-independent or demosaiced correlated ISP)."""
        if self._noise_cache is None:
            nrng = np.random.default_rng(self.noise_seed)
            rgb_n = nrng.standard_normal((self.height, self.width, 3), dtype=np.float32)
            luma_n = nrng.standard_normal((self.height, self.width, 1), dtype=np.float32)
            raw_n = np.float32(0.82) * rgb_n + np.float32(0.35) * luma_n
            if self.noise_mode == 1:
                # Spatially correlated Bayer-demosaiced camera ISP grain + coarser chroma blotches
                luma_corr = cv2.GaussianBlur(raw_n, (3, 3), sigmaX=0.65, sigmaY=0.65)
                ch_small = cv2.resize(
                    nrng.standard_normal((max(8, self.height // 2), max(8, self.width // 2), 3), dtype=np.float32),
                    (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR,
                )
                ch_zero_mean = ch_small - ch_small.mean(axis=2, keepdims=True)
                comb = np.float32(0.80) * luma_corr + np.float32(0.55) * ch_zero_mean
                # Re-normalize to ~unit std so severity scales consistently across noise modes
                std_est = max(1e-4, float(comb.std()))
                self._noise_cache = comb * np.float32(1.0 / std_est)
            else:
                self._noise_cache = raw_n
        return self._noise_cache


class SelectiveDegradationPipeline:
    """Synthesizes (x_src, A_spatial, x_tgt) pairs from crisp GT or Real Camera Deblur Pairs."""

    PHYSICAL_ORDER = [
        "dehaze",
        "super_resolution",
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

        self.min_severity = float(getattr(config, "MIN_SEVERITY", 0.25))
        self.max_compound = int(getattr(config, "MAX_COMPOUND_DEGRADATIONS", 3))
        self.compound_prob = float(getattr(config, "COMPOUND_PROB", 0.38))
        self.spatial_mask_prob = float(getattr(config, "SPATIAL_MASK_PROB", 0.20))

        self._curriculum_boost_val: float = 0.0
        self._curriculum_shared = (
            torch.zeros(1, dtype=torch.float32).share_memory_() if torch is not None else None
        )
        self._detail_focus_val: bool = False
        self._detail_focus_shared = (
            torch.zeros(1, dtype=torch.uint8).share_memory_() if torch is not None else None
        )

        raw_detail_types = getattr(
            config,
            "DETAIL_ACTION_TYPES",
            ["deblur_motion", "super_resolution", "deartifact", "dehaze"],
        )
        self.detail_indices = [self.type_to_idx[n] for n in raw_detail_types if n in self.type_to_idx]

        # Normalized Stage-1 & Stage-2 primary artifact budgets
        self.stage1_artifact_probs = self._build_artifact_probs(
            getattr(config, "ARTIFACT_SAMPLING_WEIGHTS", None)
        )
        self.stage2_artifact_probs = self._build_artifact_probs(
            getattr(config, "STAGE2_ARTIFACT_SAMPLING_WEIGHTS", None), fallback=self.stage1_artifact_probs
        )

        self.exp_kappa = float(getattr(config, "EXPOSURE_STRENGTH_KAPPA", 2.15))
        self.warm_r_kappa = float(getattr(config, "WARMTH_R_KAPPA", 1.15))
        self.warm_g_kappa = float(getattr(config, "WARMTH_G_KAPPA", 0.12))
        self.warm_b_kappa = float(getattr(config, "WARMTH_B_KAPPA", 1.30))

    def _build_artifact_probs(
        self, weights_map: Optional[Dict[str, float]], fallback: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if not weights_map:
            if fallback is not None:
                return fallback.copy()
            return np.full(self.num_types, 1.0 / max(1, self.num_types), dtype=np.float64)
        arr = np.array(
            [max(0.0, float(weights_map.get(name, 0.0))) for name in self.degradation_types],
            dtype=np.float64,
        )
        s = float(arr.sum())
        if s <= 1e-8:
            if fallback is not None:
                return fallback.copy()
            return np.full(self.num_types, 1.0 / max(1, self.num_types), dtype=np.float64)
        return arr / s

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

    @property
    def detail_stage_focus(self) -> bool:
        if self._detail_focus_shared is not None:
            return bool(int(self._detail_focus_shared[0].item()) != 0)
        return self._detail_focus_val

    @detail_stage_focus.setter
    def detail_stage_focus(self, val: bool) -> None:
        flag = bool(val)
        self._detail_focus_val = flag
        if self._detail_focus_shared is not None:
            self._detail_focus_shared[0] = 1 if flag else 0

    def is_signed(self, deg_name_or_idx) -> bool:
        if isinstance(deg_name_or_idx, int):
            return deg_name_or_idx in self.signed_indices
        return str(deg_name_or_idx) in self.signed_types

    def sample_primary_artifact(self, rng: np.random.Generator = None) -> str:
        """Sample the primary artifact strictly according to configured artifact budget."""
        rng = rng or np.random.default_rng()
        probs = self.stage2_artifact_probs if self.detail_stage_focus else self.stage1_artifact_probs
        idx = int(rng.choice(self.num_types, p=probs))
        return self.degradation_types[idx]

    def sample_params(self, height: int, width: int, rng: np.random.Generator = None) -> DegradationParams:
        """Sample all stochastic physical realizations upfront, independent of severity."""
        rng = rng or np.random.default_rng()
        min_angle, max_angle = self.config.MOTION_BLUR_ANGLE_RANGE
        max_bend = float(getattr(self.config, "MOTION_CURVE_MAX_BEND", 0.38))
        min_q, max_q = self.config.JPEG_QUALITY_RANGE
        min_lvl, max_lvl = self.config.BANDING_LEVELS_RANGE
        sr_s_lo, sr_s_hi = getattr(self.config, "SR_DOWNSAMPLE_SCALE_RANGE", (0.25, 0.65))
        sr_b_lo, sr_b_hi = getattr(self.config, "SR_BLUR_SIGMA_RANGE", (0.4, 2.2))

        air_base = float(rng.uniform(195.0, 238.0))
        air_col = (
            float(np.clip(air_base + rng.uniform(-10.0, 8.0), 175.0, 248.0)),
            float(np.clip(air_base + rng.uniform(-4.0, 8.0), 180.0, 248.0)),
            float(np.clip(air_base + rng.uniform(2.0, 18.0), 185.0, 250.0)),
        )
        nonunif_p = float(np.clip(getattr(self.config, "MOTION_NONUNIFORM_PROB", 0.25), 0.0, 0.90))
        if rng.random() < nonunif_p:
            m_spatial_mode = int(rng.choice([1, 2, 3], p=[0.38, 0.30, 0.32]))
        else:
            m_spatial_mode = 0
        hdr_p = float(np.clip(getattr(self.config, "MOTION_HDR_STREAK_PROB", 0.55), 0.0, 1.0))
        m_hdr_boost = float(rng.uniform(0.15, 0.40)) if rng.random() < hdr_p else 0.0
        m_angle_delta = float(rng.uniform(-28.0, 28.0))
        m_bg_ratio = 0.0 if rng.random() < 0.35 else float(rng.uniform(0.05, 0.24))

        return DegradationParams(
            height=height,
            width=width,
            noise_seed=int(rng.integers(0, 2**31 - 1)),
            noise_mode=int(rng.choice([0, 1, 2], p=[0.45, 0.30, 0.25])),
            heteroscedastic_noise=bool(rng.random() < 0.45),
            motion_angle=float(rng.uniform(min_angle, max_angle)),
            motion_curve_bend=float(rng.uniform(-max_bend, max_bend)),
            motion_dwell_end=float(rng.uniform(0.0, 0.35)),
            motion_shake_jitter=float(rng.uniform(-0.35, 0.35)),
            motion_pattern=int(rng.choice([0, 1, 2], p=[0.45, 0.35, 0.20])),
            motion_spatial_mode=m_spatial_mode,
            motion_hdr_boost=m_hdr_boost,
            motion_angle_delta=m_angle_delta,
            motion_bg_ratio=m_bg_ratio,
            motion_field_seed=int(rng.integers(0, 2**31 - 1)),
            sr_mode=int(rng.choice([0, 1, 2], p=[0.50, 0.30, 0.20])),
            sr_blur_angle=float(rng.uniform(0.0, 180.0)),
            sr_blur_aniso=float(rng.uniform(0.55, 1.0)),
            sr_down_interp=int(rng.choice([0, 1, 2], p=[0.45, 0.30, 0.25])),
            sr_up_interp=int(rng.choice([0, 1], p=[0.65, 0.35])),
            sr_scale_factor=float(rng.uniform(sr_s_lo, sr_s_hi)),
            sr_sigma_max=float(rng.uniform(sr_b_lo, sr_b_hi)),
            artifact_mode=int(rng.choice([0, 1, 2, 3], p=[0.40, 0.18, 0.20, 0.22])),
            jpeg_quality=int(rng.integers(min_q, max(min_q + 1, (min_q + max_q) // 2))),
            banding_levels=int(rng.integers(min_lvl, max(min_lvl + 1, (min_lvl + max_lvl) // 2))),
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
        """Return exact user action vector restoring `s_src` completely back to clean `0.0`."""
        act = np.zeros(self.num_types, dtype=np.float32)
        for k in self.unsigned_indices:
            if float(s_src[k]) > 1e-4:
                act[k] = 1.0
        for k in self.signed_indices:
            if abs(float(s_src[k])) > 1e-4:
                act[k] = float(np.clip(-float(s_src[k]), -1.0, 1.0))
        return act

    def compute_partial_restore_action(self, s_src: np.ndarray, fraction: float = 0.5) -> np.ndarray:
        """Return user action vector removing `fraction` of every active degradation."""
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

        num_taps = max(9, min(75, int(np.ceil(length_px * 1.6)) | 1))
        t_vals = np.linspace(-1.0, 1.0, num_taps, dtype=np.float32)
        weights = np.float32(1.0) + np.float32(dwell_end) * (t_vals * t_vals)
        weights /= np.sum(weights)

        quad_zero_mean = (t_vals * t_vals) - np.float32(1.0 / 3.0)
        dx_arr = t_vals * np.float32(tx) + np.float32(curve_bend) * quad_zero_mean * np.float32(nx)
        dy_arr = t_vals * np.float32(ty) + np.float32(curve_bend) * quad_zero_mean * np.float32(ny)

        if pattern == 1 and abs(shake_jitter) > 1e-4:
            s_wave = np.sin(np.float32(np.pi) * t_vals) * np.float32(shake_jitter)
            dx_arr = dx_arr + s_wave * np.float32(nx)
            dy_arr = dy_arr + s_wave * np.float32(ny)
        elif pattern == 2 and abs(shake_jitter) > 1e-4:
            wobble_n = np.sin(np.float32(2.5 * np.pi) * t_vals) * np.float32(0.75 * shake_jitter)
            wobble_t = np.cos(np.float32(2.0 * np.pi) * t_vals) * np.float32(0.18 * abs(shake_jitter))
            dx_arr = dx_arr + wobble_n * np.float32(nx) + wobble_t * np.float32(tx)
            dy_arr = dy_arr + wobble_n * np.float32(ny) + wobble_t * np.float32(ty)

        max_reach = int(np.ceil(max(float(np.max(np.abs(dx_arr))), float(np.max(np.abs(dy_arr)))) + 1.5))
        half_k = max(1, max_reach)
        k_size = 2 * half_k + 1
        psf = np.zeros((k_size, k_size), dtype=np.float32)

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

    @staticmethod
    def _build_motion_spatial_field(
        h: int,
        w: int,
        mode: int,
        seed: int,
        ref_lin_rgb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Deterministic smooth [H, W, 1] spatial field in [0, 1] for non-uniform motion blur."""
        mrng = np.random.default_rng(seed)
        gh, gw = max(16, h // 4), max(16, w // 4)

        if mode in (1, 2):
            sub = float(mrng.random())
            if sub < 0.40 and ref_lin_rgb is not None:
                srgb_small = np.clip(
                    np.sqrt(np.clip(ref_lin_rgb * np.float32(1.0 / 255.0), 0.0, 1.0)) * np.float32(255.0),
                    0.0,
                    255.0,
                ).astype(np.uint8)
                small_rgb = cv2.resize(srgb_small, (gw, gh), interpolation=cv2.INTER_AREA)
                lab = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
                cy = int(mrng.integers(gh // 5, 4 * gh // 5))
                cx = int(mrng.integers(gw // 5, 4 * gw // 5))
                diff = lab - lab[cy, cx].reshape(1, 1, 3)
                dist_sq = np.sum(diff * diff, axis=-1)
                thresh = float(np.percentile(dist_sq, mrng.uniform(28.0, 58.0)))
                raw_small = (dist_sq <= thresh).astype(np.float32)
                if float(raw_small.mean()) < 0.15 or float(raw_small.mean()) > 0.85:
                    sub = 0.55
                else:
                    raw_mask = cv2.resize(raw_small, (w, h), interpolation=cv2.INTER_LINEAR)
                    sigma = float(mrng.uniform(6.0, 14.0))
                    field = cv2.GaussianBlur(raw_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
            if sub >= 0.40 or ref_lin_rgb is None:
                if sub < 0.78:
                    mask_small = np.zeros((gh, gw), dtype=np.float32)
                    cx = int(mrng.integers(gw // 4, 3 * gw // 4))
                    cy = int(mrng.integers(gh // 4, 3 * gh // 4))
                    rx = int(mrng.integers(max(4, gw // 5), max(6, int(gw * 0.42))))
                    ry = int(mrng.integers(max(4, gh // 5), max(6, int(gh * 0.42))))
                    ang = float(mrng.uniform(0.0, 180.0))
                    cv2.ellipse(mask_small, (cx, cy), (rx, ry), ang, 0, 360, 1.0, -1)
                    raw_mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
                    sigma = float(mrng.uniform(8.0, 18.0))
                    field = cv2.GaussianBlur(raw_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
                else:
                    yy, xx = np.meshgrid(
                        np.linspace(-1.0, 1.0, gh, dtype=np.float32),
                        np.linspace(-1.0, 1.0, gw, dtype=np.float32),
                        indexing="ij",
                    )
                    theta = float(mrng.uniform(0.0, 2.0 * np.pi))
                    proj = np.float32(np.cos(theta)) * xx + np.float32(np.sin(theta)) * yy
                    offset = np.float32(mrng.uniform(-0.25, 0.25))
                    sharpness = np.float32(mrng.uniform(4.5, 9.0))
                    field_small = np.float32(1.0) / (np.float32(1.0) + np.exp(-sharpness * (proj - offset)))
                    field = cv2.resize(field_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            yy, xx = np.meshgrid(
                np.linspace(0.0, 1.0, gh, dtype=np.float32),
                np.linspace(0.0, 1.0, gw, dtype=np.float32),
                indexing="ij",
            )
            theta = float(mrng.uniform(0.0, 2.0 * np.pi))
            lin_ramp = 0.5 + 0.5 * (
                np.float32(np.cos(theta)) * (xx * 2.0 - 1.0) + np.float32(np.sin(theta)) * (yy * 2.0 - 1.0)
            )
            cx = float(mrng.uniform(0.20, 0.80))
            cy = float(mrng.uniform(0.20, 0.80))
            rad_ramp = np.sqrt((xx - np.float32(cx)) ** 2 + (yy - np.float32(cy)) ** 2) * np.float32(1.35)
            mix = float(mrng.uniform(0.30, 0.75))
            raw_field = np.clip(mix * lin_ramp + (1.0 - mix) * rad_ramp, 0.0, 1.0)
            if mrng.random() < 0.5:
                raw_field = 1.0 - raw_field
            f_min, f_max = float(raw_field.min()), float(raw_field.max())
            if f_max - f_min > 1e-4:
                raw_field = (raw_field - f_min) / (f_max - f_min)
            field = cv2.resize(raw_field.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

        f_max = float(field.max())
        if f_max > 1e-5:
            field = field * np.float32(1.0 / f_max)
        return np.clip(field, np.float32(0.0), np.float32(1.0))[:, :, None]

    def _apply_motion_blur_field(
        self, lin_rgb: np.ndarray, severity: float, params: DegradationParams
    ) -> np.ndarray:
        """Apply curved/tremor & spatially-varying motion blur scaled cleanly to patch size."""
        if severity <= 1e-4:
            return lin_rgb
        sev = float(np.clip(severity, 0.0, 1.0))
        _, max_k = self.config.MOTION_BLUR_KERNEL_RANGE
        # Keep kernel reach <= 13% of patch side so boundary reflection never dominates small debug/256 crops
        h, w = lin_rgb.shape[:2]
        eff_max_k = min(float(max_k), max(11.0, float(min(h, w)) * 0.13))
        length_px = sev * (eff_max_k - 1.0)
        if length_px <= 0.05:
            return lin_rgb

        hdr_s = float(params.motion_hdr_boost) * min(1.0, length_px / 14.0)
        if hdr_s > 1e-4:
            x_lin = np.clip(lin_rgb * np.float32(1.0 / 255.0), np.float32(0.0), np.float32(1.0))
            work = lin_rgb * (np.float32(1.0) + np.float32(hdr_s * 2.5) * (x_lin * x_lin * x_lin * x_lin))
        else:
            work = lin_rgb

        def _blur_with(l_px: float, ang_deg: float, bend_scale: float = 1.0) -> np.ndarray:
            if l_px <= 0.15:
                return work
            return self._continuous_curved_motion_blur(
                work,
                length_px=l_px,
                angle_deg=ang_deg,
                curve_bend=params.motion_curve_bend * bend_scale,
                dwell_end=params.motion_dwell_end,
                shake_jitter=params.motion_shake_jitter * bend_scale,
                pattern=params.motion_pattern,
            )

        mode = int(params.motion_spatial_mode) % 4
        if mode == 0:
            blurred = _blur_with(length_px, params.motion_angle, 1.0)
        else:
            if params._motion_mask_cache is None or params._motion_mask_cache.shape[:2] != (h, w):
                params._motion_mask_cache = self._build_motion_spatial_field(
                    h, w, mode=mode, seed=params.motion_field_seed, ref_lin_rgb=lin_rgb
                )
            m_field = params._motion_mask_cache

            if mode == 1:
                bg_b = _blur_with(length_px, params.motion_angle, 1.0)
                fg_b = _blur_with(length_px * float(params.motion_bg_ratio), params.motion_angle, 0.4)
                blurred = m_field * fg_b + (np.float32(1.0) - m_field) * bg_b
            elif mode == 2:
                fg_b = _blur_with(length_px, params.motion_angle, 1.0)
                bg_b = _blur_with(length_px * float(params.motion_bg_ratio), params.motion_angle, 0.3)
                blurred = m_field * fg_b + (np.float32(1.0) - m_field) * bg_b
            else:
                d_ang = float(params.motion_angle_delta)
                b0 = _blur_with(length_px * 0.12, params.motion_angle, 0.35)
                b1 = _blur_with(length_px * 0.55, params.motion_angle + 0.5 * d_ang, 0.70)
                b2 = _blur_with(length_px, params.motion_angle + d_ang, 1.0)
                w0 = np.clip(np.float32(1.0) - np.float32(2.0) * m_field, np.float32(0.0), np.float32(1.0))
                w2 = np.clip(np.float32(2.0) * m_field - np.float32(1.0), np.float32(0.0), np.float32(1.0))
                w1 = np.float32(1.0) - w0 - w2
                blurred = w0 * b0 + w1 * b1 + w2 * b2

        if hdr_s > 1e-4:
            xb = np.maximum(blurred * np.float32(1.0 / 255.0), np.float32(0.0))
            mapped = xb / (np.float32(1.0) + np.float32(0.14) * np.maximum(xb - np.float32(0.85), np.float32(0.0)))
            return np.clip(mapped * np.float32(255.0), np.float32(0.0), np.float32(255.0))
        return np.clip(blurred, np.float32(0.0), np.float32(255.0))

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
    def _build_anisotropic_gaussian_kernel(sigma_x: float, sigma_y: float, angle_deg: float) -> np.ndarray:
        """Real-ESRGAN anisotropic 2D Gaussian optical blur kernel."""
        sx = max(0.15, float(sigma_x))
        sy = max(0.15, float(sigma_y))
        rad_k = max(1, min(9, int(np.ceil(max(sx, sy) * 2.5))))
        coords = np.arange(-rad_k, rad_k + 1, dtype=np.float32)
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        th = float(np.deg2rad(angle_deg))
        c, s = float(np.cos(th)), float(np.sin(th))
        xr = c * xx + s * yy
        yr = -s * xx + c * yy
        kern = np.exp(-0.5 * ((xr * xr) / (sx * sx) + (yr * yr) / (sy * sy))).astype(np.float32)
        kern /= max(1e-6, float(kern.sum()))
        return kern

    def _apply_sr_degradation(self, img_float: np.ndarray, severity: float, params: DegradationParams) -> np.ndarray:
        """Real-ESRGAN-style blind super-resolution & optical softness degradation (strictly monotonic)."""
        if severity <= 1e-4:
            return img_float
        sev = float(np.clip(severity, 0.0, 1.0))
        h, w = img_float.shape[:2]
        out = img_float

        # 1. Optical anisotropic Gaussian lens softness
        blur_mul = 0.85 if params.sr_mode == 0 else (1.0 if params.sr_mode == 1 else 1.25)
        sig_x = sev * float(params.sr_sigma_max) * blur_mul
        sig_y = sig_x * float(params.sr_blur_aniso)
        if sig_x >= 0.20:
            kern = self._build_anisotropic_gaussian_kernel(sig_x, sig_y, params.sr_blur_angle)
            out = cv2.filter2D(out, cv2.CV_32F, kern, borderType=cv2.BORDER_REFLECT_101)

        # 2. Resolution loss: multi-kernel downsample -> bicubic/bilinear upsample back to (W, H)
        down_flags = (cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC)
        up_flags = (cv2.INTER_CUBIC, cv2.INTER_LINEAR)
        down_flag = down_flags[int(params.sr_down_interp) % len(down_flags)]
        up_flag = up_flags[int(params.sr_up_interp) % len(up_flags)]

        base_min_s = float(params.sr_scale_factor)
        if params.sr_mode == 2:
            # Moderate resolution loss + optical softness (trains crisp detail boost on real soft photos!)
            base_min_s = max(0.45, base_min_s + 0.15)
        scale = 1.0 - sev * (1.0 - base_min_s)
        if scale < 0.985:
            new_w = max(12, int(round(w * scale)))
            new_h = max(12, int(round(h * scale)))
            down = cv2.resize(out, (new_w, new_h), interpolation=down_flag)
            out = np.clip(
                cv2.resize(down, (w, h), interpolation=up_flag),
                np.float32(0.0),
                np.float32(255.0),
            )

        # 3. Mode 1: realistic 2nd-order mild web JPEG softness (continuous Q in [92 -> 56])
        if params.sr_mode == 1:
            q_mild = int(round(92.0 - sev * 36.0))
            out = self._render_base_jpeg(out, q_mild)

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
        del refl_donor_u8
        cur = clean_rgb_u8.astype(np.float32)

        # 1. Atmospheric haze
        if "dehaze" in self.type_to_idx:
            s_haze = float(state_vec[self.type_to_idx["dehaze"]])
            if s_haze > 1e-4:
                cur = self._apply_haze(cur, s_haze, params)

        # 2. Linear-light motion blur (physical shutter integration)
        s_mot = float(state_vec[self.type_to_idx["deblur_motion"]]) if "deblur_motion" in self.type_to_idx else 0.0
        if s_mot > 1e-4:
            inv255 = np.float32(1.0 / 255.0)
            lin = np.square(np.clip(cur * inv255, np.float32(0.0), np.float32(1.0))) * np.float32(255.0)
            lin = self._apply_motion_blur_field(lin, s_mot, params)
            cur = np.sqrt(np.clip(lin * inv255, np.float32(0.0), np.float32(1.0))) * np.float32(255.0)

        # 3. Super-resolution / Real-ESRGAN optical blur + resolution degradation
        sr_key = "super_resolution" if "super_resolution" in self.type_to_idx else "sharpen_focus"
        if sr_key in self.type_to_idx:
            s_sr = float(state_vec[self.type_to_idx[sr_key]])
            if s_sr > 1e-4:
                cur = self._apply_sr_degradation(cur, s_sr, params)

        # 4. Exact Commutative Lie-Group Exposure + Warmth in float32 [0, 1]
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

        # 6. Realistic camera ISP / sensor grain (white, demosaiced correlated, or noise-before-JPEG)
        if "denoise" in self.type_to_idx:
            s_noi = float(state_vec[self.type_to_idx["denoise"]])
            if s_noi > 1e-4:
                sev_n = float(np.clip(s_noi, 0.0, 1.0))
                _, max_std = self.config.GAUSSIAN_NOISE_STD_RANGE
                noise_std = np.float32(sev_n * float(max_std))
                if params.heteroscedastic_noise:
                    lum_mod = np.float32(0.70) + np.float32(0.60) * np.sqrt(
                        np.clip(cur * np.float32(1.0 / 255.0), np.float32(0.0), np.float32(1.0))
                    )
                    noise_delta = params.get_noise_unit() * (noise_std * lum_mod)
                else:
                    noise_delta = params.get_noise_unit() * noise_std
                cur = np.clip(cur + noise_delta, np.float32(0.0), np.float32(255.0))
                if params.noise_mode == 2:
                    # Realistic camera / web JPEG saved AFTER sensor noise (creates compressed chroma blotches!)
                    q_post = int(round(92.0 - sev_n * 42.0))
                    cur = self._render_base_jpeg(cur, q_post)

        return np.clip(np.round(cur), np.float32(0.0), np.float32(255.0)).astype(np.uint8)

    def sample_source_severity(
        self,
        rng: np.random.Generator = None,
        params: Optional[DegradationParams] = None,
        primary_artifact: Optional[str] = None,
    ) -> np.ndarray:
        """Artifact-first severity sampling with single-artifact-first curriculum & realistic compounds."""
        rng = rng or np.random.default_rng()
        if primary_artifact is None or primary_artifact not in self.type_to_idx:
            primary_artifact = self.sample_primary_artifact(rng=rng)
        p_idx = self.type_to_idx[primary_artifact]

        # Strong single-artifact focus during curriculum warmup (~84% single -> ~62% single steady state)
        eff_compound_prob = max(0.12, self.compound_prob * (1.0 - 0.72 * self.curriculum_single_boost))
        if self.detail_stage_focus:
            # Stage 2 focuses even more cleanly on single/2-task crisp detail targets
            eff_compound_prob = min(eff_compound_prob, 0.30)

        chosen = [p_idx]
        if self.max_compound > 1 and rng.random() < eff_compound_prob:
            k = 3 if (self.max_compound >= 3 and rng.random() < 0.25) else 2
            # Explicitly boost realistic noisy motion blur (`deblur_motion + denoise`)!
            mot_noi_p = float(getattr(self.config, "MOTION_NOISE_COMPOUND_PROB", 0.45))
            if (
                primary_artifact == "deblur_motion"
                and "denoise" in self.type_to_idx
                and rng.random() < mot_noi_p
            ):
                chosen.append(self.type_to_idx["denoise"])

            if len(chosen) < k:
                rem = [i for i in range(self.num_types) if i not in chosen]
                base_probs = self.stage1_artifact_probs.copy()
                # Keep `deblur_motion` and `super_resolution` mostly disentangled so each expert stays sharp
                if primary_artifact == "deblur_motion" and "super_resolution" in self.type_to_idx:
                    base_probs[self.type_to_idx["super_resolution"]] *= 0.25
                elif primary_artifact == "super_resolution" and "deblur_motion" in self.type_to_idx:
                    base_probs[self.type_to_idx["deblur_motion"]] *= 0.25
                rem_w = np.array([max(1e-4, base_probs[i]) for i in rem], dtype=np.float64)
                rem_w /= rem_w.sum()
                need = min(k - len(chosen), len(rem))
                chosen.extend(list(rng.choice(rem, size=need, replace=False, p=rem_w)))

        damping = 1.0 if len(chosen) == 1 else (0.92 if len(chosen) == 2 else 0.84)
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
        """Sample controllable user action vectors with explicit `"all"` blind-mode coverage."""
        rng = rng or np.random.default_rng()
        active_indices = [i for i in range(self.num_types) if abs(float(s_src[i])) > 1e-4]
        inactive_unsigned = [i for i in self.unsigned_indices if abs(float(s_src[i])) <= 1e-4]

        if self.detail_stage_focus:
            # Stage 2: ~82% full restoration (s_tgt == 0 -> maximizes clean DINOv2 discriminator signal!)
            p_full = 0.62
            p_all = 0.20
            p_sel = 0.10
            p_part = 0.04
            p_fp = 0.02
        else:
            p_sel = float(getattr(self.config, "ACTION_DIST_SELECTIVE", 0.16)) * (
                1.0 - 0.50 * self.curriculum_single_boost
            )
            p_part = float(getattr(self.config, "ACTION_DIST_PARTIAL", 0.10))
            p_all = float(getattr(self.config, "ACTION_DIST_BLIND_ALL", 0.16))
            p_full = float(getattr(self.config, "ACTION_DIST_FULL", 0.50)) + (
                float(getattr(self.config, "ACTION_DIST_SELECTIVE", 0.16)) - p_sel
            )
            p_fp = float(getattr(self.config, "ACTION_DIST_FALSE_POS", 0.04))

        u = float(rng.random())
        action = np.zeros(self.num_types, dtype=np.float32)
        full_act = self.compute_full_restore_action(s_src)

        if u < p_full:
            # Exact targeted restoration of the active degradations
            action = full_act.copy()

        elif u < p_full + p_all:
            # Blind "Restore All" preset (`[1, 1, 1, 1, 1, 0, 0]` + any active signed inverse) — matches app.py!
            action = full_act.copy()
            for idx in self.unsigned_indices:
                action[idx] = 1.0

        elif u < p_full + p_all + p_sel and len(active_indices) >= 2:
            # Selective: fix a subset of degradations, keep the others untouched
            num_fix = int(rng.integers(1, len(active_indices)))
            fix_ids = rng.choice(active_indices, size=num_fix, replace=False)
            for idx in fix_ids:
                frac = 1.0 if rng.random() < 0.80 else float(rng.uniform(0.5, 1.0))
                action[idx] = full_act[idx] * frac

        elif u < p_full + p_all + p_sel + p_part:
            for idx in active_indices:
                if rng.random() < 0.85:
                    frac = float(rng.uniform(0.30, 0.85))
                    action[idx] = full_act[idx] * frac

        elif u < p_full + p_all + p_sel + p_part + p_fp:
            if rng.random() < 0.50:
                for idx in active_indices:
                    if rng.random() < 0.75:
                        action[idx] = full_act[idx]
                if inactive_unsigned:
                    n_fp = int(rng.integers(1, min(2, len(inactive_unsigned)) + 1))
                    for idx in rng.choice(inactive_unsigned, size=n_fp, replace=False):
                        action[idx] = float(rng.uniform(0.45, 1.0))
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

        # In Stage 2, keep >=1 active detail slider ON (except 4% identity safety)
        if self.detail_stage_focus and self.detail_indices and rng.random() < 0.96:
            active_detail = [i for i in active_indices if i in self.detail_indices]
            if active_detail and all(abs(float(action[i])) < 0.25 for i in active_detail):
                pick = int(rng.choice(active_detail))
                action[pick] = float(full_act[pick])

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
        primary_artifact: Optional[str] = None,
        refl_donor_u8: Optional[np.ndarray] = None,
    ):
        """Fast synthesis returning `(x_src, x_tgt, a_fg, a_bg, mask_2d, s_src, gan_clean_eligible)`."""
        del refl_donor_u8
        rng = rng or np.random.default_rng()
        h, w = clean_rgb_u8.shape[:2]
        params = self.sample_params(h, w, rng=rng)
        s_src = self.sample_source_severity(rng=rng, params=params, primary_artifact=primary_artifact)
        x_src = self.render_with_params(clean_rgb_u8, s_src, params)

        if align_only:
            zeros_a = np.zeros(self.num_types, dtype=np.float32)
            ones_mask = np.ones((h, w), dtype=np.float32)
            return x_src, x_src, zeros_a, zeros_a, ones_mask, s_src, False

        a_fg = self.sample_action_vector(s_src, rng=rng)
        s_tgt_fg = self.compute_target_state(s_src, a_fg)
        fg_is_clean = bool(np.allclose(s_tgt_fg, 0.0, atol=1e-4))
        if np.allclose(s_tgt_fg, s_src, atol=1e-4):
            x_tgt_fg = x_src.copy()
        elif fg_is_clean:
            x_tgt_fg = clean_rgb_u8.copy()
        else:
            x_tgt_fg = self.render_with_params(clean_rgb_u8, s_tgt_fg, params)

        # Keep spatial mask rate low in Stage 2 so GAN-eligible full-clean targets stay abundant
        base_mask_p = 0.08 if self.detail_stage_focus else self.spatial_mask_prob
        eff_mask_prob = base_mask_p * (1.0 - 0.65 * self.curriculum_single_boost)
        if rng.random() < eff_mask_prob:
            if rng.random() < 0.68:
                a_bg = np.zeros(self.num_types, dtype=np.float32)
            else:
                a_bg = self.sample_action_vector(s_src, rng=rng)

            s_tgt_bg = self.compute_target_state(s_src, a_bg)
            bg_is_clean = bool(np.allclose(s_tgt_bg, 0.0, atol=1e-4))
            if np.allclose(s_tgt_bg, s_src, atol=1e-4):
                x_tgt_bg = x_src
            elif bg_is_clean:
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
            full_clean = fg_is_clean and bg_is_clean
        else:
            a_bg = a_fg
            mask = np.ones((h, w), dtype=np.float32)
            x_tgt = x_tgt_fg
            full_clean = fg_is_clean

        has_detail_act = any(
            abs(float(a_fg[i])) >= 0.25 or abs(float(a_bg[i])) >= 0.25 for i in self.detail_indices
        )
        gan_clean_eligible = bool(full_clean and has_detail_act)
        return x_src, x_tgt, a_fg, a_bg, mask, s_src, gan_clean_eligible

    def synthesize_real_pair_components(
        self,
        blur_rgb_u8: np.ndarray,
        sharp_rgb_u8: np.ndarray,
        blur_channel_name: str = "deblur_motion",
        rng: np.random.Generator = None,
        align_only: bool = False,
    ):
        """Compositional Real-Pair Wrapper for GoPro, HIDE, RealBlur & BlurDataset!

        - Respects single-to-compound curriculum (`MOTION_NOISE_COMPOUND_PROB` pairs real blur with
          realistic camera ISP noise!).
        - Explicitly trains `"all"` (`[1, 1, 1, 1, 1, 0, 0]`) ~18% of the time so Gradio's default
          preset is 100% in-distribution on real camera shake!
        - NEVER runs false-positive identity training on soft 720p GoPro/RealBlur frames!
        - Returns `gan_clean_eligible = False` so 720p/night video targets never lower Stage-2 GAN crispness!
        """
        rng = rng or np.random.default_rng()
        h, w = sharp_rgb_u8.shape[:2]
        params = self.sample_params(h, w, rng=rng)

        blur_idx = self.type_to_idx.get(blur_channel_name, self.type_to_idx.get("deblur_motion", 0))
        downstream_candidates = [
            i
            for i, name in enumerate(self.degradation_types)
            if i != blur_idx and name not in ("deblur_motion", "super_resolution")
        ]

        eff_compound_prob = max(0.12, self.compound_prob * (1.0 - 0.72 * self.curriculum_single_boost))
        if self.detail_stage_focus:
            eff_compound_prob = min(eff_compound_prob, 0.28)

        pres_cutoff = 0.05 if self.detail_stage_focus else 0.12
        u_mode = float(rng.random())
        need_downstream = u_mode < pres_cutoff

        s_down = np.zeros(self.num_types, dtype=np.float32)
        if downstream_candidates and (need_downstream or rng.random() < eff_compound_prob):
            chosen_down: List[int] = []
            mot_noi_p = float(getattr(self.config, "MOTION_NOISE_COMPOUND_PROB", 0.45))
            if "denoise" in self.type_to_idx and self.type_to_idx["denoise"] in downstream_candidates:
                if rng.random() < mot_noi_p:
                    chosen_down.append(self.type_to_idx["denoise"])
            n_target = 1 if rng.random() < 0.78 else min(2, len(downstream_candidates))
            if len(chosen_down) < n_target:
                rem = [i for i in downstream_candidates if i not in chosen_down]
                chosen_down.extend(list(rng.choice(rem, size=min(n_target - len(chosen_down), len(rem)), replace=False)))
            for idx in chosen_down:
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
            return x_src, sharp_rgb_u8, zeros_a, zeros_a, ones_mask, s_src, False

        if u_mode < pres_cutoff:
            # Selective real-blur preservation (`a_blur = 0.0`: clean noise/tone, KEEP real motion blur!)
            base_src = blur_rgb_u8
            base_tgt_fg = blur_rgb_u8
            s_src = s_down.copy()
            s_src[blur_idx] = 1.0
            a_fg = self.compute_full_restore_action(s_down)
            a_fg[blur_idx] = 0.0
            s_tgt_down = self.compute_target_state(s_down, a_fg)
        else:
            # Active Real Camera Deblurring (`a_blur = 1.0`, ~88% Stage 1 / ~95% Stage 2)
            base_src = blur_rgb_u8
            base_tgt_fg = sharp_rgb_u8
            s_src = s_down.copy()
            s_src[blur_idx] = 1.0
            a_fg = np.zeros(self.num_types, dtype=np.float32)
            a_fg[blur_idx] = 1.0
            full_down = self.compute_full_restore_action(s_down)
            for idx in downstream_candidates:
                if abs(float(s_down[idx])) > 1e-4:
                    if rng.random() < 0.80:
                        a_fg[idx] = full_down[idx]
            # Explicitly train blind `"all"` preset (`[1, 1, 1, 1, 1, 0, 0]`) ~18% of the time on real pairs!
            if rng.random() < 0.18:
                for idx in self.unsigned_indices:
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

        base_mask_p = 0.06 if self.detail_stage_focus else (0.55 * self.spatial_mask_prob)
        eff_mask_prob = base_mask_p * (1.0 - 0.65 * self.curriculum_single_boost)
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

        return x_src, x_tgt, a_fg, a_bg, mask, s_src, False

    def synthesize_pair(self, clean_rgb_u8: np.ndarray, rng: np.random.Generator = None):
        """Synthesize seed-synchronized (x_src, action_map [K, H, W], x_tgt, s_src)."""
        x_src, x_tgt, a_fg, a_bg, mask, s_src, _ = self.synthesize_components(
            clean_rgb_u8, rng=rng, align_only=False
        )
        h, w = clean_rgb_u8.shape[:2]
        if a_bg is a_fg or np.array_equal(a_fg, a_bg):
            action_map = np.broadcast_to(a_fg[:, None, None], (self.num_types, h, w)).copy().astype(np.float32)
        else:
            action_map = (
                mask[None, :, :] * a_fg[:, None, None]
                + (np.float32(1.0) - mask[None, :, :]) * a_bg[:, None, None]
            ).astype(np.float32)
        return x_src, action_map, x_tgt, s_src


def _infer_clean_dataset_bucket(path: Path) -> str:
    """Classify a clean reference image into its source dataset bucket (`div2k`, `flickr2k`, `lsdir`, `ost`, ...)."""
    low = str(path).replace("\\", "/").lower()
    if "div2k" in low:
        return "div2k"
    if "flickr2k" in low or "flickr" in low:
        return "flickr2k"
    if "lsdir" in low:
        return "lsdir"
    if "/ost" in low or "ost_" in low or "outdoorscene" in low:
        return "ost"
    if "ffhq" in low:
        return "ffhq"
    return path.parent.name.lower() or "default"


def _infer_paired_dataset_bucket(blur_path: Path) -> str:
    """Classify a paired motion-blur record into its sub-dataset (`gopro`, `hide`, `realblur`, `blurdataset`, ...)."""
    low = str(blur_path).replace("\\", "/").lower()
    if "gopro" in low or "gopr" in low:
        return "gopro"
    if "hide" in low:
        return "hide"
    if "realblur" in low:
        return "realblur"
    if "blurdataset" in low or "blurset" in low or "certh" in low:
        return "blurdataset"
    if "dpdd" in low or "realdof" in low:
        return "dpdd"
    return blur_path.parent.parent.name.lower() or "paired_default"


class ControllableRestorationDataset(_TorchDataset):
    """Artifact-First, Dataset-Balanced Hybrid Clean-Reference + Real-Paired Restoration Dataset."""

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

        self.motion_real_pair_frac = (
            float(getattr(config, "MOTION_REAL_PAIR_FRAC", getattr(config, "PAIRED_REAL_PROB", 0.50)))
            if self.paired_records
            else 0.0
        )

        # Group clean images into balanced dataset buckets (`div2k`, `flickr2k`, `lsdir`, `ost`, ...)
        self.clean_buckets: Dict[str, List[Path]] = {}
        for p in self.image_paths:
            bkey = _infer_clean_dataset_bucket(Path(p))
            self.clean_buckets.setdefault(bkey, []).append(Path(p))
        self._clean_bucket_names = sorted(self.clean_buckets.keys())
        self._clean_bucket_probs = self._compute_clean_bucket_probs(
            self._clean_bucket_names, getattr(config, "CLEAN_DATASET_WEIGHTS", None)
        )
        # High-priority DIV2K/LSDIR/Flickr2K weights for `super_resolution` crispness
        sr_override = dict(getattr(config, "CLEAN_DATASET_WEIGHTS", {}) or {})
        if "div2k" in sr_override:
            sr_override["div2k"] = sr_override["div2k"] * 1.35
        if "ost" in sr_override:
            sr_override["ost"] = sr_override["ost"] * 0.65
        self._sr_clean_bucket_probs = self._compute_clean_bucket_probs(self._clean_bucket_names, sr_override)

        # Group paired motion records into per-subdataset balanced buckets (`gopro`, `hide`, `realblur`, ...)
        self.paired_buckets: Dict[str, List[Tuple[Path, Path, str]]] = {}
        for rec in self.paired_records:
            bkey = _infer_paired_dataset_bucket(Path(rec[0]))
            self.paired_buckets.setdefault(bkey, []).append(rec)
        self._paired_bucket_names = sorted(self.paired_buckets.keys())

        self._sharpness_candidates = max(1, int(getattr(config, "CLEAN_PATCH_SHARPNESS_CANDIDATES", 5)))
        self._min_lap_var = float(getattr(config, "CLEAN_PATCH_MIN_LAPLACIAN_VAR", 65.0))
        self._smooth_allow_prob = float(getattr(config, "CLEAN_PATCH_SMOOTH_ALLOW_PROB", 0.08))
        self._paired_min_motion_diff = float(getattr(config, "PAIRED_MIN_MOTION_DIFF", 9.0))

        self._cache_capacity = int(getattr(config, "DECODE_CACHE_SIZE", 16)) if is_train else 0
        self._paired_cache_capacity = max(4, self._cache_capacity // 2) if (is_train and self.paired_records) else 0
        self._max_crops_per_decode = int(getattr(config, "CROPS_PER_DECODE", 8)) if is_train else 1
        # Per-bucket decode rings so dataset balancing is preserved even with decode caching!
        self._decoded_ring_by_bucket: Dict[str, list] = {}
        self._paired_ring_by_bucket: Dict[str, list] = {}

    @staticmethod
    def _compute_clean_bucket_probs(
        bucket_names: List[str], weights_map: Optional[Dict[str, float]]
    ) -> np.ndarray:
        if not bucket_names:
            return np.array([], dtype=np.float64)
        wmap = {str(k).lower(): float(v) for k, v in (weights_map or {}).items()}
        raw = np.array([max(0.05, wmap.get(name, 0.25)) for name in bucket_names], dtype=np.float64)
        return raw / raw.sum()

    def __len__(self):
        return len(self.image_paths) if self.image_paths else len(self.paired_records)

    def set_curriculum_boost(self, boost: float) -> None:
        self.pipeline.curriculum_single_boost = float(np.clip(boost, 0.0, 1.0))

    def set_detail_stage_focus(self, enabled: bool) -> None:
        self.pipeline.detail_stage_focus = bool(enabled)

    @staticmethod
    def _estimate_patch_laplacian_var(crop_rgb: np.ndarray) -> float:
        """Fast 2x-subsampled grayscale Laplacian variance to reject blurry/bokeh patches."""
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

    def _decode_rgb(self, image_path) -> np.ndarray:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_crop_from_bucket(
        self, bucket_name: str, rng: np.random.Generator, max_attempts: int = 6
    ) -> np.ndarray:
        """Load a sharpness-gated crop from a specific dataset bucket with per-bucket decode caching."""
        ring = self._decoded_ring_by_bucket.setdefault(bucket_name, [])
        per_bucket_cap = max(2, self._cache_capacity // max(1, len(self._clean_bucket_names)))
        if self.is_train and self._cache_capacity > 0 and len(ring) >= per_bucket_cap:
            slot = int(rng.integers(0, len(ring)))
            rgb_cached, rem = ring[slot]
            if rem > 1:
                ring[slot][1] = rem - 1
            else:
                ring.pop(slot)
            return self._crop_and_augment(rgb_cached)

        bucket_paths = self.clean_buckets[bucket_name]
        n = len(bucket_paths)
        ps = self.patch_size
        base_idx = int(rng.integers(0, n))
        last_err = None
        fallback_small = None
        for attempt in range(max_attempts):
            cand_idx = (base_idx + attempt) % n
            try:
                rgb_full = self._decode_rgb(bucket_paths[cand_idx])
                if self.is_train and n > 1 and (rgb_full.shape[0] < ps or rgb_full.shape[1] < ps):
                    fallback_small = rgb_full
                    continue
                if (
                    self.is_train
                    and self._cache_capacity > 0
                    and self._max_crops_per_decode > 1
                    and min(rgb_full.shape[0], rgb_full.shape[1]) >= int(ps * 1.5)
                ):
                    ring.append([rgb_full, self._max_crops_per_decode - 1])
                return self._crop_and_augment(rgb_full)
            except Exception as err:
                last_err = err

        if fallback_small is not None:
            return self._crop_and_augment(fallback_small)
        raise RuntimeError(f"Failed to read valid image from bucket '{bucket_name}': {last_err}")

    def _load_crop_with_retry(self, idx: int, max_attempts: int = 6) -> np.ndarray:
        """Deterministic indexed crop load (used by validation / evaluation sweep)."""
        n = len(self.image_paths)
        ps = self.patch_size
        last_err = None
        fallback_small = None
        for attempt in range(max_attempts):
            cand_idx = (idx + attempt) % n
            try:
                rgb_full = self._decode_rgb(self.image_paths[cand_idx])
                if self.is_train and n > 1 and (rgb_full.shape[0] < ps or rgb_full.shape[1] < ps):
                    fallback_small = rgb_full
                    continue
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
                min_hw = min(h, w)
                native_prob = float(np.clip(getattr(self.config, "PAIRED_NATIVE_CROP_PROB", 0.85), 0.0, 1.0))
                if min_hw >= int(ps * 1.25) and random.random() >= native_prob:
                    win = random.randint(int(ps * 1.10), min(min_hw, int(ps * 1.40)))
                else:
                    win = ps

                best_b, best_s = None, None
                best_score = -1.0
                n_cands = 6 if (h > win or w > win) else 1
                for _ in range(n_cands):
                    y = random.randint(0, h - win)
                    x = random.randint(0, w - win)
                    bc = blur_rgb[y : y + win, x : x + win]
                    sc = sharp_rgb[y : y + win, x : x + win]
                    b_sub = bc[::4, ::4].astype(np.float32)
                    s_sub = sc[::4, ::4].astype(np.float32)
                    motion_diff = float(np.mean(np.abs(s_sub - b_sub)))
                    edge_energy = float(np.mean(np.abs(s_sub[:, 1:] - s_sub[:, :-1])))
                    score = motion_diff + 0.30 * edge_energy
                    if score > best_score:
                        best_score = score
                        best_b, best_s = bc, sc
                    if motion_diff >= self._paired_min_motion_diff and edge_energy >= 7.5:
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

    def _load_paired_crop_from_bucket(
        self, bucket_name: str, rng: np.random.Generator, max_attempts: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        ring = self._paired_ring_by_bucket.setdefault(bucket_name, [])
        per_bucket_cap = max(2, self._paired_cache_capacity // max(1, len(self._paired_bucket_names)))
        if self.is_train and self._paired_cache_capacity > 0 and len(ring) >= per_bucket_cap:
            slot = int(rng.integers(0, len(ring)))
            b_cached, s_cached, ch_cached, rem = ring[slot]
            if rem > 1:
                ring[slot][3] = rem - 1
            else:
                ring.pop(slot)
            b_crop, s_crop = self._crop_paired_arrays(b_cached, s_cached)
            return b_crop, s_crop, ch_cached

        recs = self.paired_buckets[bucket_name]
        m = len(recs)
        ps = self.patch_size
        base_idx = int(rng.integers(0, m))
        last_err = None
        for attempt in range(max_attempts):
            p_idx = (base_idx + attempt) % m
            blur_path, sharp_path, blur_channel = recs[p_idx]
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
                    and min(sharp_rgb.shape[0], sharp_rgb.shape[1]) >= int(ps * 1.4)
                ):
                    ring.append([blur_rgb, sharp_rgb, blur_channel, max(2, self._max_crops_per_decode - 1)])
                b_crop, s_crop = self._crop_paired_arrays(blur_rgb, sharp_rgb)
                return b_crop, s_crop, blur_channel
            except Exception as err:
                last_err = err
        raise RuntimeError(f"Failed to read paired crop from bucket '{bucket_name}': {last_err}")

    def _load_paired_crop_with_retry(self, idx: int, max_attempts: int = 5) -> Tuple[np.ndarray, np.ndarray, str]:
        m = len(self.paired_records)
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
                b_crop, s_crop = self._crop_paired_arrays(blur_rgb, sharp_rgb)
                return b_crop, s_crop, blur_channel
            except Exception as err:
                last_err = err
        raise RuntimeError(f"Failed to read valid paired sample after {max_attempts} attempts: {last_err}")

    def __getitem__(self, idx):
        rng = np.random.default_rng() if self.is_train else np.random.default_rng(self.config.SEED + idx)

        # 1. Sample ARTIFACT FIRST (balanced around artifact budget!)
        primary_artifact = self.pipeline.sample_primary_artifact(rng=rng)

        # 2. Inside the artifact, balance across DATASETS!
        eff_real_frac = self.motion_real_pair_frac * (0.70 if self.pipeline.detail_stage_focus else 1.0)
        use_paired = bool(
            self.paired_records
            and (
                not self.image_paths
                or (
                    self.is_train
                    and primary_artifact == "deblur_motion"
                    and float(rng.random()) < eff_real_frac
                )
            )
        )

        if use_paired:
            if self.is_train and self._paired_bucket_names:
                p_bucket = self._paired_bucket_names[int(rng.integers(0, len(self._paired_bucket_names)))]
                blur_u8, sharp_u8, blur_ch = self._load_paired_crop_from_bucket(p_bucket, rng=rng)
            else:
                blur_u8, sharp_u8, blur_ch = self._load_paired_crop_with_retry(idx % len(self.paired_records))
            clean_u8 = sharp_u8
            x_src_u8, x_tgt_u8, a_fg, a_bg, mask_2d, s_src, gan_clean = (
                self.pipeline.synthesize_real_pair_components(
                    blur_u8, sharp_u8, blur_channel_name="deblur_motion", rng=rng, align_only=self.align_only
                )
            )
        else:
            if self.is_train and self._clean_bucket_names:
                b_probs = (
                    self._sr_clean_bucket_probs
                    if primary_artifact == "super_resolution"
                    else self._clean_bucket_probs
                )
                c_bucket = self._clean_bucket_names[int(rng.choice(len(self._clean_bucket_names), p=b_probs))]
                clean_u8 = self._load_crop_from_bucket(c_bucket, rng=rng)
            else:
                clean_u8 = self._load_crop_with_retry(idx % max(1, len(self.image_paths)))

            x_src_u8, x_tgt_u8, a_fg, a_bg, mask_2d, s_src, gan_clean = self.pipeline.synthesize_components(
                clean_u8, rng=rng, align_only=self.align_only, primary_artifact=primary_artifact
            )

        inv_255 = np.float32(1.0 / 255.0)
        clean_t = torch.from_numpy(clean_u8).permute(2, 0, 1).float().mul_(inv_255)
        src_t = torch.from_numpy(x_src_u8).permute(2, 0, 1).float().mul_(inv_255)

        if self.align_only:
            return {"clean": clean_t, "source": src_t}

        tgt_t = torch.from_numpy(x_tgt_u8).permute(2, 0, 1).float().mul_(inv_255)
        s_src_t = torch.from_numpy(s_src)
        gan_clean_t = torch.tensor(1.0 if gan_clean else 0.0, dtype=torch.float32)

        if self.compact_ipc:
            return {
                "clean": clean_t,
                "source": src_t,
                "target": tgt_t,
                "a_fg": torch.from_numpy(a_fg),
                "a_bg": torch.from_numpy(a_bg),
                "spatial_mask": torch.from_numpy(mask_2d).unsqueeze(0).half(),
                "source_severity": s_src_t,
                "gan_clean_eligible": gan_clean_t,
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
            "gan_clean_eligible": gan_clean_t,
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
        return "super_resolution"
    if any(tok in low for tok in ("motion", "gopro", "hide", "realblur")):
        return "deblur_motion"
    return default_channel


def discover_paired_blur_datasets(
    roots_spec,
    default_channel: str = "deblur_motion",
    is_train: bool = True,
    exclude_domain_subsets: bool = True,
) -> List[Tuple[Path, Path, str]]:
    """Auto-discover paired `(blur_path, sharp_path, blur_channel)` across GoPro, HIDE, RealBlur-J & BlurDataset!"""
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

        # 1. Check RealBlur/HIDE-style `.txt` pair index files first
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

        # 2. Discover sibling `(blur, sharp)` / `(motion_blurred, sharp)` folders
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
                if b_sub.lower() == "blur_gamma" and has_canonical_blur:
                    continue
                # Skip defocus subfolders when scanning for motion-blur pairs
                if default_channel == "deblur_motion" and any(
                    tok in b_sub.lower() for tok in ("defocus", "bokeh", "dof")
                ):
                    continue
                b_dir = Path(cur_dir) / b_sub
                for bp in _collect_images(b_dir):
                    rel_b = os.path.splitext(os.path.relpath(str(bp), str(b_dir)))[0]
                    key_b = _normalize_pair_stem(rel_b)
                    if key_b in sharp_index:
                        ch = _infer_pair_channel(str(bp), default_channel)
                        root_pairs.append((bp, sharp_index[key_b], ch))

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
    """Discover clean 2K references (dataset-balanced) + real paired motion-blur datasets."""
    split_name = "train" if is_train else "val"
    explicit_dir = dataset_path or (config.TRAIN_DATASET_PATH if is_train else config.VAL_DATASET_PATH)

    paired_records: List[Tuple[Path, Path, str]] = []
    if is_train:
        motion_root = getattr(config, "PAIRED_MOTION_DATASET_PATH", None)
        if motion_root:
            paired_records.extend(discover_paired_blur_datasets(motion_root, "deblur_motion", is_train=True))

    paths: List[Path] = []
    source_desc = ""

    if explicit_dir:
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

        # Exclude BOTH blurred inputs AND paired sharp targets from clean pool (unless REUSE_PAIRED_CLEAN_FOR_SYNTH=True)
        reuse_paired_clean = bool(getattr(config, "REUSE_PAIRED_CLEAN_FOR_SYNTH", False))
        blur_exclude = {str(bp) for bp, _sp, _ch in paired_records}
        if not reuse_paired_clean:
            blur_exclude |= {str(sp) for _bp, sp, _ch in paired_records}

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

    if is_train and paired_records and bool(getattr(config, "REUSE_PAIRED_CLEAN_FOR_SYNTH", False)):
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

    limit = config.TRAIN_IMAGES_COUNT if is_train else config.VAL_IMAGES_COUNT
    if limit and 0 < limit < len(paths):
        rng = random.Random(config.SEED + (0 if is_train else 1))
        shuffled = list(paths)
        rng.shuffle(shuffled)
        paths = sorted(shuffled[:limit])

    ds = ControllableRestorationDataset(
        paths,
        config,
        is_train=is_train,
        align_only=align_only,
        compact_ipc=compact_ipc,
        paired_records=paired_records,
    )
    if verbose:
        clean_b_str = ", ".join(
            f"{k}:{len(ds.clean_buckets[k])}({p*100:.0f}%)"
            for k, p in zip(ds._clean_bucket_names, ds._clean_bucket_probs)
        )
        pair_msg = ""
        if paired_records:
            pb_str = ", ".join(f"{k}:{len(ds.paired_buckets[k])}" for k in ds._paired_bucket_names)
            pair_msg = (
                f" | Real Motion Pairs: {len(paired_records)} balanced [{pb_str}] "
                f"(real_frac={ds.motion_real_pair_frac:.2f})"
            )
        art_str = ", ".join(
            f"{n}:{p*100:.0f}%"
            for n, p in zip(ds.pipeline.degradation_types, ds.pipeline.stage1_artifact_probs)
        )
        print(
            f"[{split_name}] Clean HR Pool ({len(paths)} imgs) [{clean_b_str}]{pair_msg}\n"
            f"  ↳ Artifact Budget (Stage 1): {art_str}",
            flush=True,
        )
    return ds
