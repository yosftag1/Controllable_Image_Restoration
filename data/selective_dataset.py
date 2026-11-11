"""Seed-synchronized selective & spatially localized degradation synthesis."""

from dataclasses import dataclass
from pathlib import Path
import os
import random

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

    noise_unit: np.ndarray          # (H, W, 3) float32 N(0, 1)
    motion_angle: float             # Degrees in [0, 180)
    color_gains_unit: np.ndarray    # (3,) float32 in [-1, 1]
    hue_shift_unit: float           # Float in [-1, 1]
    sat_loss_frac: float            # Float in [0.15, 1.0]
    underexpose: bool               # True = dark gamma (>1), False = bright/washed gamma (<1)
    brightness_frac: float          # Float in [0.15, 1.0]
    contrast_frac: float            # Float in [0.15, 1.0]


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

    def sample_params(self, height: int, width: int, rng: np.random.Generator = None) -> DegradationParams:
        """Sample all stochastic physical realizations upfront, independent of severity."""
        rng = rng or np.random.default_rng()
        min_angle, max_angle = self.config.MOTION_BLUR_ANGLE_RANGE
        return DegradationParams(
            noise_unit=rng.standard_normal((height, width, 3), dtype=np.float32),
            motion_angle=float(rng.uniform(min_angle, max_angle)),
            color_gains_unit=rng.uniform(-1.0, 1.0, size=(3,)).astype(np.float32),
            hue_shift_unit=float(rng.uniform(-1.0, 1.0)),
            sat_loss_frac=float(rng.uniform(0.25, 1.0)),
            underexpose=bool(rng.random() < 0.5),
            brightness_frac=float(rng.uniform(0.20, 1.0)),
            contrast_frac=float(rng.uniform(0.20, 1.0)),
        )

    @staticmethod
    def _motion_blur_kernel(size: int, angle: float) -> np.ndarray:
        if size <= 1:
            return np.ones((1, 1), dtype=np.float32)
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        kernel[center, :] = 1.0
        rot = cv2.getRotationMatrix2D((float(center), float(center)), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot, (size, size))
        total = float(kernel.sum())
        return kernel / total if total > 1e-6 else np.ones((1, 1), dtype=np.float32)

    def _apply_single(self, img_float: np.ndarray, deg_name: str, severity: float, params: DegradationParams) -> np.ndarray:
        """Apply one degradation smoothly in float32 [0, 255] space so severity -> 0 is continuous."""
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
            # Continuous kernel size from 1.0 (exact identity) to max_k
            cont_k = 1.0 + severity * (float(max_k) - 1.0)
            if cont_k <= 1.05:
                return img_float
            # Find surrounding odd kernel sizes (1, 3, 5, 7, ...) for smooth linear kernel interpolation
            k_low = max(1, int(np.floor((cont_k - 1.0) / 2.0)) * 2 + 1)
            k_high = k_low + 2
            alpha = (cont_k - float(k_low)) / 2.0

            out_low = img_float if k_low == 1 else cv2.filter2D(
                img_float, -1, self._motion_blur_kernel(k_low, params.motion_angle)
            )
            if alpha <= 1e-3:
                return out_low
            out_high = cv2.filter2D(
                img_float, -1, self._motion_blur_kernel(k_high, params.motion_angle)
            )
            return (1.0 - alpha) * out_low + alpha * out_high

        if deg_name == "exposure_issue":
            norm = np.clip(img_float / 255.0, 0.0, 1.0)
            min_g, max_g = self.config.GAMMA_RANGE
            if params.underexpose:
                gamma = 1.0 + severity * (max_g - 1.0)
                b_delta = -params.brightness_frac * self.config.BRIGHTNESS_MAX * severity
            else:
                gamma = 1.0 - severity * (1.0 - min_g)
                b_delta = params.brightness_frac * self.config.BRIGHTNESS_MAX * severity

            c_scale = 1.0 - params.contrast_frac * self.config.CONTRAST_MAX * severity
            shifted = (norm - 0.5) * c_scale + 0.5 + b_delta
            shifted = np.clip(shifted, 1e-4, 1.0)
            curved = np.power(shifted, gamma)
            return np.clip(curved * 255.0, 0.0, 255.0)

        if deg_name == "color_degradation":
            norm = np.clip(img_float / 255.0, 0.0, 1.0)
            max_shift = self.config.COLOR_SHIFT_MAX * severity
            scales = np.array([1.0, 0.6, 1.0], dtype=np.float32)
            gains = 1.0 + params.color_gains_unit * scales * max_shift
            gains = gains / max(1e-4, float(np.mean(gains)))
            cast = np.clip(norm * gains.reshape(1, 1, 3), 0.0, 1.0)

            hsv = cv2.cvtColor((cast * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hue_delta = params.hue_shift_unit * self.config.HUE_SHIFT_MAX * severity
            sat_scale = 1.0 - params.sat_loss_frac * self.config.SATURATION_LOSS_MAX * severity
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 180.0
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0.0, 255.0)
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
            # Blend smoothly at tiny severities to avoid uint8 HSV round-trip jump near 0
            blend = min(1.0, severity / 0.10)
            return (1.0 - blend) * img_float + blend * rgb

        if deg_name == "downsample_upsample":
            h, w = img_float.shape[:2]
            min_scale, _ = self.config.DOWNSAMPLE_SCALE_RANGE
            # Continuous scale from 1.0 (identity at severity=0) down to min_scale (at severity=1)
            scale = 1.0 - severity * (1.0 - min_scale)
            if scale >= 0.98:
                return img_float
            new_w = max(8, int(round(w * scale)))
            new_h = max(8, int(round(h * scale)))
            down = cv2.resize(img_float, (new_w, new_h), interpolation=cv2.INTER_AREA)
            up = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
            return np.clip(up, 0.0, 255.0)

        if deg_name == "gaussian_noise":
            _, max_std = self.config.GAUSSIAN_NOISE_STD_RANGE
            # Proportional std so severity=0 -> 0.0 std (exact identity)
            noise_std = severity * max_std
            return np.clip(img_float + params.noise_unit * noise_std, 0.0, 255.0)

        if deg_name == "color_banding":
            min_levels, max_levels = self.config.BANDING_LEVELS_RANGE
            min_bits = np.log2(float(min_levels))
            max_bits = np.log2(float(max_levels))
            bits = max_bits - severity * (max_bits - min_bits)
            levels = max(4, int(round(2.0 ** bits)))

            u8 = np.clip(img_float, 0.0, 255.0).astype(np.uint8)
            sigma_color = 6.0 + 18.0 * severity
            smoothed = cv2.bilateralFilter(u8, d=5, sigmaColor=sigma_color, sigmaSpace=5)
            lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB).astype(np.float32)
            l_step = 255.0 / float(levels - 1)
            ab_step = 255.0 / float(max(6, levels * 2) - 1)
            lab[:, :, 0] = np.round(lab[:, :, 0] / l_step) * l_step
            lab[:, :, 1] = np.round((lab[:, :, 1] - 128.0) / ab_step) * ab_step + 128.0
            lab[:, :, 2] = np.round((lab[:, :, 2] - 128.0) / ab_step) * ab_step + 128.0
            banded = cv2.cvtColor(np.clip(lab, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
            blend = min(1.0, severity / 0.12)
            return (1.0 - blend) * img_float + blend * banded

        if deg_name == "jpeg_compression":
            min_q, max_q = self.config.JPEG_QUALITY_RANGE
            quality = int(round(max_q - severity * (max_q - min_q)))
            quality = int(np.clip(quality, 5, 95))
            u8_bgr = cv2.cvtColor(np.clip(img_float, 0.0, 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".jpg", u8_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                return img_float
            dec_rgb = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
            blend = min(1.0, severity / 0.12)
            return (1.0 - blend) * img_float + blend * dec_rgb

        raise ValueError(f"Unknown degradation type: {deg_name}")

    def render_with_params(self, clean_rgb_u8: np.ndarray, severity_vec: np.ndarray, params: DegradationParams) -> np.ndarray:
        """Render a degraded uint8 RGB image from clean_rgb_u8 using frozen DegradationParams."""
        cur = clean_rgb_u8.astype(np.float32)
        for deg_name in self.PHYSICAL_ORDER:
            idx = self.type_to_idx[deg_name]
            sev = float(severity_vec[idx])
            if sev > 1e-4:
                cur = self._apply_single(cur, deg_name, sev, params)
        return np.clip(np.round(cur), 0.0, 255.0).astype(np.uint8)

    def sample_source_severity(self, rng: np.random.Generator = None) -> np.ndarray:
        """Sample 1 to 3 active degradations in [min_severity, 1.0]."""
        rng = rng or np.random.default_rng()
        if self.max_compound > 1 and rng.random() < self.compound_prob:
            k = 2 if (self.max_compound == 2 or rng.random() < 0.70) else min(3, self.max_compound)
        else:
            k = 1

        chosen = list(rng.choice(self.num_types, size=min(k, self.num_types), replace=False))
        chosen_names = [self.degradation_types[i] for i in chosen]

        # Avoid stacking two optical blurs at once so motion vs. defocus stay physically distinct
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
        return np.clip(action, 0.0, 1.0)

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
            # Selective subset removal: fix 1..(N-1) active degradations, preserve the rest (0.0)
            num_fix = int(rng.integers(1, len(active_indices)))
            fix_ids = rng.choice(active_indices, size=num_fix, replace=False)
            for idx in fix_ids:
                action[idx] = 1.0 if rng.random() < 0.7 else float(rng.uniform(0.5, 1.0))

        elif u < p_sel + p_part:
            # Continuous fractional sliders in [0.15, 0.90]
            for idx in active_indices:
                if rng.random() < 0.80:
                    action[idx] = float(rng.uniform(0.15, 0.90))

        elif u < p_sel + p_part + p_full:
            # Full restoration of all present degradations
            for idx in active_indices:
                action[idx] = 1.0

        elif u < p_sel + p_part + p_full + p_fp:
            # False-positive / over-request safety: user enables sliders on already-clean attributes!
            for idx in active_indices:
                if rng.random() < 0.65:
                    action[idx] = float(rng.choice([0.0, 0.5, 1.0]))
            if inactive_indices:
                n_fp = int(rng.integers(1, min(3, len(inactive_indices)) + 1))
                for idx in rng.choice(inactive_indices, size=n_fp, replace=False):
                    action[idx] = float(rng.uniform(0.4, 1.0))

        else:
            # Pure identity preservation (all zeros -> output must exactly equal x_src)
            action[:] = 0.0

        return self._enforce_physical_action_consistency(s_src, action)

    def sample_spatial_mask(self, clean_rgb_u8: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        """Generate a smooth feathered spatial region mask M in [0, 1]^(H, W) for localized control."""
        rng = rng or np.random.default_rng()
        h, w = clean_rgb_u8.shape[:2]
        mode = float(rng.random())

        if mode < 0.45:
            # Color/luminance object-region mask from clean image LAB clustering (fast, multiprocessing-safe)
            lab = cv2.cvtColor(clean_rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
            cy = int(rng.integers(h // 6, 5 * h // 6))
            cx = int(rng.integers(w // 6, 5 * w // 6))
            anchor = lab[cy, cx]
            dist = np.linalg.norm(lab - anchor.reshape(1, 1, 3), axis=-1)
            thresh = float(np.percentile(dist, rng.uniform(25.0, 55.0)))
            raw_mask = (dist <= thresh).astype(np.float32)
            sigma = float(rng.uniform(4.0, 12.0))
            mask = cv2.GaussianBlur(raw_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
        elif mode < 0.80:
            # Smooth elliptical / brush blob mask
            mask = np.zeros((h, w), dtype=np.float32)
            num_blobs = int(rng.integers(1, 3))
            for _ in range(num_blobs):
                cx = int(rng.integers(w // 5, 4 * w // 5))
                cy = int(rng.integers(h // 5, 4 * h // 5))
                rx = int(rng.integers(w // 6, w // 2))
                ry = int(rng.integers(h // 6, h // 2))
                angle = float(rng.uniform(0.0, 180.0))
                cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 1.0, -1)
            sigma = float(rng.uniform(8.0, 22.0))
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
        else:
            # Smooth linear directional split (e.g., sky/horizon or foreground/background gradient)
            yy, xx = np.meshgrid(
                np.linspace(-1.0, 1.0, h, dtype=np.float32),
                np.linspace(-1.0, 1.0, w, dtype=np.float32),
                indexing="ij",
            )
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            proj = np.cos(theta) * xx + np.sin(theta) * yy
            offset = float(rng.uniform(-0.35, 0.35))
            sharpness = float(rng.uniform(4.0, 10.0))
            mask = 1.0 / (1.0 + np.exp(-sharpness * (proj - offset)))

        m_max = float(mask.max())
        if m_max > 1e-5:
            mask = mask / m_max
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    def synthesize_pair(self, clean_rgb_u8: np.ndarray, rng: np.random.Generator = None):
        """Synthesize seed-synchronized (x_src, action_map [8, H, W], x_tgt, s_src)."""
        rng = rng or np.random.default_rng()
        h, w = clean_rgb_u8.shape[:2]
        params = self.sample_params(h, w, rng=rng)
        s_src = self.sample_source_severity(rng=rng)
        x_src = self.render_with_params(clean_rgb_u8, s_src, params)

        a_fg = self.sample_action_vector(s_src, rng=rng)
        s_tgt_fg = np.clip((1.0 - a_fg) * s_src, 0.0, 1.0)
        x_tgt_fg = (
            x_src.copy()
            if np.allclose(s_tgt_fg, s_src, atol=1e-4)
            else self.render_with_params(clean_rgb_u8, s_tgt_fg, params)
        )

        if rng.random() < self.spatial_mask_prob:
            # Localized / two-region control: foreground gets a_fg, background gets a_bg (often 0 = preserve!)
            if rng.random() < 0.65:
                a_bg = np.zeros(self.num_types, dtype=np.float32)
            else:
                a_bg = self.sample_action_vector(s_src, rng=rng)

            s_tgt_bg = np.clip((1.0 - a_bg) * s_src, 0.0, 1.0)
            x_tgt_bg = (
                x_src.copy()
                if np.allclose(s_tgt_bg, s_src, atol=1e-4)
                else self.render_with_params(clean_rgb_u8, s_tgt_bg, params)
            )

            mask = self.sample_spatial_mask(clean_rgb_u8, rng=rng)  # (H, W)
            mask_3d = mask[:, :, None]
            x_tgt = np.clip(
                np.round(mask_3d * x_tgt_fg.astype(np.float32) + (1.0 - mask_3d) * x_tgt_bg.astype(np.float32)),
                0.0,
                255.0,
            ).astype(np.uint8)
            action_map = (
                mask[None, :, :] * a_fg[:, None, None] + (1.0 - mask[None, :, :]) * a_bg[:, None, None]
            ).astype(np.float32)
        else:
            x_tgt = x_tgt_fg
            action_map = np.broadcast_to(a_fg[:, None, None], (self.num_types, h, w)).copy().astype(np.float32)

        return x_src, action_map, x_tgt, s_src


class ControllableRestorationDataset(_TorchDataset):
    """PyTorch Dataset yielding (clean, source, target, action_map, action_global, source_severity)."""

    def __init__(self, image_paths, config, is_train=True):
        self.image_paths = list(image_paths)
        self.config = config
        self.is_train = is_train
        self.patch_size = config.PATCH_SIZE
        self.pipeline = SelectiveDegradationPipeline(config)

    def __len__(self):
        return len(self.image_paths)

    def _load_crop(self, image_path: str) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ps = self.patch_size

        if h >= ps and w >= ps:
            if self.is_train:
                y = random.randint(0, h - ps)
                x = random.randint(0, w - ps)
            else:
                y = (h - ps) // 2
                x = (w - ps) // 2
            crop = img[y : y + ps, x : x + ps]
        else:
            crop = cv2.resize(img, (ps, ps), interpolation=cv2.INTER_CUBIC)

        if self.is_train:
            if random.random() < 0.5:
                crop = np.fliplr(crop)
            if random.random() < 0.5:
                crop = np.flipud(crop)
            k = random.randint(0, 3)
            if k > 0:
                crop = np.rot90(crop, k)
            crop = np.ascontiguousarray(crop)
        return crop

    def __getitem__(self, idx):
        clean_u8 = self._load_crop(self.image_paths[idx])
        if self.is_train:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.config.SEED + idx)

        x_src_u8, action_map, x_tgt_u8, s_src = self.pipeline.synthesize_pair(clean_u8, rng=rng)

        clean_t = torch.from_numpy(clean_u8.astype(np.float32) / 255.0).permute(2, 0, 1)
        src_t = torch.from_numpy(x_src_u8.astype(np.float32) / 255.0).permute(2, 0, 1)
        tgt_t = torch.from_numpy(x_tgt_u8.astype(np.float32) / 255.0).permute(2, 0, 1)
        action_map_t = torch.from_numpy(action_map)                         # [8, H, W]
        action_global_t = action_map_t.mean(dim=(1, 2))                     # [8]
        s_src_t = torch.from_numpy(s_src)                                   # [8]

        return {
            "clean": clean_t,
            "source": src_t,
            "target": tgt_t,
            "action_map": action_map_t,
            "action_global": action_global_t,
            "source_severity": s_src_t,
        }


def _collect_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def create_controllable_dataset(config, is_train=True):
    """Discover DIV2K or arbitrary folder images and construct ControllableRestorationDataset."""
    explicit_dir = config.TRAIN_DATASET_PATH if is_train else config.VAL_DATASET_PATH
    if explicit_dir:
        paths = _collect_images(Path(explicit_dir))
        if not paths:
            raise FileNotFoundError(f"No images found in {explicit_dir}")
    else:
        root = Path(config.DATASET_PATH)
        sub = root / ("DIV2K_train_HR" if is_train else "DIV2K_valid_HR")
        if sub.exists():
            paths = _collect_images(sub)
        else:
            all_paths = _collect_images(root)
            if not all_paths:
                raise FileNotFoundError(
                    f"No images found in {root}. Pass --data-root pointing to DIV2K or an image folder."
                )
            rng = random.Random(config.SEED)
            rng.shuffle(all_paths)
            split = max(1, int(len(all_paths) * 0.9))
            paths = all_paths[:split] if is_train else all_paths[split:]

    limit = config.TRAIN_IMAGES_COUNT if is_train else config.VAL_IMAGES_COUNT
    if limit and 0 < limit < len(paths):
        paths = paths[:limit]

    split_name = "train" if is_train else "val"
    print(f"[{split_name}] Loaded {len(paths)} clean reference images.")
    return ControllableRestorationDataset(paths, config, is_train=is_train)
