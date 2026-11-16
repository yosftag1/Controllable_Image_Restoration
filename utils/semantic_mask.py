"""Action string parser (4 unsigned fixes + 2 signed grades) & 1-Click Semantic Object Mask extractor."""

from typing import Dict, Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


DEFAULT_SIGNED_TYPES: Set[str] = {"exposure", "warmth"}

# Canonical mapping including intuitive shorthand and full backward compatibility with legacy V1-V3 names
SHORT_ALIASES: Dict[str, str] = {
    # 1. Sensor noise removal [0, 1]
    "noise": "denoise",
    "gaussian_noise": "denoise",
    "denoise": "denoise",
    "grain": "denoise",
    # 2. Compression & quantization cleanup [0, 1]
    "deartifact": "deartifact",
    "artifact": "deartifact",
    "artifacts": "deartifact",
    "jpeg": "deartifact",
    "dejpeg": "deartifact",
    "jpeg_compression": "deartifact",
    "banding": "deartifact",
    "deband": "deartifact",
    "color_banding": "deartifact",
    # 3. Directional camera-shake / motion deblur [0, 1]
    "motion": "deblur_motion",
    "motion_blur": "deblur_motion",
    "deblur_motion": "deblur_motion",
    "deblur": "deblur_motion",
    # 4. Optical defocus & low-res focus sharpening [0, 1]
    "focus": "sharpen_focus",
    "sharpen": "sharpen_focus",
    "sharpen_focus": "sharpen_focus",
    "defocus": "sharpen_focus",
    "defocus_blur": "sharpen_focus",
    "bokeh": "sharpen_focus",
    "sr": "sharpen_focus",
    "upscale": "sharpen_focus",
    "downsample": "sharpen_focus",
    "downsample_upsample": "sharpen_focus",
    # 5. Signed directional exposure [-1, +1] (- = darken, + = brighten)
    "exposure": "exposure",
    "exposure_issue": "exposure",
    "ev": "exposure",
    "light": "exposure",
    # 6. Signed directional white balance / warmth [-1, +1] (- = cool, + = warm)
    "warmth": "warmth",
    "wb": "warmth",
    "temp": "warmth",
    "temperature": "warmth",
    "color": "warmth",
    "color_degradation": "warmth",
}

# Convenience directional aliases that automatically apply +1 or -1 sign to positive magnitudes
SIGN_FLIP_ALIASES: Dict[str, Tuple[str, float]] = {
    "brighten": ("exposure", 1.0),
    "darken": ("exposure", -1.0),
    "warm": ("warmth", 1.0),
    "cool": ("warmth", -1.0),
}


def parse_action_spec(
    spec: str,
    degradation_types: List[str],
    signed_types: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Parse CLI action string into a float32 array (`[0, 1]` for fixes, `[-1, +1]` for signed tone/WB).

    Note:
        `"all"` sets all objective restoration sliders (`denoise`, `deartifact`, `deblur_motion`,
        `sharpen_focus`) to `1.0` while keeping signed artistic sliders (`exposure`, `warmth`)
        at `0.0` unless explicitly specified (e.g. `"all,exposure=0.6,warmth=-0.5"` or `"all,darken=0.5"`).

    Examples:
        "all" -> [1, 1, 1, 1, 0, 0]
        "none" -> [0, 0, 0, 0, 0, 0]
        "denoise=1.0,motion=0.0,exposure=0.7,warmth=-0.5"
        "all,motion=0.0,darken=0.6"
    """
    signed_set = set(DEFAULT_SIGNED_TYPES if signed_types is None else signed_types)
    vec = np.zeros(len(degradation_types), dtype=np.float32)
    if spec is None:
        return vec

    cleaned = spec.strip()
    if not cleaned or cleaned.lower() in ("none", "preserve", "zero", "identity"):
        return vec

    type_to_idx = {name: i for i, name in enumerate(degradation_types)}

    def _apply_all_objective():
        for i, name in enumerate(degradation_types):
            vec[i] = 0.0 if name in signed_set else 1.0

    if cleaned.lower() in ("all", "full", "restore_all"):
        _apply_all_objective()
        return vec

    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    for part in parts:
        low = part.lower()
        if low in ("all", "full", "restore_all"):
            _apply_all_objective()
            continue
        if low in ("none", "preserve", "zero", "identity"):
            vec[:] = 0.0
            continue
        if "=" not in part and ":" not in part:
            raise ValueError(
                f"Invalid action token '{part}'. Expected 'name=value' "
                f"(e.g., 'denoise=1.0,motion=0.0,exposure=0.6') or 'all'/'none'."
            )
        sep = "=" if "=" in part else ":"
        raw_key, raw_val = [s.strip() for s in part.split(sep, 1)]
        key_low = raw_key.lower()
        val = float(raw_val)

        if key_low in SIGN_FLIP_ALIASES:
            canonical, sign_mul = SIGN_FLIP_ALIASES[key_low]
            val = sign_mul * abs(val)
        else:
            canonical = SHORT_ALIASES.get(key_low, key_low)

        if canonical not in type_to_idx:
            # Support running against custom degradation_types lists if needed
            if key_low in type_to_idx:
                canonical = key_low
            else:
                valid = ", ".join(degradation_types)
                raise ValueError(f"Unknown control '{raw_key}'. Valid controls: {valid}")

        lo = -1.0 if canonical in signed_set else 0.0
        vec[type_to_idx[canonical]] = float(np.clip(val, lo, 1.0))

    return vec


def _no_grad_decorator(fn):
    return torch.no_grad()(fn) if torch is not None else fn


@_no_grad_decorator
def extract_one_click_semantic_mask(
    semantic_encoder,
    rgb_01,
    point_xy: Tuple[float, float],
    threshold: Optional[float] = None,
    feather_sigma: float = 6.0,
    max_side: int = 1024,
) -> np.ndarray:
    """Extract a soft object mask M in [0, 1]^(H, W) from a single (x, y) click."""
    _, _, h, w = rgb_01.shape
    px, py = float(point_xy[0]), float(point_xy[1])
    if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0 and (px < 1.0 or py < 1.0):
        norm_x, norm_y = px, py
    else:
        norm_x = np.clip(px / max(1.0, w - 1.0), 0.0, 1.0)
        norm_y = np.clip(py / max(1.0, h - 1.0), 0.0, 1.0)

    rgb_proc = rgb_01
    if max_side > 0 and max(h, w) > max_side:
        scale = float(max_side) / float(max(h, w))
        nh = max(16, int(round((h * scale) / 16.0)) * 16)
        nw = max(16, int(round((w * scale) / 16.0)) * 16)
        rgb_proc = F.interpolate(rgb_01, size=(nh, nw), mode="bilinear", align_corners=False)

    if rgb_proc.is_cuda:
        with torch.amp.autocast("cuda", enabled=True):
            sem_out = semantic_encoder(rgb_proc, use_lora=True)
    else:
        sem_out = semantic_encoder(rgb_proc, use_lora=True)
    grid = F.normalize(sem_out["grid"].float(), dim=1)
    del sem_out

    sample_coord = torch.tensor(
        [[[[norm_x * 2.0 - 1.0, norm_y * 2.0 - 1.0]]]], device=grid.device, dtype=torch.float32
    )
    anchor_vec = F.grid_sample(grid, sample_coord, mode="bilinear", align_corners=True)
    anchor_vec = F.normalize(anchor_vec, dim=1)

    sim_lowres = (grid * anchor_vec).sum(dim=1, keepdim=True)
    sim_full = F.interpolate(sim_lowres, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    sim_np = sim_full.detach().cpu().numpy()

    s_min, s_max = float(sim_np.min()), float(sim_np.max())
    sim_01 = (sim_np - s_min) / max(1e-6, s_max - s_min)

    if threshold is None:
        sim_u8 = np.clip(sim_01 * 255.0, 0.0, 255.0).astype(np.uint8)
        _, binary_u8 = cv2.threshold(sim_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw_mask = binary_u8.astype(np.float32) / 255.0
        cx = int(round(norm_x * (w - 1)))
        cy = int(round(norm_y * (h - 1)))
        num_labels, labels = cv2.connectedComponents(binary_u8)
        clicked_label = int(labels[cy, cx])
        if clicked_label > 0 and num_labels > 2:
            raw_mask = (labels == clicked_label).astype(np.float32)
    else:
        raw_mask = (sim_np >= float(threshold)).astype(np.float32)

    if feather_sigma > 0.0:
        mask = cv2.GaussianBlur(raw_mask, (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma)
    else:
        mask = raw_mask

    m_max = float(mask.max())
    if m_max > 1e-5:
        mask = mask / m_max
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def load_grayscale_mask(mask_path: str, height: int, width: int, feather_sigma: float = 2.0) -> np.ndarray:
    """Load an external grayscale mask image into [0, 1]^(H, W)."""
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Mask image not found: {mask_path}")
    if m.shape[:2] != (height, width):
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR)
    m_f = m.astype(np.float32) / 255.0
    if feather_sigma > 0.0:
        m_f = cv2.GaussianBlur(m_f, (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma)
    return np.clip(m_f, 0.0, 1.0)


def build_spatial_action_overlay(
    height: int,
    width: int,
    fg_action: np.ndarray,
    bg_action: Optional[np.ndarray] = None,
    mask_2d: Optional[np.ndarray] = None,
    device=None,
):
    """Construct [1, K, H, W] Pixel Overlay Sheet preserving signed channels in [-1, +1]."""
    fg_f = np.asarray(fg_action, dtype=np.float32)
    k = len(fg_f)
    if mask_2d is None:
        overlay = np.broadcast_to(fg_f[:, None, None], (k, height, width)).copy().astype(np.float32)
    else:
        bg = np.zeros(k, dtype=np.float32) if bg_action is None else np.asarray(bg_action, dtype=np.float32)
        m = np.clip(mask_2d, 0.0, 1.0).astype(np.float32)[None, :, :]
        overlay = (m * fg_f[:, None, None] + (1.0 - m) * bg[:, None, None]).astype(np.float32)
    if torch is None:
        return overlay[None, ...]
    dev = torch.device("cpu") if device is None else device
    return torch.from_numpy(overlay).unsqueeze(0).to(device=dev, dtype=torch.float32)


def render_mask_overlay_preview(
    rgb_u8: np.ndarray,
    mask_2d: np.ndarray,
    point_xy: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Render an emerald-tinted visualization of a localized mask + optional click marker."""
    h, w = rgb_u8.shape[:2]
    tint = np.zeros_like(rgb_u8, dtype=np.float32)
    tint[:, :, 1] = 230.0
    tint[:, :, 2] = 150.0
    alpha = (np.clip(mask_2d, 0.0, 1.0)[:, :, None]) * 0.45
    blended = np.clip(rgb_u8.astype(np.float32) * (1.0 - alpha) + tint * alpha, 0.0, 255.0).astype(np.uint8)

    binary = (mask_2d >= 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (80, 255, 160), 2, cv2.LINE_AA)

    if point_xy is not None:
        px, py = float(point_xy[0]), float(point_xy[1])
        if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0 and (px < 1.0 or py < 1.0):
            cx, cy = int(round(px * (w - 1))), int(round(py * (h - 1)))
        else:
            cx, cy = int(round(px)), int(round(py))
        cv2.circle(blended, (cx, cy), 7, (255, 70, 70), -1, cv2.LINE_AA)
        cv2.circle(blended, (cx, cy), 9, (255, 255, 255), 2, cv2.LINE_AA)
    return blended
