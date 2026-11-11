"""Action string parser & 1-Click Semantic Object Mask extractor (via C-RADIO / DINOv3)."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


SHORT_ALIASES: Dict[str, str] = {
    "noise": "gaussian_noise",
    "denoise": "gaussian_noise",
    "motion": "motion_blur",
    "deblur_motion": "motion_blur",
    "jpeg": "jpeg_compression",
    "dejpeg": "jpeg_compression",
    "color": "color_degradation",
    "wb": "color_degradation",
    "exposure": "exposure_issue",
    "light": "exposure_issue",
    "banding": "color_banding",
    "deband": "color_banding",
    "sr": "downsample_upsample",
    "upscale": "downsample_upsample",
    "downsample": "downsample_upsample",
    "defocus": "defocus_blur",
    "bokeh": "defocus_blur",
}


def parse_action_spec(spec: str, degradation_types: List[str]) -> np.ndarray:
    """Parse CLI action string into an 8D float32 array in [0, 1].

    Examples:
        "all" -> all 1.0
        "none" -> all 0.0
        "noise=1.0,motion=0.0,exposure=0.7"
        "all,motion_blur=0.0"  (fix everything EXCEPT motion blur!)
    """
    vec = np.zeros(len(degradation_types), dtype=np.float32)
    if spec is None:
        return vec

    cleaned = spec.strip()
    if not cleaned or cleaned.lower() in ("none", "preserve", "zero", "identity"):
        return vec
    if cleaned.lower() in ("all", "full", "restore_all"):
        vec[:] = 1.0
        return vec

    type_to_idx = {name: i for i, name in enumerate(degradation_types)}
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    for part in parts:
        low = part.lower()
        if low in ("all", "full"):
            vec[:] = 1.0
            continue
        if low in ("none", "preserve", "zero"):
            vec[:] = 0.0
            continue
        if "=" not in part and ":" not in part:
            raise ValueError(
                f"Invalid action token '{part}'. Expected 'name=value' (e.g., 'noise=1.0,motion=0.0') or 'all'/'none'."
            )
        sep = "=" if "=" in part else ":"
        raw_key, raw_val = [s.strip() for s in part.split(sep, 1)]
        canonical = SHORT_ALIASES.get(raw_key.lower(), raw_key.lower())
        if canonical not in type_to_idx:
            valid = ", ".join(degradation_types)
            raise ValueError(f"Unknown degradation '{raw_key}'. Valid names: {valid}")
        val = float(raw_val)
        vec[type_to_idx[canonical]] = float(np.clip(val, 0.0, 1.0))

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
) -> np.ndarray:
    """Extract a soft object mask M in [0, 1]^(H, W) from a single (x, y) click.

    Uses the LoRA-adapted C-RADIO / DINOv3+SigLIP2 spatial feature grid so object
    boundaries remain accurate even under heavy noise, dark exposure, or blur.

    Args:
        semantic_encoder: SemanticEncoder module
        rgb_01: [1, 3, H, W] tensor in [0, 1]
        point_xy: (x, y) either in normalized [0, 1] coords or pixel coords
        threshold: Optional cosine similarity threshold in [-1, 1]; if None, uses Otsu
        feather_sigma: Gaussian/bilateral edge softening radius in pixels

    Returns:
        mask_2d: float32 numpy array of shape (H, W) in [0, 1]
    """
    _, _, h, w = rgb_01.shape
    px, py = float(point_xy[0]), float(point_xy[1])
    if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0 and (px < 1.0 or py < 1.0):
        norm_x, norm_y = px, py
    else:
        norm_x = np.clip(px / max(1.0, w - 1.0), 0.0, 1.0)
        norm_y = np.clip(py / max(1.0, h - 1.0), 0.0, 1.0)

    sem_out = semantic_encoder(rgb_01, use_lora=True)
    grid = F.normalize(sem_out["grid"].float(), dim=1)  # [1, D, Hp, Wp]

    # Sample reference feature at clicked coordinate via bilinear grid_sample
    sample_coord = torch.tensor([[[[norm_x * 2.0 - 1.0, norm_y * 2.0 - 1.0]]]], device=grid.device, dtype=torch.float32)
    anchor_vec = F.grid_sample(grid, sample_coord, mode="bilinear", align_corners=True)  # [1, D, 1, 1]
    anchor_vec = F.normalize(anchor_vec, dim=1)

    sim_lowres = (grid * anchor_vec).sum(dim=1, keepdim=True)  # [1, 1, Hp, Wp]
    sim_full = F.interpolate(sim_lowres, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    sim_np = sim_full.detach().cpu().numpy()

    # Normalize similarity to [0, 255] for clean Otsu / threshold segmentation
    s_min, s_max = float(sim_np.min()), float(sim_np.max())
    sim_01 = (sim_np - s_min) / max(1e-6, s_max - s_min)

    if threshold is None:
        sim_u8 = np.clip(sim_01 * 255.0, 0.0, 255.0).astype(np.uint8)
        otsu_val, binary_u8 = cv2.threshold(sim_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw_mask = (binary_u8.astype(np.float32) / 255.0)
        # Keep only the connected component containing the clicked pixel if valid
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
    """Construct [1, K, H, W] Pixel Overlay Sheet from global or foreground/background actions."""
    k = len(fg_action)
    if mask_2d is None:
        overlay = np.broadcast_to(fg_action[:, None, None], (k, height, width)).copy().astype(np.float32)
    else:
        bg = np.zeros(k, dtype=np.float32) if bg_action is None else bg_action.astype(np.float32)
        m = np.clip(mask_2d, 0.0, 1.0).astype(np.float32)[None, :, :]
        overlay = m * fg_action[:, None, None] + (1.0 - m) * bg[:, None, None]
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

    # Draw contour around 0.5 isoline
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
