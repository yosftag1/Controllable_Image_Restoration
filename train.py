"""2-Stage Multi-GPU AMP Training Script for Controllable Pretrained Restoration."""

import argparse
from contextlib import nullcontext
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import warnings

if "/kaggle/working/Controllable_Image_Restoration" not in sys.path:
    sys.path.insert(0, "/kaggle/working/Controllable_Image_Restoration")
if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True":
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
_torch_kcache = os.environ.get("PYTORCH_KERNEL_CACHE_PATH", "/tmp/torch_kernels")
try:
    os.makedirs(_torch_kcache, exist_ok=True)
    os.environ.setdefault("PYTORCH_KERNEL_CACHE_PATH", _torch_kcache)
except OSError:
    pass
if os.environ.get("RANK", "0") != "0":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm(\..*)?")

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

cv2.setNumThreads(0)

try:
    import pynvml

    pynvml.nvmlInit()
    _HAS_PYNVML = True
except Exception:
    _HAS_PYNVML = False

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config, DebugConfig, PilotConfig
from data.selective_dataset import (
    SelectiveDegradationPipeline,
    create_controllable_dataset,
    dataloader_worker_init_fn,
    unpack_batch_action_map,
)
from losses.controllable_losses import ControllableRestorationLoss, DINOv2FeatureDiscriminator
from models.controllable_restorer import ControllableRestorer
from utils.semantic_mask import build_spatial_action_overlay, parse_action_spec


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model: torch.nn.Module) -> ControllableRestorer:
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def cast_frozen_base_to_fp16(raw_model: ControllableRestorer, force_unet_fp16: bool = False):
    """Keep norm/custom Vision & VAE layers in FP32 so AMP autocast runs without Float/Half clashes."""
    if not force_unet_fp16:
        return
    groups = raw_model.get_stage_parameter_groups()
    trainable_ids = {id(p) for g in groups.values() for p in g}
    for mod in raw_model.unet.modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
            for p in mod.parameters(recurse=False):
                if id(p) not in trainable_ids and p.is_floating_point():
                    p.data = p.data.half()


def format_gpu_postfix() -> str:
    """Compact live GPU utilization & VRAM readout across all CUDA GPUs."""
    if not torch.cuda.is_available():
        return "CPU"
    parts = []
    for idx in range(torch.cuda.device_count()):
        util_str = ""
        if _HAS_PYNVML:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                util_str = f"{util}%"
            except Exception:
                util_str = ""
        try:
            free_b, total_b = torch.cuda.mem_get_info(idx)
            used_gb = (total_b - free_b) / (1024 ** 3)
            total_gb = total_b / (1024 ** 3)
            parts.append(
                f"G{idx}:{util_str}({used_gb:.1f}/{total_gb:.0f}G)"
                if util_str
                else f"G{idx}:{used_gb:.1f}/{total_gb:.0f}G"
            )
        except Exception:
            alloc_gb = torch.cuda.memory_reserved(idx) / (1024 ** 3)
            parts.append(f"G{idx}:{alloc_gb:.1f}G")
    return " ".join(parts)


def print_gpu_usage(header: str = "GPU Status"):
    if not torch.cuda.is_available():
        print(f"[{header}] Running on CPU")
        return
    print(f"── {header} ──")
    for idx in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(idx)
        try:
            free_b, total_b = torch.cuda.mem_get_info(idx)
            used_gb = (total_b - free_b) / (1024 ** 3)
            total_gb = total_b / (1024 ** 3)
            print(f"  GPU {idx} ({name}): {used_gb:.2f} / {total_gb:.2f} GB VRAM used")
        except Exception:
            print(f"  GPU {idx} ({name})")


def determine_epoch_stage(epoch_1based: int, config: Config) -> str:
    """Resolve 2-stage training phase for 1-based epoch index."""
    if config.TRAINING_MODE.lower() == "joint":
        return "joint"
    if epoch_1based <= config.PIXEL_STAGE_EPOCHS:
        return "pixel_stage"
    return "semantic_stage"


def compute_curriculum_boost(epoch_1based: int, stage: str, config: Config) -> float:
    """Single-to-compound curriculum boost in early pixel_stage epochs for crisp per-slider grounding."""
    if stage not in ("pixel_stage", "joint"):
        return 0.0
    warmup_ep = int(getattr(config, "CURRICULUM_WARMUP_EPOCHS", 0) or 0)
    if warmup_ep <= 0:
        return 0.0
    rel_idx = epoch_1based - 1
    if rel_idx < 0 or rel_idx >= warmup_ep:
        return 0.0
    return float(max(0.0, 0.80 * (1.0 - float(rel_idx) / float(warmup_ep))))


def build_stage_optimizer(raw_model: ControllableRestorer, stage: str, config: Config):
    """Configure trainable parameters and build AdamW optimizer for the active stage."""
    raw_model.set_training_stage(stage)
    groups = raw_model.get_stage_parameter_groups()

    param_groups = []
    if stage == "pixel_stage":
        param_groups.append({"params": groups["pixel_stage"], "lr": config.LEARNING_RATE})
    elif stage == "semantic_stage":
        param_groups.append({"params": groups["semantic_stage"], "lr": config.LEARNING_RATE * 1.25})
    elif stage == "joint":
        param_groups.append(
            {"params": groups["pixel_stage"] + groups["semantic_stage"], "lr": config.LEARNING_RATE}
        )

    optimizer = optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    active_count = sum(p.numel() for pg in param_groups for p in pg["params"] if p.requires_grad)
    return optimizer, active_count


def _per_sample_psnr(pred_01: torch.Tensor, tgt_01: torch.Tensor, max_db: float = 55.0) -> torch.Tensor:
    min_mse = 10.0 ** (-float(max_db) / 10.0)
    mse = torch.mean((pred_01.float() - tgt_01.float()) ** 2, dim=(1, 2, 3)).clamp(min=min_mse)
    return -10.0 * torch.log10(mse)


def compute_psnr(pred_01: torch.Tensor, tgt_01: torch.Tensor) -> float:
    return float(_per_sample_psnr(pred_01, tgt_01).mean().item())


def _format_duration(seconds: float) -> str:
    secs = max(0, int(round(seconds)))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


_INLINE_IMG_PREFIX = "__INLINE_SWEEP__::"


def _in_ipython_kernel() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return False
        return "IPKernelApp" in getattr(ip, "config", {}) or ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _display_sweep_in_ipython(image_path: str, caption: str = "", max_width: int = 1800) -> bool:
    if not _in_ipython_kernel():
        return False
    try:
        from IPython.display import HTML, Image, display

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            return False
        h, w = bgr.shape[:2]
        if max_width > 0 and w > max_width:
            scale = float(max_width) / float(w)
            bgr = cv2.resize(bgr, (max_width, max(64, int(round(h * scale)))), interpolation=cv2.INTER_AREA)

        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return False
        if caption:
            display(
                HTML(
                    f"<div style='margin:8px 0 3px 0;font-family:monospace;font-size:12px;"
                    f"color:#b8f2d8;background:#1b242c;padding:4px 10px;border-radius:4px;"
                    f"display:inline-block;'><b>{caption}</b></div>"
                )
            )
        display(Image(data=enc.tobytes(), format="jpeg"))
        return True
    except Exception:
        return False


def _emit_inline_sweep(image_path: str, caption: str, config: Config) -> None:
    if not bool(getattr(config, "DISPLAY_INLINE_IMAGES", True)):
        return
    max_w = int(getattr(config, "INLINE_IMAGE_MAX_WIDTH", 1800))
    if _in_ipython_kernel():
        _display_sweep_in_ipython(image_path, caption=caption, max_width=max_w)
    elif os.environ.get("AGY_PARENT_IPYTHON") == "1":
        safe_cap = caption.replace("\n", " ").replace("\r", " ").replace("::", " - ")
        sys.stdout.write(f"{_INLINE_IMG_PREFIX}{safe_cap}::{os.path.abspath(str(image_path))}\n")
        sys.stdout.flush()


def _psnr_u8_tile(pred_u8: np.ndarray, ref_u8: np.ndarray) -> float:
    mse = float(np.mean((pred_u8.astype(np.float32) - ref_u8.astype(np.float32)) ** 2))
    if mse <= 1e-8:
        return 99.99
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def _sharpness_ratio_u8(pred_u8: np.ndarray, ref_u8: np.ndarray) -> float:
    """Ratio of Laplacian edge std (`pred / ref`) — 1.0x matches GT crispness, <0.6x means blurry."""
    p_gray = cv2.cvtColor(pred_u8, cv2.COLOR_RGB2GRAY)
    r_gray = cv2.cvtColor(ref_u8, cv2.COLOR_RGB2GRAY)
    p_std = float(cv2.Laplacian(p_gray, cv2.CV_32F).std())
    r_std = float(cv2.Laplacian(r_gray, cv2.CV_32F).std())
    return p_std / max(1e-4, r_std)


def _select_sharp_eval_crop(val_dataset, idx: int, crop_size: int = 320) -> np.ndarray:
    """Deterministically pick the sharpest textured `crop_size` window from a validation image."""
    crop_size = max(64, (int(crop_size) // 8) * 8)
    img_paths = getattr(val_dataset, "image_paths", None) or []
    if img_paths:
        try:
            rgb_full = val_dataset._decode_rgb(img_paths[idx % len(img_paths)])
            h, w = rgb_full.shape[:2]
            if h >= crop_size and w >= crop_size:
                ys = sorted({0, max(0, (h - crop_size) // 2), max(0, h - crop_size)})
                xs = sorted({0, max(0, (w - crop_size) // 2), max(0, w - crop_size)})
                best_crop = None
                best_var = -1.0
                for y in ys:
                    for x in xs:
                        cand = rgb_full[y : y + crop_size, x : x + crop_size]
                        v = val_dataset._estimate_patch_laplacian_var(cand)
                        if v > best_var:
                            best_var = v
                            best_crop = cand
                if best_crop is not None:
                    return best_crop.copy()
            return cv2.resize(rgb_full, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
        except Exception:
            pass
    fallback = val_dataset._load_crop_with_retry(idx)
    if fallback.shape[0] != crop_size or fallback.shape[1] != crop_size:
        return cv2.resize(fallback, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
    return fallback


@torch.no_grad()
def evaluate_validation(model, val_loader, device, config, stage: str, loss_fn=None):
    raw_model = unwrap_model(model)
    raw_model.eval()
    use_amp = config.USE_AMP and (device.type == "cuda")
    detail_scale = config.DEFAULT_DETAIL_SCALE if stage in ("semantic_stage", "joint") else 0.0

    total_psnr = 0.0
    total_l1 = 0.0
    total_dists = 0.0
    count = 0

    restore_psnr_sum = 0.0
    restore_gain_sum = 0.0
    restore_dists_sum = 0.0
    restore_count = 0

    for batch in val_loader:
        src = batch["source"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)
        act = batch["action_map"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = raw_model.sample(
                src_01=src,
                action_input=act,
                steps=1,
                fidelity_scale=1.0,
                detail_scale=detail_scale,
            )
            l1 = float(torch.mean(torch.abs(pred.float() - tgt.float())).item())
            dists_val = (
                float(loss_fn.perceptual(pred, tgt, use_dists_texture=True).detach().item())
                if loss_fn is not None
                else 0.0
            )

        bsz = src.shape[0]
        pred_psnrs = _per_sample_psnr(pred, tgt)
        in_psnrs = _per_sample_psnr(src, tgt)

        total_psnr += float(pred_psnrs.sum().item())
        total_l1 += l1 * bsz
        total_dists += dists_val * bsz
        count += bsz

        act_max = act.float().abs().amax(dim=(1, 2, 3))
        src_tgt_mse = torch.mean((src.float() - tgt.float()) ** 2, dim=(1, 2, 3))
        active_mask = (act_max > 0.20) & (src_tgt_mse > 2.5e-4)
        n_act = int(active_mask.sum().item())
        if n_act > 0:
            restore_psnr_sum += float(pred_psnrs[active_mask].sum().item())
            restore_gain_sum += float((pred_psnrs[active_mask] - in_psnrs[active_mask]).sum().item())
            restore_dists_sum += dists_val * n_act
            restore_count += n_act

    raw_model.clear_runtime_contexts()
    raw_model.train()
    denom = max(1, count)
    overall_psnr = total_psnr / denom
    restore_psnr = (restore_psnr_sum / restore_count) if restore_count > 0 else overall_psnr
    restore_gain = (restore_gain_sum / restore_count) if restore_count > 0 else 0.0
    val_dists = (restore_dists_sum / restore_count) if restore_count > 0 else (total_dists / denom)
    return {
        "val_psnr": overall_psnr,
        "val_restore_psnr": restore_psnr,
        "val_restore_gain": restore_gain,
        "val_l1": total_l1 / denom,
        "val_dists": val_dists,
    }


@torch.no_grad()
def save_controllable_validation_grid(
    model,
    val_dataset,
    device,
    config,
    epoch: int,
    stage: str,
    step: int = None,
    return_summary: bool = False,
):
    """Render high-res 6-col diagnostic sweep isolating Motion Deblur, Bokeh Defocus, Compound & Optics + Detail dial."""
    out_dir = Path(config.OUTPUT_DIR) / "controllable_sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_model = unwrap_model(model)
    raw_model.eval()
    use_amp = config.USE_AMP and (device.type == "cuda")
    signed_types = getattr(config, "SIGNED_ACTION_TYPES", ["exposure", "warmth"])
    crop_size = int(getattr(config, "EVAL_SWEEP_CROP_SIZE", 320))
    if getattr(config, "_debug_mode", False):
        crop_size = min(crop_size, int(config.PATCH_SIZE))

    pipeline = SelectiveDegradationPipeline(config)
    num_imgs = min(max(1, len(val_dataset)), max(1, config.EVAL_IMAGE_COUNT))
    rows_canvas = []
    row_log_parts = []

    for idx in range(num_imgs):
        clean_u8 = _select_sharp_eval_crop(val_dataset, idx, crop_size=crop_size)
        h, w = clean_u8.shape[:2]
        rng = np.random.default_rng(config.SEED + 1000 + idx)
        params = pipeline.sample_params(h, w, rng=rng)

        s_demo = np.zeros(config.NUM_ACTIONS, dtype=np.float32)
        scenario_id = idx % 4
        sr_key = "super_resolution" if "super_resolution" in pipeline.type_to_idx else "sharpen_focus"
        if scenario_id == 0:
            # Row 1: Pure curved motion blur — apples-to-apples detail_scale (0.0 vs 0.5 vs 1.0)
            row_tag = "Row1[PureMotion]"
            params.motion_spatial_mode = 0
            params.motion_pattern = 1
            params.motion_hdr_boost = 0.25
            s_demo[pipeline.type_to_idx["deblur_motion"]] = 0.70
            runs = [
                ("Denoise Only (Keep Blur)", "denoise=1.0", None, None, 1.0, 0.0),
                ("Deblur Pixel (d=0.0)", "deblur_motion=1.0", None, None, 1.0, 0.0),
                ("Deblur Detail (d=0.5)", "deblur_motion=1.0", None, None, 1.0, 0.50),
                ("Deblur Detail (d=1.0)", "deblur_motion=1.0", None, None, 1.0, 1.00),
            ]
        elif scenario_id == 1:
            # Row 2: Real-ESRGAN Blind Super-Resolution / Crisp Detail — apples-to-apples detail_scale
            row_tag = "Row2[SuperRes]"
            params.sr_mode = 0
            params.sr_scale_factor = 0.35
            params.sr_sigma_max = 1.4
            s_demo[pipeline.type_to_idx[sr_key]] = 0.80
            runs = [
                ("Denoise Only (Keep Soft)", "denoise=1.0", None, None, 1.0, 0.0),
                ("SR Pixel (d=0.0)", f"{sr_key}=1.0", None, None, 1.0, 0.0),
                ("SR Detail (d=0.5)", f"{sr_key}=1.0", None, None, 1.0, 0.50),
                ("SR Detail (d=1.0)", f"{sr_key}=1.0", None, None, 1.0, 1.00),
            ]
        elif scenario_id == 2:
            # Row 3: Heavy ISP Noise + Motion Blur + Dark Exposure (tests both exact & blind 'all' preset!)
            row_tag = "Row3[NoisyMotion+Dark]"
            params.motion_spatial_mode = 0
            params.noise_mode = 1
            s_demo[pipeline.type_to_idx["deblur_motion"]] = 0.65
            s_demo[pipeline.type_to_idx["denoise"]] = 0.75
            s_demo[pipeline.type_to_idx["exposure"]] = -0.65
            runs = [
                ("Denoise Only (Keep Blur)", "denoise=1.0", None, None, 1.0, 0.0),
                ("Exact Target (d=0.5)", "denoise=1.0,exposure=0.65,motion=1.0", None, None, 1.0, 0.50),
                ("Blind 'all'+Exp (d=0.5)", "all,exposure=0.65", None, None, 1.0, 0.50),
                ("Split Mask (L=Full R=Dn)", "denoise=1.0,exposure=0.65,motion=1.0", "denoise=1.0,exposure=0.65", "left_half", 1.0, 0.50),
            ]
        else:
            # Row 4: Banding/JPEG + Warm Cast + Haze (tests selective & blind 'all' preset)
            row_tag = "Row4[Band+Warm+Haze]"
            params.artifact_mode = 2
            s_demo[pipeline.type_to_idx["deartifact"]] = 0.78
            s_demo[pipeline.type_to_idx["warmth"]] = 0.65
            if "dehaze" in pipeline.type_to_idx:
                s_demo[pipeline.type_to_idx["dehaze"]] = 0.58
            full_spec = "deartifact=1.0,warmth=-0.65,dehaze=1.0" if "dehaze" in pipeline.type_to_idx else "deartifact=1.0,warmth=-0.65"
            runs = [
                ("Deartifact Only", "deartifact=1.0", None, None, 1.0, 0.0),
                ("+Cool WB (Keep Haze)", "deartifact=1.0,warmth=-0.65", None, None, 1.0, 0.0),
                ("Full Exact (d=0.5)", full_spec, None, None, 1.0, 0.50),
                ("Blind 'all'+Cool (d=0.5)", "all,warmth=-0.65", None, None, 1.0, 0.50),
            ]

        src_u8 = pipeline.render_with_params(clean_u8, s_demo, params)
        src_t = torch.from_numpy(src_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        in_db = _psnr_u8_tile(src_u8, clean_u8)
        in_shp = _sharpness_ratio_u8(src_u8, clean_u8)

        left_mask = np.zeros((h, w), dtype=np.float32)
        left_mask[:, : w // 2] = 1.0
        left_mask = cv2.GaussianBlur(left_mask, (0, 0), sigmaX=5.0, sigmaY=5.0)
        left_3d = left_mask[:, :, None]

        specs = [
            (f"{row_tag} GT", clean_u8, "Ground Truth | Shp 1.00x", (185, 200, 215)),
            ("Degraded Input", src_u8, f"{in_db:.1f}dB | Shp {in_shp:.2f}x", (225, 210, 170)),
        ]
        pix_full_db = in_db
        dual_full_db = in_db
        dual_full_shp = in_shp

        for label, fg_str, bg_str, mask_tag, f_scale, d_scale in runs:
            mask_2d = left_mask if mask_tag == "left_half" else None
            fg_vec = parse_action_spec(fg_str, config.DEGRADATION_TYPES, signed_types=signed_types)
            bg_vec = parse_action_spec(bg_str, config.DEGRADATION_TYPES, signed_types=signed_types) if bg_str else None
            overlay = build_spatial_action_overlay(h, w, fg_vec, bg_vec, mask_2d, device=device)
            eff_d = d_scale if stage in ("semantic_stage", "joint") else 0.0
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_t = raw_model.sample(
                    src_01=src_t,
                    action_input=overlay,
                    steps=1,
                    fidelity_scale=f_scale,
                    detail_scale=eff_d,
                )
            pred_u8 = np.clip(
                np.round(pred_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0), 0.0, 255.0
            ).astype(np.uint8)

            s_fg = pipeline.compute_target_state(s_demo, fg_vec)
            tgt_fg = clean_u8 if np.allclose(s_fg, 0.0, atol=1e-4) else pipeline.render_with_params(clean_u8, s_fg, params)
            if mask_tag == "left_half" and bg_vec is not None:
                s_bg = pipeline.compute_target_state(s_demo, bg_vec)
                tgt_bg = pipeline.render_with_params(clean_u8, s_bg, params)
                tgt_col = np.clip(np.round(left_3d * tgt_fg + (1.0 - left_3d) * tgt_bg), 0.0, 255.0).astype(np.uint8)
            else:
                tgt_col = tgt_fg

            p_db = _psnr_u8_tile(pred_u8, tgt_col)
            base_db = _psnr_u8_tile(src_u8, tgt_col)
            gain_db = p_db - base_db
            shp_r = _sharpness_ratio_u8(pred_u8, tgt_col)
            foot_txt = f"{p_db:.1f}dB ({gain_db:+.1f}) | Shp {shp_r:.2f}x"
            if gain_db >= 0.25:
                foot_col = (120, 245, 150)
            elif gain_db <= -0.15:
                foot_col = (255, 145, 120)
            else:
                foot_col = (200, 210, 220)

            if "(d=0.0)" in label:
                pix_full_db = p_db
            elif "(d=0.5)" in label:
                dual_full_db = p_db
                dual_full_shp = shp_r

            specs.append((label, pred_u8, foot_txt, foot_col))

        best_row_db = dual_full_db if (stage in ("semantic_stage", "joint") or pix_full_db == in_db) else pix_full_db
        row_log_parts.append(f"{row_tag}: {in_db:.1f}->{best_row_db:.1f}dB ({best_row_db - in_db:+.1f}, Shp {dual_full_shp:.2f}x)")

        pad = 8
        header_h = 38
        footer_h = 28
        row_w = len(specs) * (w + pad) + pad
        row_h = header_h + h + footer_h + pad
        panel = np.full((row_h, row_w, 3), 24, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        for c_idx, (col_title, tile_u8, foot_txt, foot_col) in enumerate(specs):
            x0 = pad + c_idx * (w + pad)
            cv2.putText(panel, col_title, (x0 + 5, header_h - 11), font, 0.48, (225, 242, 185), 1, cv2.LINE_AA)
            panel[header_h : header_h + h, x0 : x0 + w] = tile_u8
            cv2.rectangle(panel, (x0, header_h + h), (x0 + w, header_h + h + footer_h - 2), (16, 18, 22), -1)
            cv2.putText(panel, foot_txt, (x0 + 5, header_h + h + 19), font, 0.45, foot_col, 1, cv2.LINE_AA)

        rows_canvas.append(panel)

    full_grid = np.concatenate(rows_canvas, axis=0)
    bgr_grid = cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR)
    if step is not None:
        save_path = out_dir / f"epoch_{epoch:03d}_step_{step:05d}_{stage}.png"
    else:
        save_path = out_dir / f"epoch_{epoch:03d}_{stage}.png"
    cv2.imwrite(str(save_path), bgr_grid)
    cv2.imwrite(str(out_dir / "latest_preview.png"), bgr_grid)
    raw_model.clear_runtime_contexts()
    raw_model.train()
    if return_summary:
        return str(save_path), " | ".join(row_log_parts)
    return str(save_path)


def save_checkpoint(
    raw_model: ControllableRestorer,
    optimizer,
    scheduler,
    epoch: int,
    stage: str,
    best_psnr: float,
    config: Config,
    filename: str,
    disc_state_dict: dict = None,
    disc_opt_state_dict: dict = None,
):
    """Save lightweight unwrapped adapter checkpoint for resuming & inference."""
    ckpt_dir = Path(config.CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / filename
    payload = {
        "epoch": epoch,
        "stage": stage,
        "best_psnr": best_psnr,
        "trainable_state_dict": raw_model.trainable_state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "disc_state_dict": disc_state_dict,
        "disc_optimizer_state_dict": disc_opt_state_dict,
        "model_config": {
            "MODEL_VERSION": getattr(config, "MODEL_VERSION", "v7_sdturbo_crisp7"),
            "DEGRADATION_TYPES": list(config.DEGRADATION_TYPES),
            "SIGNED_ACTION_TYPES": list(getattr(config, "SIGNED_ACTION_TYPES", ["exposure", "warmth"])),
            "DETAIL_ACTION_TYPES": list(
                getattr(
                    config,
                    "DETAIL_ACTION_TYPES",
                    ["deblur_motion", "super_resolution", "deartifact", "dehaze"],
                )
            ),
            "DESTRUCTIVE_SKIP_ACTIONS": list(
                getattr(
                    config,
                    "DESTRUCTIVE_SKIP_ACTIONS",
                    ["deblur_motion", "super_resolution", "dehaze"],
                )
            ),
            "LORA_EXPERT_SPECS": [dict(sp) for sp in getattr(config, "LORA_EXPERT_SPECS", [])],
            "SEMANTIC_ENCODER_TYPE": config.SEMANTIC_ENCODER_TYPE,
            "DINOV2_REG_MODEL_ID": getattr(config, "DINOV2_REG_MODEL_ID", "facebook/dinov2-with-registers-base"),
            "GENERATOR_BACKBONE": config.GENERATOR_BACKBONE,
            "RESTORATION_TIMESTEPS": list(getattr(config, "RESTORATION_TIMESTEPS", (499,))),
            "PROMPT_SEQ_LEN": getattr(config, "PROMPT_SEQ_LEN", 77),
            "UNET_LORA_RANK": config.UNET_LORA_RANK,
            "UNET_LORA_ALPHA": config.UNET_LORA_ALPHA,
            "UNET_CONV_LORA_RANK": getattr(config, "UNET_CONV_LORA_RANK", config.UNET_LORA_RANK),
            "UNET_CONV_LORA_ALPHA": getattr(config, "UNET_CONV_LORA_ALPHA", config.UNET_LORA_ALPHA),
            "SEM_DETAIL_LORA_RANK": getattr(config, "SEM_DETAIL_LORA_RANK", 16),
            "SEM_DETAIL_LORA_ALPHA": getattr(config, "SEM_DETAIL_LORA_ALPHA", 16.0),
            "SEM_DETAIL_CONV_LORA_RANK": getattr(
                config, "SEM_DETAIL_CONV_LORA_RANK", getattr(config, "SEM_DETAIL_LORA_RANK", 12)
            ),
            "SEM_DETAIL_CONV_LORA_ALPHA": getattr(
                config, "SEM_DETAIL_CONV_LORA_ALPHA", getattr(config, "SEM_DETAIL_LORA_ALPHA", 12.0)
            ),
            "VAE_ENC_LORA_RANK": getattr(config, "VAE_ENC_LORA_RANK", 0),
            "VAE_ENC_LORA_ALPHA": getattr(config, "VAE_ENC_LORA_ALPHA", 0.0),
            "VAE_DEC_LORA_RANK": getattr(config, "VAE_DEC_LORA_RANK", 8),
            "VAE_DEC_LORA_ALPHA": getattr(config, "VAE_DEC_LORA_ALPHA", 8.0),
            "ACTION_HIDDEN_DIM": config.ACTION_HIDDEN_DIM,
            "SPATIAL_MOD_HIDDEN_DIM": config.SPATIAL_MOD_HIDDEN_DIM,
            "TONE_HEAD_HIDDEN_DIM": getattr(config, "TONE_HEAD_HIDDEN_DIM", 32),
            "PATCH_SIZE": int(getattr(config, "PATCH_SIZE", 384)),
        },
    }
    torch.save(payload, str(path))
    print(f"Checkpoint saved to {path}")
    return str(path)


def _detect_visible_gpu_count() -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        vis = vis.strip()
        if vis == "" or vis == "-1":
            return 0
        return len([x for x in vis.split(",") if x.strip()])
    if _HAS_PYNVML:
        try:
            return int(pynvml.nvmlDeviceGetCount())
        except Exception:
            pass
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _consume_ddp_output_line(line: str, max_width: int = 1800) -> bool:
    if line.startswith(_INLINE_IMG_PREFIX):
        payload = line[len(_INLINE_IMG_PREFIX) :].strip()
        caption, _, img_path = payload.partition("::")
        if img_path:
            _display_sweep_in_ipython(img_path, caption=caption, max_width=max_width)
        return True
    sys.stdout.write(line)
    sys.stdout.flush()
    return False


def _launch_ddp_subprocess(num_gpus: int, config: Config = None) -> None:
    """Spawn isolated multi-process DistributedDataParallel workers (ideal for Kaggle %run notebooks)."""
    sys.last_traceback = None
    sys.last_value = None
    gc.collect()

    inline_enabled = bool(getattr(config, "DISPLAY_INLINE_IMAGES", True)) if config is not None else True
    max_width = int(getattr(config, "INLINE_IMAGE_MAX_WIDTH", 1800)) if config is not None else 1800
    parent_ipython = inline_enabled and _in_ipython_kernel()

    env = os.environ.copy()
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.pop("PYTORCH_ALLOC_CONF", None)
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_KERNEL_CACHE_PATH", "/tmp/torch_kernels")
    if parent_ipython:
        env["AGY_PARENT_IPYTHON"] = "1"
    else:
        env.pop("AGY_PARENT_IPYTHON", None)
    env.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={num_gpus}",
        os.path.abspath(__file__),
        *sys.argv[1:],
    ]
    print(f"✓ Launching {num_gpus}-GPU DistributedDataParallel (DDP) training processes...", flush=True)
    if inline_enabled and not parent_ipython and os.path.exists("/kaggle"):
        print(
            "[Tip] Running via shell (!python) prints text logs only — use '%run train.py ...' "
            "in a Kaggle cell to render live sweep images inline in the notebook logs!",
            flush=True,
        )
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    try:
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        pending = ""
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            text = pending + chunk.decode("utf-8", errors="replace")
            lines = text.splitlines(keepends=True)
            if lines and not (lines[-1].endswith("\n") or lines[-1].endswith("\r")):
                pending = lines.pop()
            else:
                pending = ""
            for line in lines:
                _consume_ddp_output_line(line, max_width=max_width)
        if pending:
            _consume_ddp_output_line(pending, max_width=max_width)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"DistributedDataParallel training exited with code {rc}")
    except KeyboardInterrupt:
        print("\n[Multi-GPU] Interrupted — shutting down DDP workers cleanly...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise


class _EpochStepCapSampler(Sampler):
    """Deterministic per-epoch sampler supporting both step-floor expansion and step-ceiling caps."""

    def __init__(
        self,
        dataset_len: int,
        num_steps: int,
        per_gpu_bsz: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.dataset_len = int(dataset_len)
        self.num_steps = max(1, int(num_steps))
        self.per_gpu_bsz = max(1, int(per_gpu_bsz))
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = self.num_steps * self.per_gpu_bsz
        self.total_size = self.num_samples * self.world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.total_size <= self.dataset_len:
            perm = torch.randperm(self.dataset_len, generator=g)[: self.total_size].tolist()
        else:
            reps = (self.total_size + self.dataset_len - 1) // self.dataset_len
            perm = torch.cat([torch.randperm(self.dataset_len, generator=g) for _ in range(reps)])[
                : self.total_size
            ].tolist()
        return iter(perm[self.rank : self.total_size : self.world_size])

    def __len__(self) -> int:
        return self.num_samples


def _build_train_loader(
    train_dataset,
    config: Config,
    device: torch.device,
    is_ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    """Construct high-throughput DataLoader with virtual step floor & DDP support."""
    train_dataset.align_only = False
    train_dataset.compact_ipc = True
    if world_size > 1:
        train_dataset._cache_capacity = max(4, int(getattr(config, "DECODE_CACHE_SIZE", 16)) // world_size)

    is_debug = bool(getattr(config, "_debug_mode", False))
    global_bsz = config.BATCH_SIZE
    grad_accum = max(1, int(getattr(config, "GRAD_ACCUM_STEPS", 1) or 1))
    per_gpu_bsz = max(1, global_bsz // max(1, world_size))
    total_nw = int(config.NUM_WORKERS)
    nw = max(1, total_nw // max(1, world_size)) if total_nw > 0 else 0

    max_opt_steps = int(getattr(config, "MAX_STEPS_PER_EPOCH", 0) or 0)
    min_opt_steps = int(getattr(config, "MIN_STEPS_PER_EPOCH", 0) or 0) if not is_debug else 0
    max_micro = max_opt_steps * grad_accum if max_opt_steps > 0 else 0
    min_micro = min_opt_steps * grad_accum if min_opt_steps > 0 else 0
    full_micro = max(1, len(train_dataset) // max(1, global_bsz))

    target_steps = 0
    if max_micro > 0 and max_micro < full_micro:
        target_steps = max_micro
    elif min_micro > 0 and full_micro < min_micro:
        target_steps = min_micro
        if max_micro > 0:
            target_steps = min(target_steps, max_micro)

    sampler = None
    if target_steps > 0:
        sampler = _EpochStepCapSampler(
            dataset_len=len(train_dataset),
            num_steps=target_steps,
            per_gpu_bsz=per_gpu_bsz,
            rank=rank,
            world_size=world_size,
            seed=config.SEED,
        )
    elif is_ddp:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.SEED,
            drop_last=(len(train_dataset) >= global_bsz),
        )

    loader_kwargs = {
        "batch_size": per_gpu_bsz,
        "shuffle": (sampler is None),
        "sampler": sampler,
        "num_workers": nw,
        "pin_memory": (device.type == "cuda"),
        "drop_last": (sampler is None and len(train_dataset) >= per_gpu_bsz),
    }
    if nw > 0:
        loader_kwargs["worker_init_fn"] = dataloader_worker_init_fn
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(getattr(config, "DATALOADER_PREFETCH_FACTOR", 4))
    return DataLoader(train_dataset, **loader_kwargs), sampler


def train(
    config: Config,
    resume_path: str = None,
    debug: bool = False,
    batch_size_overridden: bool = False,
    speedrun: bool = False,
    extra_epochs: int = None,
    remaining_pixel_epochs: int = None,
    epochs_overridden: bool = False,
    pixel_epochs_overridden: bool = False,
    allow_schema_expand: bool = False,
):
    is_ddp = (
        "LOCAL_RANK" in os.environ
        and "RANK" in os.environ
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
    )
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        is_main = rank == 0
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        try:
            dist.init_process_group(backend="nccl", device_id=device, timeout=timedelta(minutes=30))
        except TypeError:
            dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        is_main = True
        device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    try:
        set_seed(config.SEED)
        config._debug_mode = debug
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True

        num_gpus = world_size if is_ddp else (torch.cuda.device_count() if device.type == "cuda" else 0)
        use_amp = config.USE_AMP and (device.type == "cuda")

        if num_gpus == 1 and not batch_size_overridden and config.BATCH_SIZE >= 8 and not debug:
            config.BATCH_SIZE = 4
        grad_accum = max(1, int(getattr(config, "GRAD_ACCUM_STEPS", 1) or 1))

        if is_main:
            mode_tag = f"DDP ({world_size} GPUs)" if is_ddp else f"{device} (GPUs: {num_gpus})"
            gckpt_tag = "ON" if getattr(config, "GRADIENT_CHECKPOINTING", True) else "OFF"
            t_steps = list(getattr(config, "RESTORATION_TIMESTEPS", (499,)))
            t_tag = f"{len(t_steps)}step_t={t_steps}"
            exp_names = [sp.get("name", "?") for sp in getattr(config, "LORA_EXPERT_SPECS", [])]
            exp_tag = str(exp_names) if exp_names else f"SharedLean(lin={config.UNET_LORA_RANK},conv={getattr(config, 'UNET_CONV_LORA_RANK', config.UNET_LORA_RANK)})"
            eff_bsz = config.BATCH_SIZE * grad_accum
            print(
                f"Device: {mode_tag} | Patch: {config.PATCH_SIZE}px | Eff Batch: {eff_bsz} "
                f"({max(1, config.BATCH_SIZE // max(1, world_size))}/GPU x {grad_accum} accum) | "
                f"Backbone: {config.GENERATOR_BACKBONE} ({t_tag}) | LoRA: {exp_tag} | "
                f"Vision: {config.SEMANTIC_ENCODER_TYPE} | Controls: {config.NUM_ACTIONS} | "
                f"DINO_GAN_w: {getattr(config, 'LOSS_GAN_WEIGHT', 0.0)} | "
                f"AMP: {'ENABLED' if use_amp else 'DISABLED'} | GradCkpt: {gckpt_tag}",
                flush=True,
            )
            print("Loading datasets...", flush=True)

        train_dataset = create_controllable_dataset(config, is_train=True, compact_ipc=True, verbose=is_main)
        val_dataset = None
        val_loader = None
        if is_main:
            val_dataset = create_controllable_dataset(config, is_train=False, compact_ipc=False, verbose=True)
            val_nw = min(2, int(config.NUM_WORKERS))
            val_loader = DataLoader(
                val_dataset,
                batch_size=max(1, min(config.BATCH_SIZE // max(1, world_size), 4)),
                shuffle=False,
                num_workers=val_nw,
                pin_memory=(device.type == "cuda"),
                worker_init_fn=dataloader_worker_init_fn if val_nw > 0 else None,
            )

        if is_ddp:
            if is_main:
                raw_model = ControllableRestorer(config).to(device)
                loss_fn = ControllableRestorationLoss(config).to(device)
            dist.barrier()
            if not is_main:
                raw_model = ControllableRestorer(config).to(device)
                loss_fn = ControllableRestorationLoss(config).to(device)
            dist.barrier()
        else:
            raw_model = ControllableRestorer(config).to(device)
            loss_fn = ControllableRestorationLoss(config).to(device)

        if is_main:
            groups = raw_model.get_stage_parameter_groups()
            print(
                f"Trainable Adapter Breakdown -> "
                f"Pixel Stage (LoRA+SkipVAE+Refiner+Overlay): {sum(p.numel() for p in groups['pixel_stage']):,} | "
                f"Semantic Detail Stage: {sum(p.numel() for p in groups['semantic_stage']):,}",
                flush=True,
            )

        start_epoch = 1
        best_psnr = -1.0
        best_stage_score = -1e9
        resumed_stage = None
        resumed_opt_state = None
        resumed_sched_state = None
        resumed_disc_state = None
        resumed_disc_opt_state = None

        if resume_path:
            if os.path.exists(resume_path):
                if is_main:
                    print(f"Resuming from {resume_path}", flush=True)
                try:
                    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
                except TypeError:
                    ckpt = torch.load(resume_path, map_location=device)
                ckpt_cfg = ckpt.get("model_config") if isinstance(ckpt.get("model_config"), dict) else {}
                ckpt_degs = ckpt_cfg.get("DEGRADATION_TYPES")
                if ckpt_degs is not None and list(ckpt_degs) != list(config.DEGRADATION_TYPES):
                    if not allow_schema_expand:
                        raise RuntimeError(
                            f"Checkpoint at {resume_path} uses {len(ckpt_degs)}-slider schema {ckpt_degs}, "
                            f"which differs from current {len(config.DEGRADATION_TYPES)}-control schema "
                            f"{config.DEGRADATION_TYPES}. Pass --allow-schema-expand or start fresh without --resume!"
                        )
                    if is_main:
                        print(
                            f"[SchemaExpand] Warm-starting matching parameters from {ckpt_degs} -> "
                            f"{config.DEGRADATION_TYPES}",
                            flush=True,
                        )

                adapter_state = ckpt.get("trainable_state_dict") or ckpt.get("model_state_dict")
                if adapter_state is not None:
                    missing_keys, unexpected_keys = raw_model.load_trainable_state_dict(adapter_state)
                    if is_main and (missing_keys or unexpected_keys):
                        print(
                            f"[Warning] Partial adapter match on resume "
                            f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)}).",
                            flush=True,
                        )
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                last_epoch = start_epoch - 1
                best_psnr = float(ckpt.get("best_psnr", -1.0))
                best_stage_score = best_psnr if best_psnr > 0 else -1e9
                resumed_stage = ckpt.get("stage")
                resumed_opt_state = ckpt.get("optimizer_state_dict")
                resumed_sched_state = ckpt.get("scheduler_state_dict")
                resumed_disc_state = ckpt.get("disc_state_dict")
                resumed_disc_opt_state = ckpt.get("disc_optimizer_state_dict")
                del ckpt

                eff_rem_pix = remaining_pixel_epochs
                eff_extra = extra_epochs
                if speedrun:
                    if eff_rem_pix is None and not pixel_epochs_overridden:
                        if resumed_stage == "pixel_stage":
                            eff_rem_pix = 2
                        elif resumed_stage == "semantic_stage":
                            eff_rem_pix = 0
                        else:
                            eff_rem_pix = 5
                    if eff_extra is None and not epochs_overridden:
                        if resumed_stage == "pixel_stage":
                            eff_extra = int(eff_rem_pix or 0) + 2
                        elif resumed_stage == "semantic_stage":
                            eff_extra = 2
                        else:
                            eff_extra = int(eff_rem_pix or 5) + 2

                if eff_rem_pix is not None:
                    config.PIXEL_STAGE_EPOCHS = max(0, last_epoch + int(eff_rem_pix))
                if eff_extra is not None:
                    config.EPOCHS = last_epoch + max(1, int(eff_extra))
                    resumed_sched_state = None

                if resumed_stage != determine_epoch_stage(start_epoch, config):
                    resumed_opt_state = None
                    resumed_sched_state = None
                if is_main:
                    sched_note = ""
                    if speedrun or eff_rem_pix is not None or eff_extra is not None:
                        sched_note = (
                            f" | schedule -> total_epochs={config.EPOCHS}, "
                            f"pixel_end={config.PIXEL_STAGE_EPOCHS}, "
                            f"steps_cap={getattr(config, 'MAX_STEPS_PER_EPOCH', 0) or 'full'}"
                        )
                    print(
                        f"Resumed trainable adapters (starting at Epoch {start_epoch}, "
                        f"best PSNR={best_psnr:.2f} dB){sched_note}",
                        flush=True,
                    )
            else:
                if speedrun:
                    if not pixel_epochs_overridden:
                        config.PIXEL_STAGE_EPOCHS = 5
                    if not epochs_overridden:
                        config.EPOCHS = 7
                if is_main:
                    print(f"No checkpoint found at {resume_path} — starting fresh training from Epoch 1.", flush=True)

        if is_ddp:
            set_seed(config.SEED + rank * 1009)

        model = raw_model
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
        if is_main:
            print_gpu_usage("Pre-Training GPU & VRAM Check")

        current_stage = None
        train_loader, train_sampler = _build_train_loader(
            train_dataset,
            config,
            device,
            is_ddp=is_ddp,
            rank=rank,
            world_size=world_size,
        )
        optimizer = None
        scheduler = None
        raw_disc = None
        disc_model = None
        disc_opt = None
        sem_step_counter = 0
        history_file = Path(config.LOG_DIR) / "training_history.json"
        history = []
        if is_main:
            os.makedirs(config.LOG_DIR, exist_ok=True)
            if start_epoch > 1 and history_file.exists():
                try:
                    history = json.loads(history_file.read_text(encoding="utf-8"))
                except Exception:
                    history = []

        for epoch in range(start_epoch, config.EPOCHS + 1):
            stage = determine_epoch_stage(epoch, config)
            cur_boost = compute_curriculum_boost(epoch, stage, config)
            train_dataset.set_curriculum_boost(cur_boost)
            train_dataset.set_detail_stage_focus(stage == "semantic_stage")

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if stage != current_stage:
                prev_stage = current_stage if current_stage is not None else resumed_stage
                if prev_stage == "pixel_stage" and stage == "semantic_stage":
                    if is_main:
                        best_ckpt_p = Path(config.CHECKPOINT_DIR) / "model_best.pt"
                        pix_backup_p = Path(config.CHECKPOINT_DIR) / "model_best_pixel.pt"
                        if best_ckpt_p.exists():
                            shutil.copy2(str(best_ckpt_p), str(pix_backup_p))
                            print(
                                f"Backed up peak Pixel-Stage PSNR checkpoint ({best_psnr:.2f} dB) -> {pix_backup_p}",
                                flush=True,
                            )
                    best_psnr = -1.0
                    best_stage_score = -1e9

                if is_ddp and model is not raw_model:
                    del model
                    gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                current_stage = stage
                optimizer, active_params = build_stage_optimizer(raw_model, stage, config)
                if stage == "pixel_stage":
                    stage_rem = max(1, config.PIXEL_STAGE_EPOCHS - epoch + 1)
                else:
                    stage_rem = max(1, config.EPOCHS - epoch + 1)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_rem, eta_min=1e-5)
                if device.type == "cuda":
                    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

                if is_main:
                    print(
                        f"\n=== Entering Stage '{stage}' at Epoch {epoch}/{config.EPOCHS} "
                        f"({active_params:,} active trainable params) ===",
                        flush=True,
                    )

                if epoch == start_epoch and resumed_stage == stage and resumed_opt_state is not None:
                    try:
                        base_lrs = [pg["lr"] for pg in optimizer.param_groups]
                        optimizer.load_state_dict(resumed_opt_state)
                        sched_exhausted = bool(
                            resumed_sched_state is not None
                            and int(resumed_sched_state.get("last_epoch", 0))
                            >= int(resumed_sched_state.get("T_max", 1))
                        )
                        if resumed_sched_state is not None and not sched_exhausted:
                            scheduler.load_state_dict(resumed_sched_state)
                            if is_main:
                                print(f"Restored optimizer & scheduler states for stage '{stage}'.", flush=True)
                        else:
                            for pg, blr in zip(optimizer.param_groups, base_lrs):
                                pg["lr"] = blr * 0.85
                                pg["initial_lr"] = blr * 0.85
                            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                                optimizer, T_max=stage_rem, eta_min=1e-5
                            )
                            if is_main:
                                print(
                                    f"Restored warm AdamW states with fresh {stage_rem}-epoch cosine LR schedule "
                                    f"for extended '{stage}' run.",
                                    flush=True,
                                )
                    except Exception as exc:
                        if is_main:
                            print(f"[Note] Optimizer state re-initialized cleanly for stage '{stage}' ({exc}).", flush=True)
                    resumed_opt_state = None
                    resumed_sched_state = None

                if is_ddp:
                    model = torch.nn.parallel.DistributedDataParallel(
                        raw_model,
                        device_ids=[local_rank],
                        output_device=local_rank,
                        find_unused_parameters=False,
                    )
                else:
                    model = raw_model

                # Setup or tear down zero-extra-VRAM Frozen-DINOv2 Multi-Layer Feature Discriminator (Stage 2)
                gan_weight = float(getattr(config, "LOSS_GAN_WEIGHT", 0.15))
                if stage in ("semantic_stage", "joint") and gan_weight > 0.0:
                    if raw_disc is None:
                        dino_dim = int(raw_model.semantic_encoder.spatial_dim)
                        raw_disc = DINOv2FeatureDiscriminator(feat_dim=dino_dim, num_scales=3).to(device)
                        disc_lr = float(getattr(config, "GAN_DISC_LR", 1.5e-4))
                        disc_opt = optim.AdamW(
                            raw_disc.parameters(), lr=disc_lr, betas=(0.0, 0.99), weight_decay=config.WEIGHT_DECAY
                        )
                        if epoch == start_epoch and resumed_stage == stage and resumed_disc_state is not None:
                            try:
                                raw_disc.load_state_dict(resumed_disc_state)
                                if resumed_disc_opt_state is not None:
                                    disc_opt.load_state_dict(resumed_disc_opt_state)
                                sem_step_counter = int(getattr(config, "GAN_WARMUP_STEPS", 50)) + 150
                                if is_main:
                                    print("Restored DINOv2 Feature Discriminator state for resumed semantic stage.", flush=True)
                            except Exception as dexc:
                                if is_main:
                                    print(f"[Note] Re-initialized DINOv2 Feature Discriminator cleanly ({dexc}).", flush=True)
                            resumed_disc_state = None
                            resumed_disc_opt_state = None
                        if is_ddp:
                            disc_model = torch.nn.parallel.DistributedDataParallel(
                                raw_disc,
                                device_ids=[local_rank],
                                output_device=local_rank,
                                broadcast_buffers=False,
                            )
                        else:
                            disc_model = raw_disc
                        if is_main:
                            d_params = sum(p.numel() for p in raw_disc.parameters())
                            print(
                                f"  [DINOv2-Disc] Frozen DINOv2 Multi-Layer Discriminator ({d_params:,} head params, "
                                f"gan_w={gan_weight}, warmup={getattr(config, 'GAN_WARMUP_STEPS', 50)} opt steps, clean HR targets only)",
                                flush=True,
                            )
                else:
                    if disc_model is not None or raw_disc is not None:
                        disc_model = None
                        raw_disc = None
                        disc_opt = None

            if is_main and cur_boost > 0.0:
                print(
                    f"  [Curriculum] Epoch {epoch} single-degradation grounding boost = {cur_boost:.2f}",
                    flush=True,
                )

            model.train()
            running = {}
            num_micro = len(train_loader)
            num_opt_steps = max(1, (num_micro + grad_accum - 1) // grad_accum)
            use_tqdm = bool(getattr(config, "USE_TQDM", False))
            log_interval = max(1, int(getattr(config, "LOG_INTERVAL", 100)))
            sample_every = max(0, int(getattr(config, "SAMPLE_EVERY_N_STEPS", 500)))
            step_width = max(4, len(str(num_opt_steps)))
            epoch_start_t = time.perf_counter()

            pbar = (
                tqdm(
                    train_loader,
                    desc=f"Epoch {epoch:03d}/{config.EPOCHS} [{stage}]",
                    file=sys.stdout,
                    ncols=118,
                    mininterval=10.0,
                )
                if (is_main and use_tqdm)
                else train_loader
            )

            csd_every = max(1, int(getattr(config, "CSD_EVERY_N_STEPS", 4)))
            gan_target_w = float(getattr(config, "LOSS_GAN_WEIGHT", 0.15))
            gan_warmup = max(0, int(getattr(config, "GAN_WARMUP_STEPS", 50)))
            dino_disc_max_sz = int(getattr(config, "DINO_DISC_MAX_SIZE", 252))
            det_idx_dev = raw_model.detail_action_indices.to(device)

            optimizer.zero_grad(set_to_none=True)
            if disc_opt is not None:
                disc_opt.zero_grad(set_to_none=True)
            opt_step_idx = 0

            for micro_idx, batch in enumerate(pbar, start=1):
                is_sync_step = (micro_idx % grad_accum == 0) or (micro_idx == num_micro)
                cur_accum_div = (
                    grad_accum
                    if (micro_idx <= (num_micro // grad_accum) * grad_accum)
                    else max(1, num_micro % grad_accum)
                )

                src = batch["source"].to(device, non_blocking=True)
                tgt = batch["target"].to(device, non_blocking=True)
                act_map = unpack_batch_action_map(batch, device)
                if "gan_clean_eligible" in batch:
                    clean_gan_mask = batch["gan_clean_eligible"].to(device, non_blocking=True).float()
                else:
                    clean_gan_mask = torch.ones((src.shape[0],), device=device, dtype=torch.float32)

                pred_detached = None
                det_mask = None

                try:
                    # Stage 2 trains at FIXED `detail_scale = 1.0` (eliminates +38% random-scale overshoot!)
                    d_scale = 0.0 if stage == "pixel_stage" else 1.0
                    next_opt_step = opt_step_idx + 1
                    do_csd = (
                        is_sync_step
                        and (stage in ("semantic_stage", "joint"))
                        and (next_opt_step == 1 or (next_opt_step % csd_every == 0))
                    )
                    use_gan = (raw_disc is not None) and (stage in ("semantic_stage", "joint")) and (gan_target_w > 0.0)
                    if use_gan:
                        eff_counter = sem_step_counter + (1 if is_sync_step else 0)
                        if eff_counter <= gan_warmup:
                            eff_gan_w = 0.0
                        else:
                            eff_gan_w = gan_target_w * min(1.0, float(eff_counter - gan_warmup) / 100.0)
                        if det_idx_dev.numel() > 0:
                            det_mask = act_map.float().index_select(1, det_idx_dev).abs().amax(dim=1, keepdim=True).clamp(0.0, 1.0)
                        else:
                            det_mask = act_map.float().abs().amax(dim=1, keepdim=True).clamp(0.0, 1.0)
                    else:
                        eff_gan_w = 0.0

                    sync_ctx = model.no_sync() if (is_ddp and not is_sync_step) else nullcontext()
                    with sync_ctx:
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            out = model(
                                src,
                                act_map,
                                tgt_01=tgt,
                                fidelity_scale=1.0,
                                detail_scale=d_scale,
                            )
                            loss, metrics = loss_fn(
                                pred_01=out["pred"],
                                target_01=tgt,
                                action_map=act_map,
                                stage=stage,
                                src_01=src,
                                hat_z_0=out.get("hat_z_0"),
                                hat_z_1=out.get("hat_z_1"),
                                z_tgt=out.get("z_tgt"),
                                restorer_model=raw_model,
                                run_csd=do_csd,
                            )

                        if use_gan:
                            pred_detached = out["pred"].detach()
                            if eff_gan_w > 0.0:
                                for dp in raw_disc.parameters():
                                    dp.requires_grad = False
                                raw_disc.eval()
                                with torch.amp.autocast("cuda", enabled=use_amp):
                                    loss_g_gan = raw_disc.compute_g_loss(
                                        raw_model.semantic_encoder,
                                        out["pred"],
                                        detail_mask=det_mask,
                                        clean_sample_mask=clean_gan_mask,
                                        max_size=dino_disc_max_sz,
                                    )
                                loss = loss + eff_gan_w * loss_g_gan
                                metrics["g_gan"] = float(loss_g_gan.detach().item())
                                metrics["total"] = float(loss.detach().item())
                            else:
                                metrics["g_gan"] = 0.0

                        scaled_loss = loss / float(cur_accum_div)
                        if scaler is not None and use_amp:
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()

                    if is_sync_step:
                        if scaler is not None and use_amp:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                [p for p in raw_model.parameters() if p.requires_grad], config.GRAD_CLIP_NORM
                            )
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                [p for p in raw_model.parameters() if p.requires_grad], config.GRAD_CLIP_NORM
                            )
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        opt_step_idx += 1
                        if use_gan:
                            sem_step_counter += 1
                finally:
                    raw_model.clear_runtime_contexts()

                # DINOv2 Feature Discriminator update (accumulated across grad_accum micro-batches, 1:1 with G!)
                if raw_disc is not None and disc_opt is not None and pred_detached is not None:
                    for dp in raw_disc.parameters():
                        dp.requires_grad = True
                    disc_model.train()
                    d_sync_ctx = disc_model.no_sync() if (is_ddp and not is_sync_step) else nullcontext()
                    with d_sync_ctx:
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            loss_d, _, _ = DINOv2FeatureDiscriminator.compute_d_loss_with_module(
                                disc_model,
                                raw_model.semantic_encoder,
                                tgt,
                                pred_detached,
                                detail_mask=det_mask,
                                clean_sample_mask=clean_gan_mask,
                                max_size=dino_disc_max_sz,
                            )
                        (loss_d / float(cur_accum_div)).backward()
                    if is_sync_step:
                        torch.nn.utils.clip_grad_norm_(raw_disc.parameters(), 1.0)
                        disc_opt.step()
                        disc_opt.zero_grad(set_to_none=True)
                    metrics["d_loss"] = float(loss_d.detach().item())

                del out, pred_detached
                for k, v in metrics.items():
                    running[k] = running.get(k, 0.0) + v

                if is_sync_step and is_main and (
                    opt_step_idx == 1 or opt_step_idx % log_interval == 0 or opt_step_idx == num_opt_steps
                ):
                    avg_tot = running["total"] / micro_idx
                    gpu_str = format_gpu_postfix()
                    if use_tqdm:
                        pbar.set_postfix({"loss": f"{avg_tot:.4f}", "gpu": gpu_str})
                    else:
                        elapsed = max(1e-6, time.perf_counter() - epoch_start_t)
                        sec_per_it = elapsed / max(1, opt_step_idx)
                        rate_str = f"{sec_per_it:.2f}s/step" if sec_per_it >= 1.0 else f"{1.0 / sec_per_it:.2f}step/s"
                        eta_str = _format_duration(sec_per_it * (num_opt_steps - opt_step_idx))
                        pct = 100.0 * opt_step_idx / max(1, num_opt_steps)
                        cur_lr = optimizer.param_groups[-1]["lr"]
                        avg_l1 = running.get("l1", 0.0) / micro_idx
                        avg_lat = running.get("lat", 0.0) / micro_idx
                        avg_perc = running.get("perc", 0.0) / micro_idx
                        avg_edge = running.get("edge", 0.0) / micro_idx
                        if stage == "pixel_stage":
                            avg_pres = running.get("pres", 0.0) / micro_idx
                            loss_parts = (
                                f"loss={avg_tot:.4f} l1={avg_l1:.4f} lat={avg_lat:.4f} "
                                f"perc={avg_perc:.4f} edge={avg_edge:.4f} pres={avg_pres:.4f}"
                            )
                        else:
                            avg_ggan = running.get("g_gan", 0.0) / micro_idx
                            avg_dloss = running.get("d_loss", 0.0) / micro_idx
                            avg_csd = running.get("csd", 0.0) / micro_idx
                            gan_part = f" g_gan={avg_ggan:.3f} d={avg_dloss:.3f}" if raw_disc is not None else ""
                            csd_part = f" csd={avg_csd:.4f}" if getattr(config, "LOSS_CSD_WEIGHT", 0.0) > 0.0 else ""
                            loss_parts = (
                                f"loss={avg_tot:.4f} l1={avg_l1:.4f} lat={avg_lat:.4f} "
                                f"perc={avg_perc:.4f} edge={avg_edge:.4f}{gan_part}{csd_part}"
                            )
                        print(
                            f"[Epoch {epoch:03d}/{config.EPOCHS} | {stage}] "
                            f"{opt_step_idx:0{step_width}d}/{num_opt_steps} ({pct:5.1f}%) | "
                            f"{loss_parts} | lr={cur_lr:.2e} | {rate_str} ETA {eta_str} | {gpu_str}",
                            flush=True,
                        )

                should_mid_sample = (
                    is_sync_step
                    and sample_every > 0
                    and (opt_step_idx % sample_every == 0)
                    and (num_opt_steps - opt_step_idx) > min(75, sample_every // 5)
                )
                if should_mid_sample:
                    if is_main and val_dataset is not None:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        preview_path, row_summary = save_controllable_validation_grid(
                            raw_model, val_dataset, device, config, epoch, stage, step=opt_step_idx, return_summary=True
                        )
                        cap_str = f"Preview @ Epoch {epoch:03d}/{config.EPOCHS} Step {opt_step_idx:0{step_width}d}/{num_opt_steps} [{stage}]"
                        print(
                            f"  ↳ [{cap_str}] Saved sweep -> {preview_path}\n"
                            f"    {row_summary}",
                            flush=True,
                        )
                        _emit_inline_sweep(preview_path, caption=f"{cap_str} | {row_summary}", config=config)
                    if is_ddp:
                        dist.barrier()

            if scheduler is not None:
                scheduler.step()

            epoch_summary = {k: v / max(1, num_micro) for k, v in running.items()}
            if is_ddp and epoch_summary:
                keys_sorted = sorted(epoch_summary.keys())
                vec = torch.tensor([epoch_summary[k] for k in keys_sorted], device=device, dtype=torch.float64)
                dist.all_reduce(vec, op=dist.ReduceOp.AVG)
                for idx_k, k in enumerate(keys_sorted):
                    epoch_summary[k] = float(vec[idx_k].item())

            epoch_summary["epoch"] = epoch
            epoch_summary["stage"] = stage

            if is_main:
                if epoch % max(1, config.VAL_EPOCH_INTERVAL) == 0 or epoch == config.EPOCHS:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    val_metrics = evaluate_validation(raw_model, val_loader, device, config, stage, loss_fn=loss_fn)
                    epoch_summary.update(val_metrics)
                    grid_path, row_summary = save_controllable_validation_grid(
                        raw_model, val_dataset, device, config, epoch, stage, return_summary=True
                    )
                    rest_psnr = val_metrics.get("val_restore_psnr", val_metrics["val_psnr"])
                    rest_gain = val_metrics.get("val_restore_gain", 0.0)
                    val_dists = val_metrics.get("val_dists", 0.0)
                    print(
                        f"Epoch {epoch:03d} | train_loss={epoch_summary['total']:.4f} | "
                        f"restore_psnr={rest_psnr:.2f} dB ({rest_gain:+.2f} dB) | "
                        f"val_dists={val_dists:.4f} | val_psnr={val_metrics['val_psnr']:.2f} dB | "
                        f"val_l1={val_metrics['val_l1']:.4f} | sweep={grid_path}\n"
                        f"  ↳ {row_summary}",
                        flush=True,
                    )
                    _emit_inline_sweep(
                        grid_path,
                        caption=(
                            f"Epoch {epoch:03d}/{config.EPOCHS} [{stage}] | "
                            f"restore_psnr={rest_psnr:.2f} dB ({rest_gain:+.2f} dB) | "
                            f"dists={val_dists:.4f} | {row_summary}"
                        ),
                        config=config,
                    )

                    cur_score = (rest_psnr - 12.0 * val_dists) if stage == "semantic_stage" else rest_psnr
                    if cur_score > best_stage_score:
                        best_stage_score = cur_score
                        best_psnr = rest_psnr
                        save_checkpoint(raw_model, optimizer, scheduler, epoch, stage, best_psnr, config, "model_best.pt")

                disc_sd = {k: v.detach().cpu() for k, v in raw_disc.state_dict().items()} if raw_disc is not None else None
                disc_osd = disc_opt.state_dict() if disc_opt is not None else None
                save_checkpoint(
                    raw_model,
                    optimizer,
                    scheduler,
                    epoch,
                    stage,
                    best_psnr,
                    config,
                    "model_latest.pt",
                    disc_state_dict=disc_sd,
                    disc_opt_state_dict=disc_osd,
                )
                if epoch % config.CHECKPOINT_EVERY_N_EPOCHS == 0:
                    save_checkpoint(
                        raw_model, optimizer, scheduler, epoch, stage, best_psnr, config, f"model_epoch_{epoch:03d}.pt"
                    )

                history.append(epoch_summary)
                with open(history_file, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)

            if is_ddp:
                dist.barrier()
    except BaseException as fatal_err:
        if not isinstance(fatal_err, KeyboardInterrupt):
            import traceback

            sys.stdout.write(f"\n[Rank {rank} Fatal Error] {fatal_err}\n")
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            if is_ddp:
                os._exit(1)
        raise
    else:
        if is_ddp and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train Controllable Pretrained Image Restoration Model")
    parser.add_argument("--debug", action="store_true", help="Use DebugConfig for quick verification")
    parser.add_argument("--mock", action="store_true", help="Use offline mock_tiny backbones (no HF downloads)")
    parser.add_argument("--no-ddp", action="store_true", help="Disable automatic multi-GPU DDP worker launch")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root (DIV2K, LSDIR, or image folder)")
    parser.add_argument("--train-root", type=str, default=None, help="Clean reference folder(s), comma-separated allowed")
    parser.add_argument("--val-root", type=str, default=None, help="Validation reference image folder (e.g. DIV2K_valid)")
    parser.add_argument(
        "--paired-motion-root",
        type=str,
        default=None,
        help="Real motion-deblurring dataset root(s) (e.g. GoPro / HIDE / RealBlur curated Kaggle pack)",
    )
    parser.add_argument(
        "--paired-defocus-root",
        type=str,
        default=None,
        help="Real optical-defocus dataset root(s) (e.g. DPDD / RealDOF)",
    )
    parser.add_argument(
        "--paired-prob",
        type=float,
        default=None,
        help="Fraction of training batches sampled from real paired camera datasets (default: 0.42)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for sweep grids")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory for adapter checkpoints")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for training logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to model_latest.pt / model_best.pt checkpoint")
    parser.add_argument(
        "--allow-schema-expand",
        action="store_true",
        help="Allow warm-starting matching adapters when changing control sliders",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["sd-2.1-base", "sd-turbo", "sd-1.5", "bk-sdm-small", "tiny-sd", "bk-sdm-tiny", "mock_tiny"],
        default=None,
    )
    parser.add_argument(
        "--vision-encoder",
        type=str,
        choices=["dinov2_reg", "mock_tiny"],
        default=None,
    )
    parser.add_argument("--training-mode", type=str, choices=["staged", "joint"], default=None)
    parser.add_argument("--patch-size", type=int, default=None, help="Training crop resolution (e.g. 256, 384, 512)")
    parser.add_argument("--batch-size", type=int, default=None, help="Global micro-batch size across all GPUs")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps (eff batch = batch_size * grad_accum)")
    parser.add_argument("--timestep", type=int, default=None, help="Override 1-step diffusion timestep (default: 499 for sd-turbo)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--pixel-epochs", type=int, help="Override Pixel-Stage epochs", default=None)
    parser.add_argument("--extra-epochs", type=int, default=None, help="When resuming, run N more epochs past checkpoint")
    parser.add_argument("--remaining-pixel-epochs", type=int, default=None, help="When resuming, run K more pixel_stage epochs")
    parser.add_argument("--gan-weight", type=float, default=None, help="Override Stage-2 Frozen DINOv2 Discriminator weight (0=disable)")
    parser.add_argument("--csd-weight", type=float, default=None, help="Override Stage-2 Target-Anchored CSD loss weight")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Cap optimizer steps per epoch")
    parser.add_argument("--min-steps-per-epoch", type=int, default=None, help="Minimum optimizer steps per epoch on small folders")
    parser.add_argument("--train-images", type=int, default=None, help="Cap number of training reference images")
    parser.add_argument("--val-images", type=int, default=None, help="Cap number of validation reference images")
    parser.add_argument(
        "--pilot",
        "--5h",
        dest="pilot",
        action="store_true",
        help="Balanced ~5-hour 2-stage run on 2x T4 (9 pixel + 3 CSD epochs @ 950 steps/ep)",
    )
    parser.add_argument(
        "--speedrun",
        action="store_true",
        help="Quick sanity preset (350 steps/epoch, 5 pixel + 2 detail epochs)",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None, help="Steps between clean log lines (default: 100)")
    parser.add_argument("--sample-steps", type=int, default=None, help="Save mid-epoch sweep grid every N steps (0=disable)")
    parser.add_argument("--no-inline-images", action="store_true", help="Disable inline sweep image display in Kaggle/Jupyter")
    parser.add_argument("--tqdm", action="store_true", help="Use interactive tqdm progress bar instead of line logs")
    args = parser.parse_args()

    if args.debug:
        config = DebugConfig()
    elif args.pilot:
        config = PilotConfig()
    else:
        config = Config()

    if args.mock:
        config.GENERATOR_BACKBONE = "mock_tiny"
        config.SEMANTIC_ENCODER_TYPE = "mock_tiny"
    if args.backbone:
        config.GENERATOR_BACKBONE = args.backbone
    if args.vision_encoder:
        config.SEMANTIC_ENCODER_TYPE = args.vision_encoder
    if args.training_mode:
        config.TRAINING_MODE = args.training_mode
    if args.data_root:
        config.DATASET_PATH = args.data_root
    if args.train_root:
        config.TRAIN_DATASET_PATH = args.train_root
    if args.val_root:
        config.VAL_DATASET_PATH = args.val_root
    if args.paired_motion_root:
        config.PAIRED_MOTION_DATASET_PATH = args.paired_motion_root
    if args.paired_defocus_root:
        config.PAIRED_DEFOCUS_DATASET_PATH = args.paired_defocus_root
    if args.paired_prob is not None:
        config.PAIRED_REAL_PROB = float(np.clip(args.paired_prob, 0.0, 1.0))
        config.MOTION_REAL_PAIR_FRAC = config.PAIRED_REAL_PROB
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.checkpoint_dir:
        config.CHECKPOINT_DIR = args.checkpoint_dir
    if args.log_dir:
        config.LOG_DIR = args.log_dir
    if args.patch_size is not None:
        ps_aligned = max(64, (int(args.patch_size) // 8) * 8)
        config.PATCH_SIZE = ps_aligned
        config.EVAL_SWEEP_CROP_SIZE = ps_aligned
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.grad_accum is not None:
        config.GRAD_ACCUM_STEPS = max(1, int(args.grad_accum))
    if args.timestep is not None:
        config.RESTORATION_TIMESTEPS = (max(1, min(999, int(args.timestep))),)
    if args.speedrun:
        config.MAX_STEPS_PER_EPOCH = 300
        config.MIN_STEPS_PER_EPOCH = 300
        config.CURRICULUM_WARMUP_EPOCHS = 1
        config.LEARNING_RATE = 1.35e-4
        config.VAL_IMAGES_COUNT = 40
        config.LOG_INTERVAL = 40
        config.SAMPLE_EVERY_N_STEPS = 150
        if not args.resume:
            config.PIXEL_STAGE_EPOCHS = 5
            config.EPOCHS = 7
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.pixel_epochs is not None:
        config.PIXEL_STAGE_EPOCHS = args.pixel_epochs
    if args.gan_weight is not None:
        config.LOSS_GAN_WEIGHT = max(0.0, float(args.gan_weight))
    if args.csd_weight is not None:
        config.LOSS_CSD_WEIGHT = float(args.csd_weight)
    if args.steps_per_epoch is not None:
        config.MAX_STEPS_PER_EPOCH = args.steps_per_epoch
        if args.steps_per_epoch > 0:
            config.MIN_STEPS_PER_EPOCH = min(config.MIN_STEPS_PER_EPOCH, args.steps_per_epoch)
    if args.min_steps_per_epoch is not None:
        config.MIN_STEPS_PER_EPOCH = args.min_steps_per_epoch
    if args.train_images is not None:
        config.TRAIN_IMAGES_COUNT = args.train_images
    if args.val_images is not None:
        config.VAL_IMAGES_COUNT = args.val_images
    if args.num_workers is not None:
        config.NUM_WORKERS = args.num_workers
    if args.log_interval is not None:
        config.LOG_INTERVAL = args.log_interval
    if args.sample_steps is not None:
        config.SAMPLE_EVERY_N_STEPS = args.sample_steps
    if args.no_inline_images:
        config.DISPLAY_INLINE_IMAGES = False
    if args.tqdm:
        config.USE_TQDM = True

    in_ddp_worker = "LOCAL_RANK" in os.environ and "RANK" in os.environ
    if not in_ddp_worker and not args.no_ddp and config.MULTI_GPU and config.DEVICE == "cuda":
        visible_gpus = _detect_visible_gpu_count()
        if visible_gpus > 1:
            _launch_ddp_subprocess(visible_gpus, config=config)
            return

    train(
        config,
        resume_path=args.resume,
        debug=args.debug,
        batch_size_overridden=(args.batch_size is not None),
        speedrun=args.speedrun,
        extra_epochs=args.extra_epochs,
        remaining_pixel_epochs=args.remaining_pixel_epochs,
        epochs_overridden=(args.epochs is not None),
        pixel_epochs_overridden=(args.pixel_epochs is not None),
        allow_schema_expand=args.allow_schema_expand,
    )


if __name__ == "__main__":
    main()
