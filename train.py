"""3-Phase Multi-GPU AMP Training Script for Controllable Pretrained Restoration."""

import argparse
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

# Kaggle notebook %run path fix; ensure expandable_segments is NOT set (prevents multi-GPU T4 illegal address)
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

from config import Config, DebugConfig
from data.selective_dataset import (
    SelectiveDegradationPipeline,
    create_controllable_dataset,
    dataloader_worker_init_fn,
    unpack_batch_action_map,
)
from losses.controllable_losses import ControllableRestorationLoss, VisionAidedDiscriminator
from models.controllable_restorer import ControllableRestorer, SplitMultiGPURestorer
from utils.semantic_mask import build_spatial_action_overlay, parse_action_spec


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model: torch.nn.Module) -> ControllableRestorer:
    if isinstance(model, SplitMultiGPURestorer):
        return model.raw_model
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
    """Compact live GPU utilization & VRAM readout across all CUDA GPUs for tqdm."""
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
            parts.append(f"G{idx}:{util_str}({used_gb:.1f}/{total_gb:.0f}G)" if util_str else f"G{idx}:{used_gb:.1f}/{total_gb:.0f}G")
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
    """Resolve training stage for 1-based epoch index."""
    if config.TRAINING_MODE.lower() == "joint":
        return "joint"
    if epoch_1based <= config.ALIGN_WARMUP_EPOCHS:
        return "align_warmup"
    if epoch_1based <= config.ALIGN_WARMUP_EPOCHS + config.PIXEL_STAGE_EPOCHS:
        return "pixel_stage"
    return "semantic_stage"


def build_stage_optimizer(raw_model: ControllableRestorer, stage: str, config: Config):
    """Configure trainable parameters and build AdamW optimizer with distinct Vision-LoRA LR."""
    raw_model.set_training_stage(stage)
    groups = raw_model.get_stage_parameter_groups()

    param_groups = []
    if stage == "align_warmup":
        param_groups.append({"params": groups["vision_lora"], "lr": config.VISION_LORA_LR})
    elif stage == "pixel_stage":
        param_groups.append({"params": groups["vision_lora"], "lr": config.VISION_LORA_LR * 0.5})
        param_groups.append({"params": groups["pixel_stage"], "lr": config.LEARNING_RATE})
    elif stage == "semantic_stage":
        param_groups.append({"params": groups["semantic_stage"], "lr": config.LEARNING_RATE})
    elif stage == "joint":
        param_groups.append({"params": groups["vision_lora"], "lr": config.VISION_LORA_LR})
        param_groups.append(
            {"params": groups["pixel_stage"] + groups["semantic_stage"], "lr": config.LEARNING_RATE}
        )

    optimizer = optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    active_count = sum(p.numel() for pg in param_groups for p in pg["params"] if p.requires_grad)
    return optimizer, active_count


def _per_sample_psnr(pred_01: torch.Tensor, tgt_01: torch.Tensor, max_db: float = 55.0) -> torch.Tensor:
    """Compute per-image PSNR capped at `max_db` (55 dB ~ sub-8-bit floor) so identity tasks never spike 100 dB."""
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


@torch.no_grad()
def evaluate_validation(model, val_loader, device, config, stage: str):
    raw_model = unwrap_model(model)
    raw_model.eval()
    use_amp = config.USE_AMP and (device.type == "cuda")
    detail_scale = config.DEFAULT_DETAIL_SCALE if stage in ("semantic_stage", "joint") else 0.0

    total_psnr = 0.0
    total_l1 = 0.0
    count = 0

    restore_psnr_sum = 0.0
    restore_gain_sum = 0.0
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

        bsz = src.shape[0]
        pred_psnrs = _per_sample_psnr(pred, tgt)
        in_psnrs = _per_sample_psnr(src, tgt)

        total_psnr += float(pred_psnrs.sum().item())
        total_l1 += l1 * bsz
        count += bsz

        # Track real non-trivial restoration PSNR & dB gain separately from identity/no-op samples
        act_max = act.float().amax(dim=(1, 2, 3))
        src_tgt_mse = torch.mean((src.float() - tgt.float()) ** 2, dim=(1, 2, 3))
        active_mask = (act_max > 0.20) & (src_tgt_mse > 2.5e-4)
        n_act = int(active_mask.sum().item())
        if n_act > 0:
            restore_psnr_sum += float(pred_psnrs[active_mask].sum().item())
            restore_gain_sum += float((pred_psnrs[active_mask] - in_psnrs[active_mask]).sum().item())
            restore_count += n_act

    raw_model.clear_runtime_contexts()
    raw_model.train()
    denom = max(1, count)
    overall_psnr = total_psnr / denom
    restore_psnr = (restore_psnr_sum / restore_count) if restore_count > 0 else overall_psnr
    restore_gain = (restore_gain_sum / restore_count) if restore_count > 0 else 0.0
    return {
        "val_psnr": overall_psnr,
        "val_restore_psnr": restore_psnr,
        "val_restore_gain": restore_gain,
        "val_l1": total_l1 / denom,
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
):
    """Render visual sweep covering all 8 degradations across 3 validation rows."""
    out_dir = Path(config.OUTPUT_DIR) / "controllable_sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_model = unwrap_model(model)
    raw_model.eval()
    use_amp = config.USE_AMP and (device.type == "cuda")

    pipeline = SelectiveDegradationPipeline(config)
    num_imgs = min(len(val_dataset), max(1, config.EVAL_IMAGE_COUNT))
    rows_canvas = []

    for idx in range(num_imgs):
        clean_u8 = val_dataset._load_crop_with_retry(idx)
        h, w = clean_u8.shape[:2]
        rng = np.random.default_rng(config.SEED + 1000 + idx)
        params = pipeline.sample_params(h, w, rng=rng)

        s_demo = np.zeros(config.NUM_ACTIONS, dtype=np.float32)
        scenario_id = idx % 3
        if scenario_id == 0:
            # Row 0: Motion Blur + Sensor Noise + Dark Underexposure
            params.underexpose = True
            s_demo[pipeline.type_to_idx["motion_blur"]] = 0.80
            s_demo[pipeline.type_to_idx["gaussian_noise"]] = 0.75
            s_demo[pipeline.type_to_idx["exposure_issue"]] = 0.65
            runs = [
                ("Identity (a=0)", "none", None, None, 1.0, 0.0),
                ("Denoise (Keep Blur)", "noise=1.0,motion=0.0,exposure=0.0", None, None, 1.0, 0.0),
                ("Denoise+Exp (Keep Blur)", "noise=1.0,exposure=1.0,motion=0.0", None, None, 1.0, 0.0),
                ("50% Partial Deblur", "noise=1.0,exposure=1.0,motion=0.5", None, None, 1.0, 0.5),
                ("Full (Pixel LoRA)", "all", None, None, 1.0, 0.0),
                ("Full (Dual LoRA)", "all", None, None, 1.0, 1.0),
                ("Split Mask Overlay", "all", "noise=1.0,motion=0.0", "left_half", 1.0, 0.8),
            ]
        elif scenario_id == 1:
            # Row 1: Defocus Blur + 8-bit Color Banding + Warm Color Cast
            params.rgb_gains = np.array([1.25, 1.02, 0.76], dtype=np.float32)
            params.saturation_scale = 0.88
            s_demo[pipeline.type_to_idx["defocus_blur"]] = 0.80
            s_demo[pipeline.type_to_idx["color_banding"]] = 0.82
            s_demo[pipeline.type_to_idx["color_degradation"]] = 0.85
            runs = [
                ("Identity (a=0)", "none", None, None, 1.0, 0.0),
                ("Deband (Keep Cast)", "banding=1.0,color=0.0,defocus=0.0", None, None, 1.0, 0.0),
                ("Deband+Color (Keep Blur)", "banding=1.0,color=1.0,defocus=0.0", None, None, 1.0, 0.0),
                ("50% Partial Defocus", "banding=1.0,color=1.0,defocus=0.5", None, None, 1.0, 0.5),
                ("Full (Pixel LoRA)", "all", None, None, 1.0, 0.0),
                ("Full (Dual LoRA)", "all", None, None, 1.0, 1.0),
                ("Split Mask Overlay", "all", "banding=1.0,defocus=0.0", "left_half", 1.0, 0.8),
            ]
        else:
            # Row 2: 4x Downsample/SR + Heavy JPEG + Bright Overexposure
            params.underexpose = False
            s_demo[pipeline.type_to_idx["downsample_upsample"]] = 0.85
            s_demo[pipeline.type_to_idx["jpeg_compression"]] = 0.78
            s_demo[pipeline.type_to_idx["exposure_issue"]] = 0.65
            runs = [
                ("Identity (a=0)", "none", None, None, 1.0, 0.0),
                ("DeJPEG (Keep LowRes)", "jpeg=1.0,sr=0.0,exposure=0.0", None, None, 1.0, 0.0),
                ("DeJPEG+Exp (Keep LowRes)", "jpeg=1.0,exposure=1.0,sr=0.0", None, None, 1.0, 0.0),
                ("50% Partial SR", "jpeg=1.0,exposure=1.0,sr=0.5", None, None, 1.0, 0.5),
                ("Full (Pixel LoRA)", "all", None, None, 1.0, 0.0),
                ("Full (Dual LoRA)", "all", None, None, 1.0, 1.0),
                ("Split Mask Overlay", "all", "jpeg=1.0,sr=0.0", "left_half", 1.0, 0.8),
            ]

        src_u8 = pipeline.render_with_params(clean_u8, s_demo, params)
        src_t = torch.from_numpy(src_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        left_mask = np.zeros((h, w), dtype=np.float32)
        left_mask[:, : w // 2] = 1.0
        left_mask = cv2.GaussianBlur(left_mask, (0, 0), sigmaX=5.0, sigmaY=5.0)

        specs = [
            ("Clean GT", clean_u8),
            ("Degraded Input", src_u8),
        ]

        for label, fg_str, bg_str, mask_tag, f_scale, d_scale in runs:
            mask_2d = left_mask if mask_tag == "left_half" else None
            fg_vec = parse_action_spec(fg_str, config.DEGRADATION_TYPES)
            bg_vec = parse_action_spec(bg_str, config.DEGRADATION_TYPES) if bg_str else None
            overlay = build_spatial_action_overlay(h, w, fg_vec, bg_vec, mask_2d, device=device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_t = raw_model.sample(
                    src_01=src_t,
                    action_input=overlay,
                    steps=1,
                    fidelity_scale=f_scale,
                    detail_scale=d_scale if stage in ("semantic_stage", "joint") else 0.0,
                )
            pred_u8 = np.clip(
                np.round(pred_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0), 0.0, 255.0
            ).astype(np.uint8)
            specs.append((label, pred_u8))

        pad = 6
        header_h = 44
        row_w = len(specs) * (w + pad) + pad
        row_h = header_h + h + pad
        panel = np.full((row_h, row_w, 3), 24, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        for c_idx, (col_title, tile_u8) in enumerate(specs):
            x0 = pad + c_idx * (w + pad)
            cv2.putText(panel, col_title, (x0 + 4, header_h - 12), font, 0.39, (220, 240, 180), 1, cv2.LINE_AA)
            panel[header_h : header_h + h, x0 : x0 + w] = tile_u8

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
    raw_disc: torch.nn.Module = None,
    disc_optimizer=None,
):
    """Save lightweight unwrapped adapter checkpoint (~50 MB) for seamless resuming & inference."""
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
        "model_config": {
            "SEMANTIC_ENCODER_TYPE": config.SEMANTIC_ENCODER_TYPE,
            "RADIO_MODEL_ID": getattr(config, "RADIO_MODEL_ID", "nvidia/C-RADIOv3-B"),
            "GENERATOR_BACKBONE": config.GENERATOR_BACKBONE,
            "RESTORATION_MAX_TIMESTEP": getattr(config, "RESTORATION_MAX_TIMESTEP", 999),
            "PROMPT_SEQ_LEN": getattr(config, "PROMPT_SEQ_LEN", 77),
            "UNET_LORA_RANK": config.UNET_LORA_RANK,
            "UNET_LORA_ALPHA": config.UNET_LORA_ALPHA,
            "VAE_ENC_LORA_RANK": config.VAE_ENC_LORA_RANK,
            "VAE_DEC_LORA_RANK": getattr(config, "VAE_DEC_LORA_RANK", config.VAE_ENC_LORA_RANK),
            "VISION_LORA_RANK": config.VISION_LORA_RANK,
            "ACTION_HIDDEN_DIM": config.ACTION_HIDDEN_DIM,
            "SPATIAL_MOD_HIDDEN_DIM": config.SPATIAL_MOD_HIDDEN_DIM,
        },
    }
    if filename == "model_latest.pt" and raw_disc is not None:
        payload["disc_state_dict"] = {k: v.detach().cpu() for k, v in raw_disc.state_dict().items()}
        if disc_optimizer is not None:
            payload["disc_optimizer_state_dict"] = disc_optimizer.state_dict()
    torch.save(payload, str(path))
    print(f"Checkpoint saved to {path}")
    return str(path)


def _detect_visible_gpu_count() -> int:
    """Detect visible CUDA GPU count without initializing a CUDA context in the launcher process."""
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


def _launch_ddp_subprocess(num_gpus: int) -> None:
    """Spawn isolated multi-process DistributedDataParallel workers (ideal for Kaggle %run notebooks)."""
    sys.last_traceback = None
    sys.last_value = None
    gc.collect()

    env = os.environ.copy()
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.pop("PYTORCH_ALLOC_CONF", None)
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_KERNEL_CACHE_PATH", "/tmp/torch_kernels")
    if "--tqdm" not in sys.argv[1:]:
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

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
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            sys.stdout.write(chunk.decode("utf-8", errors="replace"))
            sys.stdout.flush()
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
    """Deterministic per-epoch sub-sampler that draws a fresh random slice of exact `num_steps` batches."""

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
    align_only: bool,
    is_ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    """Construct high-throughput DataLoader with DistributedSampler, persistent workers & prefetching."""
    train_dataset.align_only = align_only
    train_dataset.compact_ipc = True
    if world_size > 1:
        train_dataset._cache_capacity = max(4, int(getattr(config, "DECODE_CACHE_SIZE", 16)) // world_size)

    global_bsz = (
        min(32, config.BATCH_SIZE * 4)
        if (align_only and not getattr(config, "_debug_mode", False))
        else config.BATCH_SIZE
    )
    per_gpu_bsz = max(1, global_bsz // max(1, world_size))
    total_nw = int(config.NUM_WORKERS)
    nw = max(1, total_nw // max(1, world_size)) if total_nw > 0 else 0

    max_steps = int(getattr(config, "MAX_STEPS_PER_EPOCH", 0) or 0)
    full_steps = len(train_dataset) // max(1, global_bsz)
    sampler = None
    if max_steps > 0 and max_steps < full_steps:
        sampler = _EpochStepCapSampler(
            dataset_len=len(train_dataset),
            num_steps=max_steps,
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

        if is_main:
            mode_tag = f"DDP ({world_size} GPUs)" if is_ddp else f"{device} (GPUs: {num_gpus})"
            gckpt_tag = "ON" if getattr(config, "GRADIENT_CHECKPOINTING", True) else "OFF"
            print(
                f"Device: {mode_tag} | Global Batch Size: {config.BATCH_SIZE} "
                f"({max(1, config.BATCH_SIZE // max(1, world_size))}/GPU) | "
                f"Backbone: {config.GENERATOR_BACKBONE} | Vision: {config.SEMANTIC_ENCODER_TYPE} | "
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

        # Rank 0 populates Hugging Face / torchvision disk caches first to prevent concurrent file-write races
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
                f"Trainable Adapter Breakdown -> Vision LoRA: {sum(p.numel() for p in groups['vision_lora']):,} | "
                f"Pixel Stage (LoRA+SkipVAE+Overlay): {sum(p.numel() for p in groups['pixel_stage']):,} | "
                f"Semantic LoRA: {sum(p.numel() for p in groups['semantic_stage']):,}",
                flush=True,
            )

        start_epoch = 1
        best_psnr = -1.0
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
                adapter_state = ckpt.get("trainable_state_dict") or ckpt.get("model_state_dict")
                if adapter_state is not None:
                    missing_keys, unexpected_keys = raw_model.load_trainable_state_dict(adapter_state)
                    if is_main and (missing_keys or unexpected_keys):
                        print(
                            f"[Warning] Partial adapter match on resume "
                            f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)}). "
                            f"If this is an older V1/V2 checkpoint, start V3 fresh without --resume!",
                            flush=True,
                        )
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                last_epoch = start_epoch - 1
                best_psnr = float(ckpt.get("best_psnr", -1.0))
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
                            eff_rem_pix = 1
                        elif resumed_stage == "semantic_stage":
                            eff_rem_pix = 0
                        else:
                            eff_rem_pix = 4
                    if eff_extra is None and not epochs_overridden:
                        if resumed_stage == "pixel_stage":
                            eff_extra = int(eff_rem_pix or 0) + 5
                        elif resumed_stage == "semantic_stage":
                            eff_extra = 4
                        else:
                            eff_extra = int(eff_rem_pix or 4) + 5

                if eff_rem_pix is not None:
                    config.ALIGN_WARMUP_EPOCHS = min(config.ALIGN_WARMUP_EPOCHS, max(0, last_epoch))
                    config.PIXEL_STAGE_EPOCHS = max(
                        0, (last_epoch - config.ALIGN_WARMUP_EPOCHS) + int(eff_rem_pix)
                    )
                if eff_extra is not None:
                    config.EPOCHS = last_epoch + max(1, int(eff_extra))
                    resumed_sched_state = None

                if resumed_stage != determine_epoch_stage(start_epoch, config):
                    resumed_opt_state = None
                    resumed_sched_state = None
                    resumed_disc_state = None
                    resumed_disc_opt_state = None
                if is_main:
                    sched_note = ""
                    if speedrun or eff_rem_pix is not None or eff_extra is not None:
                        sched_note = (
                            f" | schedule -> total_epochs={config.EPOCHS}, "
                            f"pixel_end={config.ALIGN_WARMUP_EPOCHS + config.PIXEL_STAGE_EPOCHS}, "
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
                        config.ALIGN_WARMUP_EPOCHS = 1
                        config.PIXEL_STAGE_EPOCHS = 4
                    if not epochs_overridden:
                        config.EPOCHS = 10
                if is_main:
                    print(f"No checkpoint found at {resume_path} — starting fresh training from Epoch 1.", flush=True)

        # Diversify per-rank augmentation RNG after identical model weight initialization
        if is_ddp:
            set_seed(config.SEED + rank * 1009)

        model = raw_model
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
        if is_main:
            print_gpu_usage("Pre-Training GPU & VRAM Check")

        current_stage = None
        current_loader_align_only = None
        train_loader = None
        train_sampler = None
        optimizer = None
        scheduler = None
        raw_disc = None
        disc_model = None
        disc_optimizer = None
        disc_scaler = None
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
            need_align_only = stage == "align_warmup"
            if train_loader is None or need_align_only != current_loader_align_only:
                if train_loader is not None:
                    if getattr(train_loader, "_iterator", None) is not None:
                        train_loader._iterator._shutdown_workers()
                    del train_loader
                    gc.collect()
                current_loader_align_only = need_align_only
                train_loader, train_sampler = _build_train_loader(
                    train_dataset,
                    config,
                    device,
                    align_only=need_align_only,
                    is_ddp=is_ddp,
                    rank=rank,
                    world_size=world_size,
                )

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if stage != current_stage:
                prev_stage = current_stage if current_stage is not None else resumed_stage
                if prev_stage in ("align_warmup", "pixel_stage") and stage == "semantic_stage":
                    if is_main:
                        best_ckpt_p = Path(config.CHECKPOINT_DIR) / "model_best.pt"
                        pix_backup_p = Path(config.CHECKPOINT_DIR) / "model_best_pixel.pt"
                        if best_ckpt_p.exists() and not pix_backup_p.exists():
                            shutil.copy2(str(best_ckpt_p), str(pix_backup_p))
                            print(f"Backed up best Pixel-Stage checkpoint -> {pix_backup_p}", flush=True)
                    # Reset best_psnr so model_best.pt saves the best Dual-LoRA (semantic_stage) checkpoint
                    best_psnr = -1.0

                if is_ddp and model is not raw_model:
                    del model
                    gc.collect()
                if disc_model is not None:
                    del disc_model, raw_disc, disc_optimizer
                    disc_model = None
                    raw_disc = None
                    disc_optimizer = None
                    disc_scaler = None
                    gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                current_stage = stage
                optimizer, active_params = build_stage_optimizer(raw_model, stage, config)
                remaining = max(1, config.EPOCHS - epoch + 1)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-6)
                if device.type == "cuda":
                    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

                gan_w = float(getattr(config, "LOSS_GAN_WEIGHT", 0.20))
                if stage in ("semantic_stage", "joint") and gan_w > 0.0:
                    raw_disc = VisionAidedDiscriminator(
                        sem_dim=raw_model.semantic_encoder.spatial_dim,
                        num_actions=config.NUM_ACTIONS,
                    ).to(device)
                    if epoch == start_epoch and resumed_stage == stage and resumed_disc_state is not None:
                        try:
                            raw_disc.load_state_dict(resumed_disc_state)
                        except Exception:
                            pass
                        resumed_disc_state = None
                    disc_optimizer = optim.AdamW(
                        raw_disc.parameters(),
                        lr=float(getattr(config, "DISC_LR", 1e-4)),
                        betas=(0.5, 0.999),
                        weight_decay=1e-4,
                    )
                    if epoch == start_epoch and resumed_stage == stage and resumed_disc_opt_state is not None:
                        try:
                            disc_optimizer.load_state_dict(resumed_disc_opt_state)
                        except Exception:
                            pass
                        resumed_disc_opt_state = None
                    if device.type == "cuda":
                        disc_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
                    if is_ddp:
                        disc_model = torch.nn.parallel.DistributedDataParallel(
                            raw_disc,
                            device_ids=[local_rank],
                            output_device=local_rank,
                            find_unused_parameters=False,
                        )
                    else:
                        disc_model = raw_disc

                if is_main:
                    disc_note = (
                        f" + Vision-Aided GAN ({sum(p.numel() for p in raw_disc.parameters()):,} D params)"
                        if raw_disc is not None
                        else ""
                    )
                    print(
                        f"\n=== Entering Stage '{stage}' at Epoch {epoch}/{config.EPOCHS} "
                        f"({active_params:,} active G params{disc_note}) ===",
                        flush=True,
                    )

                # Restore optimizer & scheduler momentum if resuming within the exact same stage
                if epoch == start_epoch and resumed_stage == stage and resumed_opt_state is not None:
                    try:
                        optimizer.load_state_dict(resumed_opt_state)
                        if resumed_sched_state is not None:
                            scheduler.load_state_dict(resumed_sched_state)
                        if is_main:
                            print(f"Restored optimizer & scheduler states for stage '{stage}'.", flush=True)
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

            model.train()
            running = {}
            num_steps = len(train_loader)
            use_tqdm = bool(getattr(config, "USE_TQDM", False))
            log_interval = max(1, int(getattr(config, "LOG_INTERVAL", 100)))
            sample_every = max(0, int(getattr(config, "SAMPLE_EVERY_N_STEPS", 500)))
            step_width = max(4, len(str(num_steps)))
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

            for step_idx, batch in enumerate(pbar, start=1):
                clean = batch["clean"].to(device, non_blocking=True)
                src = batch["source"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                aux = {}
                out = None
                tgt = None
                act_map = None

                try:
                    if stage == "align_warmup":
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            align_out = model(src, clean_01=clean, align_only=True)
                            loss = align_out.float().mean()
                        val_f = float(loss.detach().item())
                        metrics = {"align": val_f, "total": val_f}
                    else:
                        tgt = batch["target"].to(device, non_blocking=True)
                        act_map = unpack_batch_action_map(batch, device)
                        d_scale = 0.0 if stage == "pixel_stage" else float(random.uniform(0.4, 1.25))

                        if raw_disc is not None:
                            # Freeze D weights & spectral_norm power buffers during G forward/backward
                            raw_disc.eval()
                            raw_disc.requires_grad_(False)

                        with torch.amp.autocast("cuda", enabled=use_amp):
                            out = model(
                                src,
                                act_map,
                                tgt_01=tgt,
                                clean_01=clean,
                                fidelity_scale=1.0,
                                detail_scale=d_scale,
                            )
                            loss, metrics, aux = loss_fn(
                                pred_01=out["pred"],
                                target_01=tgt,
                                action_map=act_map,
                                align_loss=out["align_loss"],
                                stage=stage,
                                t_vec=out["t"],
                                semantic_encoder=(
                                    raw_model.semantic_encoder if stage in ("semantic_stage", "joint") else None
                                ),
                                discriminator=raw_disc,
                                return_aux=True,
                            )

                    if scaler is not None and use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in raw_model.parameters() if p.requires_grad], config.GRAD_CLIP_NORM
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in raw_model.parameters() if p.requires_grad], config.GRAD_CLIP_NORM
                        )
                        optimizer.step()
                finally:
                    raw_model.clear_runtime_contexts()

                if (
                    disc_model is not None
                    and raw_disc is not None
                    and disc_optimizer is not None
                    and out is not None
                    and "pred_grid" in aux
                    and "tgt_grid" in aux
                ):
                    raw_disc.requires_grad_(True)
                    disc_model.train()
                    disc_optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        d_loss = VisionAidedDiscriminator.discriminator_hinge_loss(
                            disc_module=disc_model,
                            real_rgb=tgt,
                            fake_rgb=out["pred"].detach(),
                            real_grid=aux["tgt_grid"],
                            fake_grid=aux["pred_grid"],
                            action_map=act_map,
                        )
                    if disc_scaler is not None and use_amp:
                        disc_scaler.scale(d_loss).backward()
                        disc_scaler.unscale_(disc_optimizer)
                        torch.nn.utils.clip_grad_norm_(raw_disc.parameters(), config.GRAD_CLIP_NORM)
                        disc_scaler.step(disc_optimizer)
                        disc_scaler.update()
                    else:
                        d_loss.backward()
                        torch.nn.utils.clip_grad_norm_(raw_disc.parameters(), config.GRAD_CLIP_NORM)
                        disc_optimizer.step()
                    metrics["d_loss"] = float(d_loss.detach().item())
                del aux, out

                for k, v in metrics.items():
                    running[k] = running.get(k, 0.0) + v

                if is_main and (step_idx == 1 or step_idx % log_interval == 0 or step_idx == num_steps):
                    avg_tot = running["total"] / step_idx
                    avg_al = running.get("align", 0.0) / step_idx
                    gpu_str = format_gpu_postfix()
                    if use_tqdm:
                        pbar.set_postfix(
                            {"loss": f"{avg_tot:.4f}", "align": f"{avg_al:.4f}", "gpu": gpu_str}
                        )
                    else:
                        elapsed = max(1e-6, time.perf_counter() - epoch_start_t)
                        sec_per_it = elapsed / step_idx
                        rate_str = f"{sec_per_it:.2f}s/it" if sec_per_it >= 1.0 else f"{1.0 / sec_per_it:.2f}it/s"
                        eta_str = _format_duration(sec_per_it * (num_steps - step_idx))
                        pct = 100.0 * step_idx / max(1, num_steps)
                        cur_lr = optimizer.param_groups[-1]["lr"]
                        if stage == "align_warmup":
                            loss_parts = f"loss={avg_tot:.4f} align={avg_al:.4f}"
                        elif stage == "pixel_stage":
                            avg_l1 = running.get("l1", 0.0) / step_idx
                            avg_perc = running.get("perc", 0.0) / step_idx
                            loss_parts = f"loss={avg_tot:.4f} l1={avg_l1:.4f} perc={avg_perc:.4f} align={avg_al:.4f}"
                        else:
                            avg_l1 = running.get("l1", 0.0) / step_idx
                            avg_perc = running.get("perc", 0.0) / step_idx
                            avg_d = running.get("d_loss", 0.0) / step_idx
                            loss_parts = f"loss={avg_tot:.4f} l1={avg_l1:.4f} perc={avg_perc:.4f} d={avg_d:.3f}"
                        print(
                            f"[Epoch {epoch:03d}/{config.EPOCHS} | {stage}] "
                            f"{step_idx:0{step_width}d}/{num_steps} ({pct:5.1f}%) | "
                            f"{loss_parts} | lr={cur_lr:.2e} | {rate_str} ETA {eta_str} | {gpu_str}",
                            flush=True,
                        )

                # Rank-symmetric mid-epoch visual sweep check (avoids redundant save right before epoch end)
                should_mid_sample = (
                    sample_every > 0
                    and stage != "align_warmup"
                    and (step_idx % sample_every == 0)
                    and (num_steps - step_idx) > min(75, sample_every // 5)
                )
                if should_mid_sample:
                    if is_main and val_dataset is not None:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        preview_path = save_controllable_validation_grid(
                            raw_model, val_dataset, device, config, epoch, stage, step=step_idx
                        )
                        print(
                            f"  ↳ [Preview @ Epoch {epoch:03d} Step {step_idx:0{step_width}d}/{num_steps}] "
                            f"Saved sweep -> {preview_path}",
                            flush=True,
                        )
                    if is_ddp:
                        dist.barrier()

            # Step scheduler BEFORE saving checkpoints so resumed runs have the exact next-epoch LR
            if scheduler is not None:
                scheduler.step()

            epoch_summary = {k: v / max(1, num_steps) for k, v in running.items()}
            if is_ddp and epoch_summary:
                keys_sorted = sorted(epoch_summary.keys())
                vec = torch.tensor([epoch_summary[k] for k in keys_sorted], device=device, dtype=torch.float64)
                dist.all_reduce(vec, op=dist.ReduceOp.AVG)
                for idx_k, k in enumerate(keys_sorted):
                    epoch_summary[k] = float(vec[idx_k].item())

            epoch_summary["epoch"] = epoch
            epoch_summary["stage"] = stage

            if is_main:
                if stage != "align_warmup" and (epoch % max(1, config.VAL_EPOCH_INTERVAL) == 0 or epoch == config.EPOCHS):
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    val_metrics = evaluate_validation(raw_model, val_loader, device, config, stage)
                    epoch_summary.update(val_metrics)
                    grid_path = save_controllable_validation_grid(raw_model, val_dataset, device, config, epoch, stage)
                    rest_psnr = val_metrics.get("val_restore_psnr", val_metrics["val_psnr"])
                    rest_gain = val_metrics.get("val_restore_gain", 0.0)
                    print(
                        f"Epoch {epoch:03d} | train_loss={epoch_summary['total']:.4f} | "
                        f"restore_psnr={rest_psnr:.2f} dB ({rest_gain:+.2f} dB) | "
                        f"val_psnr={val_metrics['val_psnr']:.2f} dB | val_l1={val_metrics['val_l1']:.4f} | sweep={grid_path}",
                        flush=True,
                    )

                    if rest_psnr > best_psnr:
                        best_psnr = rest_psnr
                        save_checkpoint(raw_model, optimizer, scheduler, epoch, stage, best_psnr, config, "model_best.pt")

                # Always save model_latest.pt after every epoch for seamless Kaggle resuming
                save_checkpoint(
                    raw_model,
                    optimizer,
                    scheduler,
                    epoch,
                    stage,
                    best_psnr,
                    config,
                    "model_latest.pt",
                    raw_disc=raw_disc,
                    disc_optimizer=disc_optimizer,
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
    finally:
        if is_ddp and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train Controllable Pretrained Image Restoration Model")
    parser.add_argument("--debug", action="store_true", help="Use DebugConfig for quick verification")
    parser.add_argument("--mock", action="store_true", help="Use offline mock_tiny backbones (no HF downloads)")
    parser.add_argument("--no-ddp", action="store_true", help="Disable automatic multi-GPU DDP worker launch")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root (DIV2K or image folder)")
    parser.add_argument("--train-root", type=str, default=None, help="Explicit training image folder (e.g. Kaggle df2k-ost/train)")
    parser.add_argument("--val-root", type=str, default=None, help="Explicit validation image folder (e.g. Kaggle DIV2K_valid)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for sweep grids")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory for adapter checkpoints")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for training logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to model_latest.pt / model_best.pt checkpoint")
    parser.add_argument("--backbone", type=str, choices=["sd-turbo", "tiny-sd", "bk-sdm-tiny", "mock_tiny"], default=None)
    parser.add_argument("--vision-encoder", type=str, choices=["c_radio", "dinov3_siglip2", "mock_tiny"], default=None)
    parser.add_argument("--radio-id", type=str, default=None, help="Override HF repo ID for C-RADIO (e.g. nvidia/C-RADIOv3-B)")
    parser.add_argument("--training-mode", type=str, choices=["staged", "joint"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--align-epochs", type=int, help="Override Phase-1 Vision LoRA warmup epochs", default=None)
    parser.add_argument("--pixel-epochs", type=int, help="Override Phase-2 Pixel-LoRA stage epochs", default=None)
    parser.add_argument(
        "--extra-epochs",
        type=int,
        default=None,
        help="When resuming, run N more epochs past the checkpoint epoch",
    )
    parser.add_argument(
        "--remaining-pixel-epochs",
        type=int,
        default=None,
        help="When resuming, run K more pixel_stage epochs (0 = jump straight to semantic_stage)",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Cap optimizer steps per epoch (draws a fresh random dataset slice each epoch)",
    )
    parser.add_argument("--train-images", type=int, default=None, help="Cap number of training reference images")
    parser.add_argument("--val-images", type=int, default=None, help="Cap number of validation reference images")
    parser.add_argument(
        "--speedrun",
        action="store_true",
        help="Fast ~50-min finish preset (300 steps/epoch + quick transition into semantic_stage)",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None, help="Steps between clean log lines (default: 100)")
    parser.add_argument("--sample-steps", type=int, default=None, help="Save mid-epoch sweep grid every N steps (0=disable, default: 500)")
    parser.add_argument("--tqdm", action="store_true", help="Use interactive tqdm progress bar instead of background-safe line logs")
    args = parser.parse_args()

    config = DebugConfig() if args.debug else Config()
    if args.mock:
        config.GENERATOR_BACKBONE = "mock_tiny"
        config.SEMANTIC_ENCODER_TYPE = "mock_tiny"
    if args.backbone:
        config.GENERATOR_BACKBONE = args.backbone
    if args.vision_encoder:
        config.SEMANTIC_ENCODER_TYPE = args.vision_encoder
    if args.radio_id:
        config.RADIO_MODEL_ID = args.radio_id
    if args.training_mode:
        config.TRAINING_MODE = args.training_mode
    if args.data_root:
        config.DATASET_PATH = args.data_root
    if args.train_root:
        config.TRAIN_DATASET_PATH = args.train_root
    if args.val_root:
        config.VAL_DATASET_PATH = args.val_root
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.checkpoint_dir:
        config.CHECKPOINT_DIR = args.checkpoint_dir
    if args.log_dir:
        config.LOG_DIR = args.log_dir
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.speedrun:
        config.MAX_STEPS_PER_EPOCH = 400
        config.LOG_INTERVAL = 50
        config.SAMPLE_EVERY_N_STEPS = 200
        if not args.resume:
            config.ALIGN_WARMUP_EPOCHS = 1
            config.PIXEL_STAGE_EPOCHS = 4
            config.EPOCHS = 10
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.align_epochs is not None:
        config.ALIGN_WARMUP_EPOCHS = args.align_epochs
    if args.pixel_epochs is not None:
        config.PIXEL_STAGE_EPOCHS = args.pixel_epochs
    if args.steps_per_epoch is not None:
        config.MAX_STEPS_PER_EPOCH = args.steps_per_epoch
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
    if args.tqdm:
        config.USE_TQDM = True

    in_ddp_worker = "LOCAL_RANK" in os.environ and "RANK" in os.environ
    if not in_ddp_worker and not args.no_ddp and config.MULTI_GPU and config.DEVICE == "cuda":
        visible_gpus = _detect_visible_gpu_count()
        if visible_gpus > 1:
            _launch_ddp_subprocess(visible_gpus)
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
    )


if __name__ == "__main__":
    main()
