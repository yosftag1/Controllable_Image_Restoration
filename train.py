"""3-Phase Multi-GPU AMP Training Script for Controllable Pretrained Restoration."""

import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys

# Kaggle notebook %run path fix & CUDA memory fragmentation prevention
if "/kaggle/working/Controllable_Image_Restoration" not in sys.path:
    sys.path.insert(0, "/kaggle/working/Controllable_Image_Restoration")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
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
from losses.controllable_losses import ControllableRestorationLoss
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
    return model.module if isinstance(model, torch.nn.DataParallel) else model


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


def compute_psnr(pred_01: torch.Tensor, tgt_01: torch.Tensor) -> float:
    mse = torch.mean((pred_01.float() - tgt_01.float()) ** 2, dim=(1, 2, 3)).clamp(min=1e-10)
    psnr = -10.0 * torch.log10(mse)
    return float(psnr.mean().item())


@torch.no_grad()
def evaluate_validation(model, val_loader, device, config, stage: str):
    raw_model = unwrap_model(model)
    raw_model.eval()
    use_amp = config.USE_AMP and (device.type == "cuda")
    detail_scale = config.DEFAULT_DETAIL_SCALE if stage in ("semantic_stage", "joint") else 0.0

    total_psnr = 0.0
    total_l1 = 0.0
    count = 0

    for batch in tqdm(val_loader, desc="Validation", leave=False):
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
        total_psnr += compute_psnr(pred, tgt) * bsz
        total_l1 += l1 * bsz
        count += bsz

    raw_model.train()
    denom = max(1, count)
    return {"val_psnr": total_psnr / denom, "val_l1": total_l1 / denom}


@torch.no_grad()
def save_controllable_validation_grid(model, val_dataset, device, config, epoch: int, stage: str):
    """Render visual sweep proving selective, partial, Dual-LoRA, and localized mask control."""
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
        params.underexpose = True  # Consistent dark mood + motion blur + noise demo

        s_demo = np.zeros(config.NUM_ACTIONS, dtype=np.float32)
        s_demo[pipeline.type_to_idx["motion_blur"]] = 0.80
        s_demo[pipeline.type_to_idx["gaussian_noise"]] = 0.75
        s_demo[pipeline.type_to_idx["exposure_issue"]] = 0.65
        src_u8 = pipeline.render_with_params(clean_u8, s_demo, params)

        src_t = torch.from_numpy(src_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        # Split mask: Left half = Full restore, Right half = Denoise only (keep motion blur & mood)
        left_mask = np.zeros((h, w), dtype=np.float32)
        left_mask[:, : w // 2] = 1.0
        left_mask = cv2.GaussianBlur(left_mask, (0, 0), sigmaX=5.0, sigmaY=5.0)

        specs = [
            ("Clean GT", clean_u8),
            ("Degraded Input", src_u8),
        ]

        runs = [
            ("Identity (a=0)", "none", None, None, 1.0, 0.0),
            ("Denoise (Keep Blur!)", "noise=1.0,motion=0.0,exposure=0.0", None, None, 1.0, 0.0),
            ("Denoise+Exp (Keep Blur)", "noise=1.0,exposure=1.0,motion=0.0", None, None, 1.0, 0.0),
            ("50% Partial Deblur", "noise=1.0,exposure=1.0,motion=0.5", None, None, 1.0, 0.5),
            ("Full (Pixel LoRA)", "all", None, None, 1.0, 0.0),
            ("Full (Dual LoRA)", "all", None, None, 1.0, 1.0),
            ("Split Mask Overlay", "all", "noise=1.0,motion=0.0", left_mask, 1.0, 0.8),
        ]

        for label, fg_str, bg_str, mask_2d, f_scale, d_scale in runs:
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
            if idx == 0:
                cv2.putText(panel, col_title, (x0 + 4, header_h - 12), font, 0.42, (220, 240, 180), 1, cv2.LINE_AA)
            panel[header_h : header_h + h, x0 : x0 + w] = tile_u8

        rows_canvas.append(panel)

    full_grid = np.concatenate(rows_canvas, axis=0)
    save_path = out_dir / f"epoch_{epoch:03d}_{stage}.png"
    cv2.imwrite(str(save_path), cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR))
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
):
    """Save lightweight unwrapped adapter checkpoint (~45 MB) for seamless resuming & inference."""
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
            "UNET_LORA_RANK": config.UNET_LORA_RANK,
            "UNET_LORA_ALPHA": config.UNET_LORA_ALPHA,
            "VAE_ENC_LORA_RANK": config.VAE_ENC_LORA_RANK,
            "VISION_LORA_RANK": config.VISION_LORA_RANK,
            "ACTION_HIDDEN_DIM": config.ACTION_HIDDEN_DIM,
            "SPATIAL_MOD_HIDDEN_DIM": config.SPATIAL_MOD_HIDDEN_DIM,
        },
    }
    torch.save(payload, str(path))
    print(f"Checkpoint saved to {path}")
    return str(path)


def _wrap_multi_gpu_safely(raw_model: ControllableRestorer, num_gpus: int, device: torch.device, use_amp: bool):
    """Wrap across all GPUs; if Vision HF remote code binds methods to GPU 0, parallelize the 950M Generator across all GPUs."""
    device_ids = list(range(num_gpus))
    dummy_img = torch.zeros(num_gpus, 3, 128, 128, device=device)
    dummy_act = torch.ones(num_gpus, raw_model.config.NUM_ACTIONS, 128, 128, device=device) * 0.5

    dp_model = torch.nn.DataParallel(raw_model, device_ids=device_ids)
    try:
        raw_model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            _ = dp_model(dummy_img, dummy_act, tgt_01=dummy_img, clean_01=dummy_img, fidelity_scale=1.0, detail_scale=0.0)
        raw_model.train()
        print(f"✓ Multi-GPU DataParallel ENABLED across {num_gpus} GPUs {device_ids}")
        return dp_model
    except Exception:
        del dp_model
        gc.collect()
        torch.cuda.empty_cache()

    try:
        split_model = SplitMultiGPURestorer(raw_model, device_ids=device_ids)
        raw_model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            _ = split_model(dummy_img, dummy_act, tgt_01=dummy_img, clean_01=dummy_img, fidelity_scale=1.0, detail_scale=0.0)
        raw_model.train()
        print(
            f"✓ Split Multi-GPU ENABLED: Vision Encoder on cuda:0 + "
            f"Generator (VAE+UNet) DataParallel across {num_gpus} GPUs {device_ids}"
        )
        return split_model
    except Exception as exc:
        gc.collect()
        torch.cuda.empty_cache()
        raw_model.train()
        print(f"[Note] Multi-GPU fallback to single device {device} ({exc}).")
        return raw_model


def _build_train_loader(train_dataset, config: Config, device: torch.device, align_only: bool) -> DataLoader:
    """Construct high-throughput DataLoader with persistent workers, prefetching & OpenCV lock prevention."""
    train_dataset.align_only = align_only
    train_dataset.compact_ipc = True
    bsz = min(32, config.BATCH_SIZE * 2) if (align_only and not getattr(config, "_debug_mode", False)) else config.BATCH_SIZE
    nw = int(config.NUM_WORKERS)
    loader_kwargs = {
        "batch_size": bsz,
        "shuffle": True,
        "num_workers": nw,
        "pin_memory": (device.type == "cuda"),
        "drop_last": (len(train_dataset) >= bsz),
    }
    if nw > 0:
        loader_kwargs["worker_init_fn"] = dataloader_worker_init_fn
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(getattr(config, "DATALOADER_PREFETCH_FACTOR", 4))
    return DataLoader(train_dataset, **loader_kwargs)


def train(config: Config, resume_path: str = None, debug: bool = False, batch_size_overridden: bool = False):
    set_seed(config.SEED)
    config._debug_mode = debug
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_multi_gpu = config.MULTI_GPU and num_gpus > 1
    use_amp = config.USE_AMP and (device.type == "cuda")

    if num_gpus == 1 and not batch_size_overridden and config.BATCH_SIZE >= 16 and not debug:
        config.BATCH_SIZE = 8

    print(
        f"Device: {device} (GPUs: {num_gpus}) | Batch Size: {config.BATCH_SIZE} | "
        f"Backbone: {config.GENERATOR_BACKBONE} | Vision: {config.SEMANTIC_ENCODER_TYPE} | "
        f"AMP: {'ENABLED' if use_amp else 'DISABLED'}"
    )

    print("Loading datasets...")
    train_dataset = create_controllable_dataset(config, is_train=True, compact_ipc=True)
    val_dataset = create_controllable_dataset(config, is_train=False, compact_ipc=False)

    val_nw = min(2, int(config.NUM_WORKERS))
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, min(config.BATCH_SIZE, 8)),
        shuffle=False,
        num_workers=val_nw,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=dataloader_worker_init_fn if val_nw > 0 else None,
    )

    raw_model = ControllableRestorer(config).to(device)
    groups = raw_model.get_stage_parameter_groups()
    print(
        f"Trainable Adapter Breakdown -> Vision LoRA: {sum(p.numel() for p in groups['vision_lora']):,} | "
        f"Pixel Stage (LoRA+SkipVAE+Overlay): {sum(p.numel() for p in groups['pixel_stage']):,} | "
        f"Semantic LoRA: {sum(p.numel() for p in groups['semantic_stage']):,}"
    )

    start_epoch = 1
    best_psnr = -1.0
    resumed_stage = None
    resumed_opt_state = None
    resumed_sched_state = None

    if resume_path:
        if os.path.exists(resume_path):
            print(f"Resuming from {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            adapter_state = ckpt.get("trainable_state_dict") or ckpt.get("model_state_dict")
            if adapter_state is not None:
                raw_model.load_trainable_state_dict(adapter_state)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_psnr = float(ckpt.get("best_psnr", -1.0))
            resumed_stage = ckpt.get("stage")
            resumed_opt_state = ckpt.get("optimizer_state_dict")
            resumed_sched_state = ckpt.get("scheduler_state_dict")
            print(f"Resumed trainable adapters (starting at Epoch {start_epoch}, best PSNR={best_psnr:.2f} dB)")
        else:
            print(f"No checkpoint found at {resume_path} — starting fresh training from Epoch 1.")

    model = _wrap_multi_gpu_safely(raw_model, num_gpus, device, use_amp) if use_multi_gpu else raw_model
    loss_fn = ControllableRestorationLoss(config).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    print_gpu_usage("Pre-Training GPU & VRAM Check")

    current_stage = None
    current_loader_align_only = None
    train_loader = None
    optimizer = None
    scheduler = None
    os.makedirs(config.LOG_DIR, exist_ok=True)
    history_file = Path(config.LOG_DIR) / "training_history.json"
    history = []
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
            train_loader = _build_train_loader(train_dataset, config, device, align_only=need_align_only)

        if stage != current_stage:
            current_stage = stage
            optimizer, active_params = build_stage_optimizer(raw_model, stage, config)
            remaining = max(1, config.EPOCHS - epoch + 1)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-6)
            print(f"\n=== Entering Stage '{stage}' at Epoch {epoch}/{config.EPOCHS} ({active_params:,} active params) ===")

            # Restore optimizer & scheduler momentum if resuming within the exact same stage
            if epoch == start_epoch and resumed_stage == stage and resumed_opt_state is not None:
                try:
                    optimizer.load_state_dict(resumed_opt_state)
                    if resumed_sched_state is not None:
                        scheduler.load_state_dict(resumed_sched_state)
                    print(f"Restored optimizer & scheduler states for stage '{stage}'.")
                except Exception as exc:
                    print(f"[Note] Optimizer state re-initialized cleanly for stage '{stage}' ({exc}).")
                resumed_opt_state = None
                resumed_sched_state = None

        model.train()
        running = {}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{config.EPOCHS} [{stage}]")

        for step_idx, batch in enumerate(pbar, start=1):
            clean = batch["clean"].to(device, non_blocking=True)
            src = batch["source"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if stage == "align_warmup":
                with torch.amp.autocast("cuda", enabled=use_amp):
                    align_out = model(src, clean_01=clean, align_only=True)
                    loss = align_out.float().mean()
                metrics = {"align": float(loss.detach().item()), "total": float(loss.detach().item())}
            else:
                tgt = batch["target"].to(device, non_blocking=True)
                act_map = unpack_batch_action_map(batch, device)
                d_scale = 0.0 if stage == "pixel_stage" else float(random.uniform(0.4, 1.25))

                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(
                        src,
                        act_map,
                        tgt_01=tgt,
                        clean_01=clean,
                        fidelity_scale=1.0,
                        detail_scale=d_scale,
                    )
                    loss, metrics = loss_fn(
                        pred_01=out["pred"],
                        target_01=tgt,
                        action_map=act_map,
                        align_loss=out["align_loss"],
                        stage=stage,
                        t_vec=out["t"],
                        semantic_encoder=raw_model.semantic_encoder if stage in ("semantic_stage", "joint") else None,
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

            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v

            if step_idx % config.LOG_INTERVAL == 0 or step_idx == len(train_loader):
                avg_tot = running["total"] / step_idx
                avg_al = running.get("align", 0.0) / step_idx
                pbar.set_postfix(
                    {"loss": f"{avg_tot:.4f}", "align": f"{avg_al:.4f}", "gpu": format_gpu_postfix()}
                )

        # Step scheduler BEFORE saving checkpoints so resumed runs have the exact next-epoch LR
        if scheduler is not None:
            scheduler.step()

        epoch_summary = {k: v / max(1, len(train_loader)) for k, v in running.items()}
        epoch_summary["epoch"] = epoch
        epoch_summary["stage"] = stage

        if stage != "align_warmup" and (epoch % max(1, config.VAL_EPOCH_INTERVAL) == 0 or epoch == config.EPOCHS):
            val_metrics = evaluate_validation(model, val_loader, device, config, stage)
            epoch_summary.update(val_metrics)
            grid_path = save_controllable_validation_grid(model, val_dataset, device, config, epoch, stage)
            print(
                f"Epoch {epoch:03d} | train_loss={epoch_summary['total']:.4f} | "
                f"val_psnr={val_metrics['val_psnr']:.2f} dB | val_l1={val_metrics['val_l1']:.4f} | sweep={grid_path}"
            )

            if val_metrics["val_psnr"] > best_psnr:
                best_psnr = val_metrics["val_psnr"]
                save_checkpoint(raw_model, optimizer, scheduler, epoch, stage, best_psnr, config, "model_best.pt")

        # Always save model_latest.pt after every epoch for seamless Kaggle resuming
        save_checkpoint(raw_model, optimizer, scheduler, epoch, stage, best_psnr, config, "model_latest.pt")
        if epoch % config.CHECKPOINT_EVERY_N_EPOCHS == 0:
            save_checkpoint(
                raw_model, optimizer, scheduler, epoch, stage, best_psnr, config, f"model_epoch_{epoch:03d}.pt"
            )

        history.append(epoch_summary)
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train Controllable Pretrained Image Restoration Model")
    parser.add_argument("--debug", action="store_true", help="Use DebugConfig for quick verification")
    parser.add_argument("--mock", action="store_true", help="Use offline mock_tiny backbones (no HF downloads)")
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
    parser.add_argument("--align-epochs", type=int, default=None, help="Override Phase-1 Vision LoRA warmup epochs")
    parser.add_argument("--pixel-epochs", type=int, default=None, help="Override Phase-2 Pixel-LoRA stage epochs")
    parser.add_argument("--num-workers", type=int, default=None)
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
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.align_epochs is not None:
        config.ALIGN_WARMUP_EPOCHS = args.align_epochs
    if args.pixel_epochs is not None:
        config.PIXEL_STAGE_EPOCHS = args.pixel_epochs
    if args.num_workers is not None:
        config.NUM_WORKERS = args.num_workers

    train(
        config,
        resume_path=args.resume,
        debug=args.debug,
        batch_size_overridden=(args.batch_size is not None),
    )


if __name__ == "__main__":
    main()
