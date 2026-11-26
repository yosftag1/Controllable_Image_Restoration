"""Controllable Inference CLI (6 Orthogonal Controls, Generative Detail Dial, 1-Click Object Masks)."""

import argparse
import gc
import os
from pathlib import Path
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

if "/kaggle/working/Controllable_Image_Restoration" not in sys.path:
    sys.path.insert(0, "/kaggle/working/Controllable_Image_Restoration")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from data.selective_dataset import SelectiveDegradationPipeline
from models.controllable_restorer import ControllableRestorer
from train import cast_frozen_base_to_fp16
from utils.image_benchmark import run_single_image_benchmark
from utils.semantic_mask import (
    build_spatial_action_overlay,
    extract_one_click_semantic_mask,
    load_grayscale_mask,
    parse_action_spec,
    render_mask_overlay_preview,
)


class ControllableInferencer:
    """High-level wrapper for V4 controllable global, directional & localized image restoration."""

    def __init__(self, checkpoint_path: Optional[str] = None, config: Optional[Config] = None, device: str = "cuda"):
        sys.last_traceback = None
        sys.last_value = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = config or Config()

        self.checkpoint_path: Optional[str] = None
        self.ckpt_stage: Optional[str] = None
        self.ckpt_epoch: Optional[int] = None
        self.ckpt_best_psnr: Optional[float] = None
        self._loaded_model_config: dict = {}

        adapter_state = None
        if checkpoint_path:
            ckpt = self._read_checkpoint_cpu(checkpoint_path)
            self.checkpoint_path = str(checkpoint_path)
            self.ckpt_stage = ckpt.get("stage")
            self.ckpt_epoch = ckpt.get("epoch")
            self.ckpt_best_psnr = ckpt.get("best_psnr")
            if "model_config" in ckpt and isinstance(ckpt["model_config"], dict):
                self._loaded_model_config = dict(ckpt["model_config"])
                self._sync_config_from_dict(self._loaded_model_config)
            adapter_state = ckpt.get("trainable_state_dict") or ckpt.get("model_state_dict")
            del ckpt
            gc.collect()

        self.model = ControllableRestorer(self.config).to(self.device)
        if adapter_state is not None:
            self._apply_adapter_state(adapter_state)
            del adapter_state
            gc.collect()

        if self.device.type == "cuda" and self.config.USE_AMP:
            cast_frozen_base_to_fp16(self.model)

        self.model.eval()

    def _sync_config_from_dict(self, cfg_dict: dict) -> None:
        for key, val in cfg_dict.items():
            setattr(self.config, key, val)
        if hasattr(self.config, "DEGRADATION_TYPES"):
            self.config.NUM_ACTIONS = len(self.config.DEGRADATION_TYPES)

    @staticmethod
    def _read_checkpoint_cpu(checkpoint_path: str) -> dict:
        try:
            return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(checkpoint_path, map_location="cpu")

    def _apply_adapter_state(self, adapter_state: dict) -> Tuple[list, list]:
        missing_k, unexp_k = self.model.load_trainable_state_dict(adapter_state)
        if missing_k or unexp_k:
            print(
                f"[Warning] Checkpoint had {len(missing_k)} missing / {len(unexp_k)} legacy adapter keys. "
                f"Make sure you are loading a compatible V5 checkpoint!"
            )
        return missing_k, unexp_k

    def switch_checkpoint(self, checkpoint_path: str) -> dict:
        """Hot-swap adapter weights in-place (~0.2s) without reloading frozen backbones when compatible."""
        ckpt = self._read_checkpoint_cpu(checkpoint_path)
        new_cfg = ckpt.get("model_config") if isinstance(ckpt.get("model_config"), dict) else {}
        needs_rebuild = bool(self._loaded_model_config and new_cfg and new_cfg != self._loaded_model_config)

        self.checkpoint_path = str(checkpoint_path)
        self.ckpt_stage = ckpt.get("stage")
        self.ckpt_epoch = ckpt.get("epoch")
        self.ckpt_best_psnr = ckpt.get("best_psnr")
        adapter_state = ckpt.get("trainable_state_dict") or ckpt.get("model_state_dict")
        del ckpt
        gc.collect()

        if needs_rebuild:
            self._sync_config_from_dict(new_cfg)
            self._loaded_model_config = dict(new_cfg)
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = ControllableRestorer(self.config).to(self.device)
            if self.device.type == "cuda" and self.config.USE_AMP:
                cast_frozen_base_to_fp16(self.model)

        if adapter_state is not None:
            self._apply_adapter_state(adapter_state)
            del adapter_state
            gc.collect()

        self.model.eval()
        return {
            "path": self.checkpoint_path,
            "epoch": self.ckpt_epoch,
            "stage": self.ckpt_stage,
            "best_psnr": self.ckpt_best_psnr,
        }

    @staticmethod
    def _parse_point(point_str: str) -> Tuple[float, float]:
        parts = [p.strip() for p in point_str.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Expected '--point-mask x,y' (e.g. '140,120' or '0.5,0.5'), got '{point_str}'")
        return float(parts[0]), float(parts[1])

    @torch.no_grad()
    def restore(
        self,
        image_rgb_u8: np.ndarray,
        action: str = "all",
        bg_action: str = "none",
        point_mask: Optional[Tuple[float, float]] = None,
        mask_path: Optional[str] = None,
        mask_2d: Optional[np.ndarray] = None,
        mask_threshold: Optional[float] = None,
        feather_sigma: float = 6.0,
        steps: int = 1,
        fidelity_scale: Optional[float] = None,
        detail_scale: Optional[float] = None,
        return_trajectory: bool = False,
        tile_size: Optional[int] = None,
        tile_overlap: Optional[int] = None,
        lens_distortion_k1: float = 0.0,
        upscale: int = 1,
    ):
        """Restore an RGB uint8 image with 7 orthogonal controls, optional 2x/4x SR upscale & 1-click masks."""
        if abs(float(lens_distortion_k1)) > 1e-4:
            image_rgb_u8 = SelectiveDegradationPipeline.render_lens_distortion(
                image_rgb_u8, k1=float(lens_distortion_k1)
            )

        up_factor = max(1, int(upscale or 1))
        if up_factor > 1:
            h0, w0 = image_rgb_u8.shape[:2]
            image_rgb_u8 = cv2.resize(
                image_rgb_u8, (w0 * up_factor, h0 * up_factor), interpolation=cv2.INTER_CUBIC
            )
            if point_mask is not None:
                px, py = float(point_mask[0]), float(point_mask[1])
                if not (0.0 <= px <= 1.0 and 0.0 <= py <= 1.0 and (px < 1.0 or py < 1.0)):
                    point_mask = (px * up_factor, py * up_factor)

        h, w = image_rgb_u8.shape[:2]
        src_t = (
            torch.from_numpy(image_rgb_u8.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        signed_types = getattr(self.config, "SIGNED_ACTION_TYPES", ["exposure", "warmth"])
        fg_vec = parse_action_spec(action, self.config.DEGRADATION_TYPES, signed_types=signed_types)
        bg_vec = parse_action_spec(bg_action, self.config.DEGRADATION_TYPES, signed_types=signed_types)

        # When 2x/4x Super-Resolution Upscale is requested and action is not explicit 'none', engage SR slider!
        if up_factor > 1 and (action or "").strip().lower() not in ("none", "preserve", "zero", "identity"):
            sr_key = "super_resolution" if "super_resolution" in self.config.DEGRADATION_TYPES else "sharpen_focus"
            if sr_key in self.config.DEGRADATION_TYPES:
                sr_idx = self.config.DEGRADATION_TYPES.index(sr_key)
                if fg_vec[sr_idx] < 1e-4:
                    fg_vec[sr_idx] = 1.0

        resolved_mask = None
        if point_mask is not None:
            resolved_mask = extract_one_click_semantic_mask(
                self.model.semantic_encoder,
                src_t,
                point_xy=point_mask,
                threshold=mask_threshold,
                feather_sigma=feather_sigma,
            )
        elif mask_path is not None:
            resolved_mask = load_grayscale_mask(mask_path, height=h, width=w, feather_sigma=feather_sigma)
        elif mask_2d is not None:
            resolved_mask = np.clip(mask_2d.astype(np.float32), 0.0, 1.0)
            if resolved_mask.shape[:2] != (h, w):
                resolved_mask = cv2.resize(resolved_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        if resolved_mask is None:
            overlay = torch.from_numpy(fg_vec).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        else:
            overlay = build_spatial_action_overlay(
                height=h,
                width=w,
                fg_action=fg_vec,
                bg_action=bg_vec,
                mask_2d=resolved_mask,
                device=self.device,
            )

        effective_detail_scale = detail_scale
        if effective_detail_scale is None and self.ckpt_stage == "pixel_stage":
            effective_detail_scale = 0.0

        eff_tile = (
            int(getattr(self.config, "INFERENCE_TILE_SIZE", 512) or 0)
            if tile_size is None
            else int(tile_size)
        )
        eff_overlap = (
            int(getattr(self.config, "INFERENCE_TILE_OVERLAP", 96) or 0)
            if tile_overlap is None
            else int(tile_overlap)
        )
        if eff_tile > 0 and (h > eff_tile or w > eff_tile):
            print(
                f"Restoring {w}x{h} image via seamless {eff_tile}x{eff_tile} tiles "
                f"(overlap={eff_overlap}px)..."
            )

        use_amp = (self.device.type == "cuda") and self.config.USE_AMP
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = self.model.sample(
                src_01=src_t,
                action_input=overlay,
                steps=steps,
                fidelity_scale=fidelity_scale,
                detail_scale=effective_detail_scale,
                return_trajectory=return_trajectory,
                tile_size=eff_tile,
                tile_overlap=eff_overlap,
            )

        if return_trajectory:
            pred_t, traj_tensors = out
            traj_u8 = [
                np.clip(np.round(frame[0].permute(1, 2, 0).cpu().numpy() * 255.0), 0.0, 255.0).astype(np.uint8)
                for frame in traj_tensors
            ]
        else:
            pred_t = out
            traj_u8 = None

        pred_u8 = np.clip(
            np.round(pred_t[0].permute(1, 2, 0).cpu().numpy() * 255.0), 0.0, 255.0
        ).astype(np.uint8)
        return pred_u8, resolved_mask, traj_u8

    def benchmark(
        self,
        image_rgb_u8: np.ndarray,
        output_dir: str = "./restored",
        image_stem: str = "image",
        user_action: str = "all",
        point_mask: Optional[Tuple[float, float]] = None,
        steps: int = 1,
        benchmark_size: int = 512,
        seed: int = 1234,
        tile_size: Optional[int] = None,
        tile_overlap: Optional[int] = None,
    ):
        """Run Raw As-Is + 4-Scenario Synthetic Artifact Benchmark Suite on `image_rgb_u8`."""

        def _restore_cb(
            image_rgb_u8: np.ndarray,
            action: str = "all",
            bg_action: str = "none",
            point_mask: Optional[Tuple[float, float]] = None,
            mask_2d: Optional[np.ndarray] = None,
            steps: int = 1,
            fidelity_scale: Optional[float] = None,
            detail_scale: Optional[float] = None,
        ):
            return self.restore(
                image_rgb_u8=image_rgb_u8,
                action=action,
                bg_action=bg_action,
                point_mask=point_mask,
                mask_2d=mask_2d,
                steps=steps,
                fidelity_scale=fidelity_scale,
                detail_scale=detail_scale,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )

        return run_single_image_benchmark(
            image_rgb_u8=image_rgb_u8,
            config=self.config,
            restore_fn=_restore_cb,
            output_dir=output_dir,
            image_stem=image_stem,
            user_action=user_action,
            point_xy=point_mask,
            steps=steps,
            benchmark_size=benchmark_size,
            seed=seed,
            ckpt_stage=self.ckpt_stage,
        )


def main():
    parser = argparse.ArgumentParser(description="Controllable Pretrained Image Restoration CLI (V5)")
    parser.add_argument("--model", type=str, required=False, default=None, help="Path to model_best.pt / model_latest.pt")
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["sd-2.1-base", "sd-turbo", "sd-1.5", "bk-sdm-small", "tiny-sd", "bk-sdm-tiny", "mock_tiny"],
        default=None,
        help="Optional generator backbone override when running without a checkpoint",
    )
    parser.add_argument(
        "--vision-encoder",
        type=str,
        choices=["dinov2_reg", "mock_tiny"],
        default=None,
        help="Optional vision semantic encoder override when running without a checkpoint",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="./restored", help="Directory to save restored result")
    parser.add_argument(
        "--action",
        type=str,
        default="all",
        help="Control spec, e.g. 'denoise=1.0,motion=0.0,exposure=0.6,warmth=-0.4' or 'all'",
    )
    parser.add_argument(
        "--bg-action",
        type=str,
        default="none",
        help="Background control outside localized mask (default: 'none' = 100%% preserve)",
    )
    parser.add_argument(
        "--point-mask",
        type=str,
        default=None,
        help="1-click semantic object selection 'x,y' (pixels e.g. '128,140' or normalized '0.5,0.5')",
    )
    parser.add_argument("--mask-path", type=str, default=None, help="Optional grayscale mask PNG path")
    parser.add_argument("--mask-threshold", type=float, default=None, help="Optional cosine threshold for --point-mask")
    parser.add_argument("--feather-sigma", type=float, default=6.0, help="Mask edge feathering sigma in pixels")
    parser.add_argument(
        "--detail-scale",
        type=float,
        default=None,
        help="Generative Texture Detail dial in [0.0, 1.0] (0.0=pixel-faithful PSNR, 0.65-1.0=crisp texture)",
    )
    parser.add_argument("--fidelity-scale", type=float, default=None, help="Internal base adapter scale (default: 1.0)")
    parser.add_argument("--steps", type=int, default=1, help="Number of passes (default: 1)")
    parser.add_argument("--tile-size", type=int, default=None, help="Seamless tile size for 2K/4K photos (default: 512)")
    parser.add_argument("--tile-overlap", type=int, default=None, help="Overlap margin between tiles (default: 96)")
    parser.add_argument(
        "--upscale",
        type=int,
        choices=[1, 2, 4],
        default=1,
        help="Optional Super-Resolution output upscale factor (1=same size, 2=2x SR, 4=4x SR)",
    )
    parser.add_argument(
        "--lens-k1",
        type=float,
        default=0.0,
        help="Optional full-frame Brown-Conrady radial lens distortion correction k1 (-0.35..+0.35)",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run full Raw 'As-Is' + 4-Scenario Synthetic benchmark")
    parser.add_argument("--benchmark-size", type=int, default=512, help="Max side in px for benchmark panels (default: 512)")
    parser.add_argument("--benchmark-seed", type=int, default=1234, help="Deterministic seed for synthetic benchmark")
    parser.add_argument("--save-trajectory", action="store_true", help="Save intermediate frames")
    parser.add_argument("--mock", action="store_true", help="Run with offline mock_tiny backbones")
    args = parser.parse_args()

    cfg = Config()
    if args.backbone:
        cfg.GENERATOR_BACKBONE = args.backbone
    if args.vision_encoder:
        cfg.SEMANTIC_ENCODER_TYPE = args.vision_encoder
    if args.mock:
        cfg.GENERATOR_BACKBONE = "mock_tiny"
        cfg.SEMANTIC_ENCODER_TYPE = "mock_tiny"

    inferencer = ControllableInferencer(checkpoint_path=args.model, config=cfg)

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")
    rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    pt = inferencer._parse_point(args.point_mask) if args.point_mask else None
    stem = Path(args.image).stem

    if args.benchmark:
        inferencer.benchmark(
            image_rgb_u8=rgb_u8,
            output_dir=args.output_dir,
            image_stem=stem,
            user_action=args.action,
            point_mask=pt,
            steps=args.steps,
            benchmark_size=args.benchmark_size,
            seed=args.benchmark_seed,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
        )
        return

    restored_u8, mask_2d, traj_u8 = inferencer.restore(
        image_rgb_u8=rgb_u8,
        action=args.action,
        bg_action=args.bg_action,
        point_mask=pt,
        mask_path=args.mask_path,
        mask_threshold=args.mask_threshold,
        feather_sigma=args.feather_sigma,
        steps=args.steps,
        fidelity_scale=args.fidelity_scale,
        detail_scale=args.detail_scale,
        return_trajectory=args.save_trajectory,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        lens_distortion_k1=args.lens_k1,
        upscale=args.upscale,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_restored.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(restored_u8, cv2.COLOR_RGB2BGR))
    print(f"Saved restored image -> {out_path}")

    if mask_2d is not None:
        preview_u8 = render_mask_overlay_preview(rgb_u8, mask_2d, point_xy=pt)
        mask_out = out_dir / f"{stem}_mask_preview.png"
        cv2.imwrite(str(mask_out), cv2.cvtColor(preview_u8, cv2.COLOR_RGB2BGR))
        print(f"Saved localized mask preview -> {mask_out}")

    if traj_u8 is not None:
        traj_dir = out_dir / f"{stem}_trajectory"
        traj_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame_u8 in enumerate(traj_u8):
            cv2.imwrite(str(traj_dir / f"step_{idx:02d}.png"), cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))
        print(f"Saved {len(traj_u8)} trajectory frames -> {traj_dir}")


if __name__ == "__main__":
    main()
