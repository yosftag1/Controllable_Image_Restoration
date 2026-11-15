"""Interactive Gradio Web UI for Controllable Pretrained Image Restoration.

Designed to run effortlessly inside Kaggle / Colab / Jupyter notebooks or locally:
    # Inside a Kaggle or Colab notebook cell:
    !pip install -q gradio
    %run app.py --model ./checkpoints/model_best.pt

    # Or from terminal / shell cell:
    !python app.py --model ./checkpoints/model_best.pt --share
"""

import argparse
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import List, Optional, Tuple

if "/kaggle/working/Controllable_Image_Restoration" not in sys.path:
    sys.path.insert(0, "/kaggle/working/Controllable_Image_Restoration")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Gradio is required for app.py. In your notebook cell, run:\n"
        "    !pip install -q gradio\n"
        "and then re-run app.py!"
    ) from exc

import torch

from config import Config
from data.selective_dataset import SelectiveDegradationPipeline
from inference import ControllableInferencer
from utils.image_benchmark import compute_psnr_u8, compute_ssim_u8
from utils.semantic_mask import (
    extract_one_click_semantic_mask,
    parse_action_spec,
    render_mask_overlay_preview,
)

# Serialize GPU forward passes so device-scoped Dual-LoRA / Spatial Overlay contexts never overlap
_INFER_LOCK = threading.Lock()

SLIDER_SPECS: List[Tuple[str, str, str]] = [
    ("gaussian_noise", "Gaussian / Sensor Noise", "Removes ISO grain & color/luminance noise"),
    ("motion_blur", "Motion Blur", "Deblurs camera shake or subject motion streaks"),
    ("defocus_blur", "Defocus / Out-of-Focus Blur", "Sharpens optical defocus (keep at 0.0 to preserve bokeh!)"),
    ("jpeg_compression", "JPEG Compression Artifacts", "Removes 8x8 blockiness & mosquito ringing"),
    ("exposure_issue", "Exposure & Gamma", "Recovers dark underexposure or washed-out highlights"),
    ("color_degradation", "Color Cast & White Balance", "Neutralizes color tints & restores natural saturation"),
    ("color_banding", "Color Banding", "Smooths 8-bit sky/gradient posterization steps"),
    ("downsample_upsample", "Low-Res / 4x Super-Res Detail", "Synthesizes crisp high-frequency edges on upscaled inputs"),
]

RESTORE_PRESETS = {
    "Fix Everything ('all')": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "Identity / Zero-Action Lock ('none')": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Night Action Shot (Denoise + Exposure, KEEP Motion Blur)": [1.0, 0.0, 0.0, 0.3, 1.0, 0.5, 0.5, 0.0],
    "Portrait Cleanup (Denoise + Color + Banding, KEEP Bokeh)": [1.0, 0.0, 0.0, 0.5, 0.6, 1.0, 1.0, 0.0],
    "Old Web Image (DeJPEG + Crisp 4x Detail + Deband)": [0.4, 0.0, 0.3, 1.0, 0.3, 0.4, 1.0, 1.0],
    "Handheld Deblur Only (Motion + Defocus)": [0.2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.3],
}

SYNTH_RECIPES = {
    "Custom Sliders": None,
    "Handheld Low-Light (Motion + Noise + Dark Exposure)": {
        "gaussian_noise": 0.75,
        "motion_blur": 0.80,
        "defocus_blur": 0.0,
        "jpeg_compression": 0.0,
        "exposure_issue": 0.65,
        "color_degradation": 0.0,
        "color_banding": 0.0,
        "downsample_upsample": 0.0,
    },
    "Portrait Cast & Posterization (Defocus + Banding + Color Shift)": {
        "gaussian_noise": 0.0,
        "motion_blur": 0.0,
        "defocus_blur": 0.80,
        "jpeg_compression": 0.0,
        "exposure_issue": 0.0,
        "color_degradation": 0.85,
        "color_banding": 0.82,
        "downsample_upsample": 0.0,
    },
    "Compressed Low-Res Web Photo (4x Downsample + Heavy JPEG + Washed Exp)": {
        "gaussian_noise": 0.0,
        "motion_blur": 0.0,
        "defocus_blur": 0.0,
        "jpeg_compression": 0.80,
        "exposure_issue": 0.60,
        "color_degradation": 0.0,
        "color_banding": 0.0,
        "downsample_upsample": 0.85,
    },
}


def _in_ipython_kernel() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return False
        return "IPKernelApp" in getattr(ip, "config", {}) or ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _is_cloud_notebook() -> bool:
    return (
        os.path.exists("/kaggle")
        or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "COLAB_GPU" in os.environ
        or "COLAB_RELEASE_TAG" in os.environ
    )


def discover_checkpoints(explicit_path: Optional[str] = None) -> List[str]:
    """Locate `.pt` checkpoints across local folders, `/kaggle/working`, and `/kaggle/input`."""
    candidates: List[Path] = []
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            candidates.append(p.resolve())

    search_roots = [
        Path(PROJECT_ROOT) / "checkpoints",
        Path.cwd() / "checkpoints",
        Path("/kaggle/working/checkpoints"),
        Path("/kaggle/working/Controllable_Image_Restoration/checkpoints"),
    ]
    for root in search_roots:
        if root.is_dir():
            candidates.extend(sorted(root.glob("*.pt")))

    for extra_root in (Path("/kaggle/working"), Path("/kaggle/input")):
        if extra_root.is_dir():
            try:
                for pt in extra_root.rglob("*.pt"):
                    candidates.append(pt.resolve())
            except OSError:
                pass

    # Deduplicate and rank model_best.pt first
    seen = set()
    unique: List[Path] = []
    for p in candidates:
        key = str(p)
        if key not in seen and p.is_file():
            seen.add(key)
            unique.append(p)

    priority = {
        "model_best.pt": 0,
        "model_best_pixel.pt": 1,
        "model_latest.pt": 2,
    }
    unique.sort(key=lambda p: (priority.get(p.name, 10), p.name))
    return [str(p) for p in unique]


def _format_sliders_to_action_spec(slider_vals: List[float]) -> str:
    parts = []
    for (deg_key, _, _), val in zip(SLIDER_SPECS, slider_vals):
        parts.append(f"{deg_key}={float(val):.2f}")
    return ",".join(parts)


def _ensure_rgb_u8(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _make_side_by_side(before_u8: np.ndarray, after_u8: np.ndarray) -> np.ndarray:
    """Build a clean labeled Before | After comparison image when ImageSlider is unavailable."""
    h, w = before_u8.shape[:2]
    sep = np.full((h, 4, 3), 240, dtype=np.uint8)
    canvas = np.concatenate([before_u8, sep, after_u8], axis=1)
    cv2.putText(canvas, "Input", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 120), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Restored", (w + 16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 255, 160), 2, cv2.LINE_AA)
    return canvas


def _make_diff_heatmap(before_u8: np.ndarray, after_u8: np.ndarray, amp: float = 4.0) -> np.ndarray:
    """Render 4x amplified absolute pixel modification heatmap so zero-action locks are visually obvious."""
    diff = np.abs(after_u8.astype(np.float32) - before_u8.astype(np.float32)).mean(axis=2)
    scaled = np.clip(np.round(diff * amp), 0.0, 255.0).astype(np.uint8)
    bgr_heat = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr_heat, cv2.COLOR_BGR2RGB)


def _save_temp_png(rgb_u8: np.ndarray, prefix: str = "restored") -> str:
    out_dir = Path(tempfile.gettempdir()) / "controllable_restoration_gradio"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    out_path = out_dir / f"{prefix}_{ts}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
    return str(out_path)


def _format_ckpt_banner(inferencer: ControllableInferencer) -> str:
    ckpt_name = Path(inferencer.checkpoint_path).name if inferencer.checkpoint_path else "Untrained Base Weights (No Checkpoint)"
    stage = inferencer.ckpt_stage or "N/A"
    ep = f"Epoch {inferencer.ckpt_epoch}" if inferencer.ckpt_epoch is not None else "Epoch N/A"
    psnr = (
        f"{float(inferencer.ckpt_best_psnr):.2f} dB"
        if isinstance(inferencer.ckpt_best_psnr, (int, float)) and inferencer.ckpt_best_psnr > 0
        else "N/A"
    )
    dev_str = str(inferencer.device)
    if inferencer.device.type == "cuda" and torch.cuda.is_available():
        dev_str = f"{torch.cuda.get_device_name(inferencer.device)} ({inferencer.device})"
    return (
        f"**Active Checkpoint:** `{ckpt_name}` &nbsp;|&nbsp; "
        f"**Stage:** `{stage}` &nbsp;|&nbsp; **{ep}** (Best Val PSNR: `{psnr}`) &nbsp;|&nbsp; "
        f"**Device:** `{dev_str}` &nbsp;|&nbsp; "
        f"**Backbone:** `{inferencer.config.GENERATOR_BACKBONE}` + `{inferencer.config.SEMANTIC_ENCODER_TYPE}`"
    )


def build_gradio_app(
    inferencer: ControllableInferencer,
    discovered_ckpts: List[str],
):
    has_image_slider = hasattr(gr, "ImageSlider")
    synth_pipeline = SelectiveDegradationPipeline(inferencer.config)

    dropdown_kwargs = {
        "choices": discovered_ckpts,
        "value": inferencer.checkpoint_path if inferencer.checkpoint_path in discovered_ckpts else (discovered_ckpts[0] if discovered_ckpts else None),
        "label": "Checkpoint (.pt path — model_best.pt recommended)",
        "scale": 4,
    }
    try:
        ckpt_Probe = gr.Dropdown(allow_custom_value=True, **dropdown_kwargs)
        del ckpt_Probe
        dropdown_kwargs["allow_custom_value"] = True
    except TypeError:
        pass

    with gr.Blocks(title="Controllable Image Restoration Studio") as demo:
        gr.Markdown(
            "## Controllable Pretrained Image Restoration Studio\n"
            "Upload any image to fix specific degradations with **8 independent sliders**, dial between "
            "**Pixel-Faithful (`detail_scale=0`)** and **Semantic-Detail (`detail_scale=0.75–1.2`)** LoRAs, "
            "or **click directly on an object** in the input photo for localized restoration."
        )

        # ── Top Checkpoint Switcher Bar ──
        status_md = gr.Markdown(_format_ckpt_banner(inferencer))
        with gr.Accordion("Switch / Load Another Checkpoint (.pt)", open=not bool(inferencer.checkpoint_path)):
            with gr.Row():
                ckpt_dropdown = gr.Dropdown(**dropdown_kwargs)
                refresh_btn = gr.Button("Scan Checkpoints", scale=1)
                load_ckpt_btn = gr.Button("Load Checkpoint", variant="secondary", scale=1)

        # Shared state for 1-click object mask & synthetic clean ground-truth reference
        click_xy_state = gr.State(value=None)
        extracted_mask_state = gr.State(value=None)
        clean_ref_state = gr.State(value=None)

        with gr.Tabs() as main_tabs:
            # =========================================================================
            # TAB 1: Interactive Controllable Restoration (Real Photos + 1-Click Mask)
            # =========================================================================
            with gr.TabItem("1. Interactive Restore", id="tab_restore"):
                with gr.Row():
                    # Left Column: Input Image + Sliders + Localized Mask Controls
                    with gr.Column(scale=5):
                        input_image = gr.Image(
                            type="numpy",
                            label="Input Image (Upload photo — or click an object when '1-Click Object Mask' is active!)",
                        )

                        with gr.Row():
                            preset_dropdown = gr.Dropdown(
                                choices=list(RESTORE_PRESETS.keys()),
                                value="Fix Everything ('all')",
                                label="Quick Slider Preset",
                                scale=3,
                            )
                            restore_btn = gr.Button("Restore Image", variant="primary", scale=2)

                        gr.Markdown("#### Restoration Action Sliders (`0.0` = Preserve Untouched, `1.0` = Full Fix)")
                        slider_controls = []
                        for row_idx in range(0, len(SLIDER_SPECS), 2):
                            with gr.Row():
                                for _, label_txt, info_txt in SLIDER_SPECS[row_idx : row_idx + 2]:
                                    s = gr.Slider(
                                        minimum=0.0,
                                        maximum=1.0,
                                        value=1.0,
                                        step=0.05,
                                        label=label_txt,
                                        info=info_txt,
                                    )
                                    slider_controls.append(s)

                        with gr.Accordion("Dual-LoRA Quality & Tiling Dials", open=True):
                            with gr.Row():
                                fidelity_slider = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.5,
                                    value=float(inferencer.config.DEFAULT_FIDELITY_SCALE),
                                    step=0.05,
                                    label="Pixel-Faithful LoRA (fidelity_scale)",
                                    info="1.0 = strict structural & color accuracy",
                                )
                                default_det = (
                                    0.0
                                    if inferencer.ckpt_stage in ("align_warmup", "pixel_stage")
                                    else float(inferencer.config.DEFAULT_DETAIL_SCALE)
                                )
                                detail_slider = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.5,
                                    value=default_det,
                                    step=0.05,
                                    label="Semantic-Detail LoRA (detail_scale)",
                                    info="0.0 = conservative Stage-2 mode | 0.75-1.1 = crisp Stage-3 texture",
                                )
                            with gr.Row():
                                steps_slider = gr.Slider(
                                    minimum=1,
                                    maximum=4,
                                    value=1,
                                    step=1,
                                    label="Flow Refinement Steps",
                                    info="1 step native SD-Turbo (2 steps for extra heavy blur/SR)",
                                )
                                tile_slider = gr.Slider(
                                    minimum=0,
                                    maximum=1024,
                                    value=int(getattr(inferencer.config, "INFERENCE_TILE_SIZE", 512) or 512),
                                    step=64,
                                    label="Seamless Tile Size (px)",
                                    info="512 recommended for 2K/4K photos (0 = untiled)",
                                )

                        with gr.Accordion("Localized Spatial Mask / 1-Click Semantic Object Control", open=False):
                            mask_mode = gr.Radio(
                                choices=[
                                    "Global (Whole Image)",
                                    "1-Click Object Mask (Click on Input Image above)",
                                    "Upload Grayscale Mask",
                                ],
                                value="Global (Whole Image)",
                                label="Spatial Control Mode",
                            )
                            bg_action_box = gr.Textbox(
                                value="none",
                                label="Background Action (outside mask)",
                                info="Examples: 'none' (100% preserve background), 'noise=1.0,motion=0.0' (denoise bg, keep bg motion blur)",
                            )
                            with gr.Row():
                                feather_slider = gr.Slider(
                                    minimum=0.0,
                                    maximum=24.0,
                                    value=6.0,
                                    step=1.0,
                                    label="Mask Edge Feathering (px)",
                                )
                                mask_thresh_slider = gr.Slider(
                                    minimum=0.0,
                                    maximum=0.95,
                                    value=0.0,
                                    step=0.05,
                                    label="Cosine Threshold (0 = Auto Otsu)",
                                )
                                clear_mask_btn = gr.Button("Clear Click Mask")
                            custom_mask_upload = gr.Image(
                                type="numpy",
                                label="Custom Grayscale Mask (used when 'Upload Grayscale Mask' is selected)",
                            )
                            mask_preview_img = gr.Image(
                                type="numpy",
                                label="Active Semantic / Spatial Mask Preview",
                                interactive=False,
                            )

                    # Right Column: Before/After Slider + Restored Image + Diff Heatmap + Download
                    with gr.Column(scale=6):
                        if has_image_slider:
                            compare_view = gr.ImageSlider(
                                type="numpy",
                                label="Interactive Before / After Drag Comparison",
                            )
                        else:
                            compare_view = gr.Image(
                                type="numpy",
                                label="Before | After Side-by-Side Comparison",
                                interactive=False,
                            )

                        metrics_md = gr.Markdown("*Upload an image and click **Restore Image** to run inference.*")

                        with gr.Tabs():
                            with gr.TabItem("Restored Full Image"):
                                restored_out = gr.Image(
                                    type="numpy",
                                    label="Restored Output",
                                    interactive=False,
                                )
                            with gr.TabItem("4x Change Heatmap (Zero-Action Check)"):
                                diff_heatmap_out = gr.Image(
                                    type="numpy",
                                    label="4x Amplified |Restored - Input| Heatmap",
                                    interactive=False,
                                )

                        download_png = gr.File(label="Download Lossless Full-Resolution PNG")

            # =========================================================================
            # TAB 2: Synthetic Degradation Test Lab (Quantitative PSNR / SSIM Check)
            # =========================================================================
            with gr.TabItem("2. Synthetic Degradation Test Lab (PSNR / SSIM)", id="tab_synth"):
                gr.Markdown(
                    "Upload any **clean sharp photo**, inject controlled camera/compression degradations, "
                    "and send it directly to **Tab 1** so every restoration run automatically reports exact "
                    "**PSNR (dB) & SSIM gains** against your clean ground truth!"
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        clean_upload = gr.Image(type="numpy", label="Clean Ground-Truth Photo")
                        with gr.Row():
                            synth_recipe_dd = gr.Dropdown(
                                choices=list(SYNTH_RECIPES.keys()),
                                value="Handheld Low-Light (Motion + Noise + Dark Exposure)",
                                label="Degradation Recipe Preset",
                                scale=3,
                            )
                            synth_seed_num = gr.Number(value=42, precision=0, label="Random Seed", scale=1)

                        synth_sliders = []
                        default_recipe = SYNTH_RECIPES["Handheld Low-Light (Motion + Noise + Dark Exposure)"]
                        for row_idx in range(0, len(SLIDER_SPECS), 2):
                            with gr.Row():
                                for deg_key, label_txt, _ in SLIDER_SPECS[row_idx : row_idx + 2]:
                                    s = gr.Slider(
                                        minimum=0.0,
                                        maximum=1.0,
                                        value=float(default_recipe.get(deg_key, 0.0)),
                                        step=0.05,
                                        label=f"Inject {label_txt}",
                                    )
                                    synth_sliders.append(s)

                        with gr.Row():
                            synth_btn = gr.Button("1. Synthesize Degraded Image", variant="secondary")
                            send_to_restore_btn = gr.Button(
                                "2. Send to Tab 1 (With Clean GT Reference)", variant="primary"
                            )

                    with gr.Column(scale=6):
                        synth_preview = gr.Image(
                            type="numpy", label="Synthesized Degraded Output", interactive=False
                        )
                        synth_info_md = gr.Markdown("*Upload a clean photo and click **Synthesize Degraded Image**.*")

            # =========================================================================
            # TAB 3: 1-Click Automated Multi-Scenario Benchmark Suite
            # =========================================================================
            with gr.TabItem("3. Full Benchmark Contact Sheets", id="tab_bench"):
                gr.Markdown(
                    "Runs `ControllableInferencer.benchmark()` on any uploaded photo to generate the complete "
                    "**Raw As-Is Slider Sweep** + **4-Scenario Synthetic Stress-Test Sheet** with per-tile PSNR/SSIM."
                )
                with gr.Row():
                    bench_input = gr.Image(type="numpy", label="Benchmark Input Image")
                    with gr.Column():
                        bench_size_slider = gr.Slider(
                            minimum=256,
                            maximum=768,
                            value=512,
                            step=64,
                            label="Benchmark Tile Max Side (px)",
                        )
                        bench_steps_slider = gr.Slider(
                            minimum=1,
                            maximum=4,
                            value=1,
                            step=1,
                            label="Inference Steps",
                        )
                        run_bench_btn = gr.Button("Run Full Benchmark Suite", variant="primary")
                        bench_status_md = gr.Markdown("")
                bench_raw_sheet = gr.Image(type="numpy", label="Sheet 1: Raw 'As-Is' Controllable Sweep", interactive=False)
                bench_synth_sheet = gr.Image(type="numpy", label="Sheet 2: 4-Scenario Synthetic Stress Test", interactive=False)

        # -------------------------------------------------------------------------
        # Event Callbacks
        # -------------------------------------------------------------------------
        def _on_refresh_checkpoints():
            found = discover_checkpoints()
            val = inferencer.checkpoint_path if inferencer.checkpoint_path in found else (found[0] if found else None)
            return gr.update(choices=found, value=val)

        def _on_load_checkpoint(ckpt_path: str):
            if not ckpt_path or not str(ckpt_path).strip():
                raise gr.Error("Please select or enter a valid .pt checkpoint path.")
            ckpt_path = str(ckpt_path).strip()
            if not os.path.isfile(ckpt_path):
                raise gr.Error(f"Checkpoint file not found: {ckpt_path}")
            with _INFER_LOCK:
                info = inferencer.switch_checkpoint(ckpt_path)
            new_det = (
                0.0
                if info.get("stage") in ("align_warmup", "pixel_stage")
                else float(inferencer.config.DEFAULT_DETAIL_SCALE)
            )
            return _format_ckpt_banner(inferencer), gr.update(value=new_det)

        refresh_btn.click(_on_refresh_checkpoints, outputs=[ckpt_dropdown])
        load_ckpt_btn.click(_on_load_checkpoint, inputs=[ckpt_dropdown], outputs=[status_md, detail_slider])

        def _on_preset_change(preset_name: str):
            vals = RESTORE_PRESETS.get(preset_name)
            if vals is None:
                return [gr.update() for _ in SLIDER_SPECS]
            return [gr.update(value=v) for v in vals]

        preset_dropdown.change(_on_preset_change, inputs=[preset_dropdown], outputs=slider_controls)

        def _on_new_user_upload(img_np):
            # When the user manually uploads/clears an image in Tab 1, clear stale click masks & synthetic GT
            return None, None, None, None

        input_image.upload(
            _on_new_user_upload,
            inputs=[input_image],
            outputs=[click_xy_state, extracted_mask_state, clean_ref_state, mask_preview_img],
        )
        input_image.clear(
            _on_new_user_upload,
            inputs=[input_image],
            outputs=[click_xy_state, extracted_mask_state, clean_ref_state, mask_preview_img],
        )

        def _extract_click_mask_helper(
            img_np: Optional[np.ndarray],
            xy: Optional[Tuple[float, float]],
            feather_px: float,
            thresh_val: float,
        ):
            rgb = _ensure_rgb_u8(img_np)
            if rgb is None or xy is None:
                return None, None
            src_t = (
                torch.from_numpy(rgb.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(inferencer.device)
            )
            eff_thresh = float(thresh_val) if float(thresh_val) > 1e-4 else None
            with _INFER_LOCK:
                mask_2d = extract_one_click_semantic_mask(
                    inferencer.model.semantic_encoder,
                    src_t,
                    point_xy=xy,
                    threshold=eff_thresh,
                    feather_sigma=float(feather_px),
                )
            preview = render_mask_overlay_preview(rgb, mask_2d, point_xy=xy)
            return mask_2d, preview

        def _on_image_click(
            img_np: Optional[np.ndarray],
            cur_mask_mode: str,
            feather_px: float,
            thresh_val: float,
            evt: gr.SelectData,
        ):
            if img_np is None or evt is None or evt.index is None:
                return None, None, None, gr.update()
            xy = (float(evt.index[0]), float(evt.index[1]))
            mask_2d, preview = _extract_click_mask_helper(img_np, xy, feather_px, thresh_val)
            next_mode = (
                "1-Click Object Mask (Click on Input Image above)"
                if cur_mask_mode == "Global (Whole Image)"
                else cur_mask_mode
            )
            return xy, mask_2d, preview, gr.update(value=next_mode)

        input_image.select(
            _on_image_click,
            inputs=[input_image, mask_mode, feather_slider, mask_thresh_slider],
            outputs=[click_xy_state, extracted_mask_state, mask_preview_img, mask_mode],
        )

        def _on_clear_mask():
            return None, None, None, gr.update(value="Global (Whole Image)")

        clear_mask_btn.click(
            _on_clear_mask,
            outputs=[click_xy_state, extracted_mask_state, mask_preview_img, mask_mode],
        )

        def _run_restore(
            img_np: Optional[np.ndarray],
            mode_str: str,
            bg_spec: str,
            feather_px: float,
            thresh_val: float,
            custom_mask_np: Optional[np.ndarray],
            click_xy: Optional[Tuple[float, float]],
            cached_mask: Optional[np.ndarray],
            clean_gt_np: Optional[np.ndarray],
            f_scale: float,
            d_scale: float,
            steps_val: int,
            tile_val: int,
            *slider_vals,
        ):
            rgb = _ensure_rgb_u8(img_np)
            if rgb is None:
                raise gr.Error("Please upload an input image first!")

            fg_spec = _format_sliders_to_action_spec(list(slider_vals))
            bg_clean = (bg_spec or "none").strip()
            # Validate bg action syntax nicely for UI feedback
            try:
                parse_action_spec(bg_clean, inferencer.config.DEGRADATION_TYPES)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc

            active_mask = None
            mask_preview = None
            if mode_str == "1-Click Object Mask (Click on Input Image above)":
                if cached_mask is not None and cached_mask.shape[:2] == rgb.shape[:2]:
                    active_mask = cached_mask
                    mask_preview = render_mask_overlay_preview(rgb, active_mask, point_xy=click_xy)
                elif click_xy is not None:
                    active_mask, mask_preview = _extract_click_mask_helper(rgb, click_xy, feather_px, thresh_val)
                else:
                    raise gr.Error(
                        "1-Click Object Mask mode is selected — click any object on the Input Image first "
                        "(or switch Spatial Control Mode back to 'Global')!"
                    )
            elif mode_str == "Upload Grayscale Mask":
                if custom_mask_np is None:
                    raise gr.Error("Please upload a grayscale mask image or switch Spatial Control Mode to 'Global'.")
                m_arr = np.asarray(custom_mask_np)
                if m_arr.ndim == 3:
                    m_arr = cv2.cvtColor(m_arr, cv2.COLOR_RGB2GRAY)
                if m_arr.shape[:2] != rgb.shape[:2]:
                    m_arr = cv2.resize(m_arr, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                m_f = np.clip(m_arr.astype(np.float32) / 255.0, 0.0, 1.0)
                if float(feather_px) > 0.0:
                    m_f = cv2.GaussianBlur(m_f, (0, 0), sigmaX=float(feather_px), sigmaY=float(feather_px))
                active_mask = np.clip(m_f, 0.0, 1.0)
                mask_preview = render_mask_overlay_preview(rgb, active_mask)

            t0 = time.perf_counter()
            with _INFER_LOCK:
                restored_u8, _, _ = inferencer.restore(
                    image_rgb_u8=rgb,
                    action=fg_spec,
                    bg_action=bg_clean,
                    mask_2d=active_mask,
                    steps=int(steps_val),
                    fidelity_scale=float(f_scale),
                    detail_scale=float(d_scale),
                    tile_size=int(tile_val),
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            diff_abs = np.abs(restored_u8.astype(np.float32) - rgb.astype(np.float32))
            mean_px_diff = float(diff_abs.mean())
            max_px_diff = float(diff_abs.max())
            heatmap_u8 = _make_diff_heatmap(rgb, restored_u8, amp=4.0)
            png_path = _save_temp_png(restored_u8, prefix="restored")

            h, w = rgb.shape[:2]
            summary_parts = [
                f"**Resolution:** `{w}×{h}`",
                f"**Latency:** `{elapsed_ms:.0f} ms` (`{int(steps_val)}` step{'s' if int(steps_val) > 1 else ''})",
                f"**Dual-LoRA:** `λ_pix={float(f_scale):.2f}, λ_sem={float(d_scale):.2f}`",
                f"**Pixel Δ:** mean `{mean_px_diff:.2f}` / max `{max_px_diff:.1f}` (out of 255)",
            ]
            if clean_gt_np is not None and clean_gt_np.shape == restored_u8.shape:
                in_psnr = compute_psnr_u8(rgb, clean_gt_np)
                out_psnr = compute_psnr_u8(restored_u8, clean_gt_np)
                in_ssim = compute_ssim_u8(rgb, clean_gt_np)
                out_ssim = compute_ssim_u8(restored_u8, clean_gt_np)
                gain = out_psnr - in_psnr
                summary_parts.append(
                    f"**vs. Clean GT:** `{in_psnr:.2f} dB -> {out_psnr:.2f} dB ({gain:+.2f} dB)` | "
                    f"SSIM `{in_ssim:.3f} -> {out_ssim:.3f}`"
                )

            comp_payload = (rgb, restored_u8) if has_image_slider else _make_side_by_side(rgb, restored_u8)
            return (
                comp_payload,
                restored_u8,
                heatmap_u8,
                png_path,
                " &nbsp;|&nbsp; ".join(summary_parts),
                mask_preview if mask_preview is not None else gr.update(),
            )

        restore_btn.click(
            _run_restore,
            inputs=[
                input_image,
                mask_mode,
                bg_action_box,
                feather_slider,
                mask_thresh_slider,
                custom_mask_upload,
                click_xy_state,
                extracted_mask_state,
                clean_ref_state,
                fidelity_slider,
                detail_slider,
                steps_slider,
                tile_slider,
                *slider_controls,
            ],
            outputs=[
                compare_view,
                restored_out,
                diff_heatmap_out,
                download_png,
                metrics_md,
                mask_preview_img,
            ],
        )

        # ── Tab 2 Callbacks: Synthetic Degradation Test Lab ──
        def _on_synth_recipe(recipe_name: str):
            rec = SYNTH_RECIPES.get(recipe_name)
            if rec is None:
                return [gr.update() for _ in SLIDER_SPECS]
            return [gr.update(value=float(rec.get(k, 0.0))) for k, _, _ in SLIDER_SPECS]

        synth_recipe_dd.change(_on_synth_recipe, inputs=[synth_recipe_dd], outputs=synth_sliders)

        def _synthesize_degraded(clean_np: Optional[np.ndarray], seed_val: float, *sev_vals):
            clean_rgb = _ensure_rgb_u8(clean_np)
            if clean_rgb is None:
                raise gr.Error("Upload a clean photo first!")
            h, w = clean_rgb.shape[:2]
            rng = np.random.default_rng(int(seed_val or 42))
            params = synth_pipeline.sample_params(h, w, rng=rng)
            sev_vec = np.array([float(v) for v in sev_vals], dtype=np.float32)
            deg_u8 = synth_pipeline.render_with_params(clean_rgb, sev_vec, params)
            in_psnr = compute_psnr_u8(deg_u8, clean_rgb)
            in_ssim = compute_ssim_u8(deg_u8, clean_rgb)
            info = (
                f"**Synthesized Degraded Input:** `{w}×{h}` &nbsp;|&nbsp; "
                f"**Input PSNR:** `{in_psnr:.2f} dB` &nbsp;|&nbsp; **Input SSIM:** `{in_ssim:.3f}`\n\n"
                f"Click **2. Send to Tab 1** to test selective or full restoration with live PSNR gain tracking!"
            )
            return deg_u8, clean_rgb, info

        synth_btn.click(
            _synthesize_degraded,
            inputs=[clean_upload, synth_seed_num, *synth_sliders],
            outputs=[synth_preview, clean_ref_state, synth_info_md],
        )

        def _send_synth_to_tab1(deg_np: Optional[np.ndarray], clean_np: Optional[np.ndarray], *sev_vals):
            deg_rgb = _ensure_rgb_u8(deg_np)
            if deg_rgb is None:
                raise gr.Error("Click '1. Synthesize Degraded Image' first!")
            clean_rgb = _ensure_rgb_u8(clean_np)
            # Pre-set restoration sliders to 1.0 on active degradations and 0.0 on untouched ones!
            suggested = [1.0 if float(v) > 0.02 else 0.0 for v in sev_vals]
            if all(v == 0.0 for v in suggested):
                suggested = [1.0] * len(SLIDER_SPECS)
            return (
                deg_rgb,
                clean_rgb,
                None,
                None,
                None,
                gr.update(selected="tab_restore"),
                *[gr.update(value=v) for v in suggested],
            )

        send_to_restore_btn.click(
            _send_synth_to_tab1,
            inputs=[synth_preview, clean_upload, *synth_sliders],
            outputs=[
                input_image,
                clean_ref_state,
                click_xy_state,
                extracted_mask_state,
                mask_preview_img,
                main_tabs,
                *slider_controls,
            ],
        )

        # ── Tab 3 Callback: Full Benchmark Sheets ──
        def _run_benchmark_suite(img_np: Optional[np.ndarray], b_size: int, b_steps: int):
            rgb = _ensure_rgb_u8(img_np)
            if rgb is None:
                raise gr.Error("Please upload an image for the benchmark first!")
            out_dir = Path(tempfile.gettempdir()) / "controllable_restoration_gradio" / "benchmark"
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            with _INFER_LOCK:
                res = inferencer.benchmark(
                    image_rgb_u8=rgb,
                    output_dir=str(out_dir),
                    image_stem="gradio_bench",
                    steps=int(b_steps),
                    benchmark_size=int(b_size),
                )
            dt = time.perf_counter() - t0
            outputs_map = res.get("outputs", {})
            raw_bgr = cv2.imread(outputs_map.get("raw_asis_sweep_png", ""))
            syn_bgr = cv2.imread(outputs_map.get("synthetic_artifact_sweep_png", ""))
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB) if raw_bgr is not None else None
            syn_rgb = cv2.cvtColor(syn_bgr, cv2.COLOR_BGR2RGB) if syn_bgr is not None else None
            return (
                raw_rgb,
                syn_rgb,
                f"Benchmark completed in `{dt:.1f}s`! Saved sheets & JSON metrics in `{out_dir}`.",
            )

        run_bench_btn.click(
            _run_benchmark_suite,
            inputs=[bench_input, bench_size_slider, bench_steps_slider],
            outputs=[bench_raw_sheet, bench_synth_sheet, bench_status_md],
        )

    return demo


def launch_app(
    checkpoint_path: Optional[str] = None,
    share: Optional[bool] = None,
    port: Optional[int] = None,
    mock: bool = False,
    block: Optional[bool] = None,
):
    """Python-callable launcher ideal for Kaggle / Colab / Jupyter notebook cells."""
    try:
        gr.close_all()
    except Exception:
        pass

    discovered = discover_checkpoints(checkpoint_path)
    resolved_ckpt = checkpoint_path
    if resolved_ckpt is None and discovered:
        resolved_ckpt = discovered[0]
        print(f"[Gradio] Auto-selected checkpoint: {resolved_ckpt}", flush=True)
    elif resolved_ckpt is not None and not os.path.isfile(resolved_ckpt):
        print(f"[Warning] Checkpoint '{resolved_ckpt}' not found; starting with base weights.", flush=True)
        resolved_ckpt = None

    cfg = Config()
    if mock:
        cfg.GENERATOR_BACKBONE = "mock_tiny"
        cfg.SEMANTIC_ENCODER_TYPE = "mock_tiny"

    inferencer = ControllableInferencer(checkpoint_path=resolved_ckpt, config=cfg)
    demo = build_gradio_app(inferencer, discovered)

    in_nb = _in_ipython_kernel()
    use_share = (_is_cloud_notebook() or in_nb) if share is None else bool(share)
    # In an interactive IPython cell (%run app.py), non-blocking keeps the server + public URL alive
    # while letting you inspect logs or run follow-up cells; in CLI (!python app.py), block so process stays alive!
    prevent_lock = in_nb if block is None else (not bool(block))

    launch_kwargs = {
        "share": use_share,
        "inline": in_nb,
        "prevent_thread_lock": prevent_lock,
    }
    if port is not None:
        launch_kwargs["server_port"] = int(port)
    if _is_cloud_notebook():
        launch_kwargs["server_name"] = "0.0.0.0"

    return demo.launch(**launch_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio Web UI for Controllable Image Restoration")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to checkpoint (auto-discovers model_best.pt in ./checkpoints or /kaggle if omitted)",
    )
    parser.add_argument("--share", action="store_true", default=None, help="Create a public *.gradio.live share URL")
    parser.add_argument("--no-share", action="store_true", help="Disable public *.gradio.live share tunnel")
    parser.add_argument("--port", type=int, default=None, help="Optional local server port (e.g. 7860)")
    parser.add_argument("--block", action="store_true", help="Keep cell blocked even inside IPython (%run)")
    parser.add_argument("--mock", action="store_true", help="Use tiny offline mock backbones for quick UI testing")
    args, _ = parser.parse_known_args()

    share_pref = False if args.no_share else (True if args.share else None)
    launch_app(
        checkpoint_path=args.model,
        share=share_pref,
        port=args.port,
        mock=args.mock,
        block=True if args.block else None,
    )


if __name__ == "__main__":
    main()
