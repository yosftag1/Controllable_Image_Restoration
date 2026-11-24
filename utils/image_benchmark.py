"""Single-Image Benchmark Suite: Raw 'As-Is' Sweep + Controlled Synthetic Artifact Stress Test (V4).

Evaluates all 6 orthogonal controls:
- 4 objective [0, 1] restoration sliders (`denoise`, `deartifact`, `deblur_motion`, `sharpen_focus`)
- 2 directional [-1, +1] tone/WB sliders (`exposure`: darken/brighten, `warmth`: cool/warm)
- Single `detail_scale` generative texture dial (`0.0` = pixel-faithful vs. `0.70` = crisp detail)
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from data.selective_dataset import SelectiveDegradationPipeline
from .semantic_mask import parse_action_spec, render_mask_overlay_preview


def compute_psnr_u8(pred_u8: np.ndarray, ref_u8: np.ndarray) -> float:
    """Compute PSNR (dB) between two uint8 RGB images (returns 99.99 dB on exact match)."""
    mse = float(np.mean((pred_u8.astype(np.float32) - ref_u8.astype(np.float32)) ** 2))
    if mse <= 1e-10:
        return 99.99
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def compute_ssim_u8(pred_u8: np.ndarray, ref_u8: np.ndarray) -> float:
    """Standard 11x11 sigma=1.5 Gaussian-windowed RGB SSIM using OpenCV."""
    x = pred_u8.astype(np.float32)
    y = ref_u8.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
    sig_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
    sig_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sig_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2)
    return float(np.mean(num / np.maximum(den, 1e-8)))


def compute_laplacian_sharpness(rgb_u8: np.ndarray) -> float:
    """Estimate high-frequency edge/noise energy via RMS 3x3 Laplacian."""
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(np.sqrt(np.mean(lap * lap)))


def _prepare_scene_and_crop(
    rgb_u8: np.ndarray, benchmark_size: int = 512
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return proportional `scene_rgb` (<= benchmark_size) and optional native 1:1 `center_crop_rgb`."""
    h, w = rgb_u8.shape[:2]
    if benchmark_size <= 0 or max(h, w) <= benchmark_size:
        return rgb_u8.copy(), None

    scale = float(benchmark_size) / float(max(h, w))
    nh = max(64, int(round((h * scale) / 8.0)) * 8)
    nw = max(64, int(round((w * scale) / 8.0)) * 8)
    scene_rgb = cv2.resize(rgb_u8, (nw, nh), interpolation=cv2.INTER_AREA)

    ch = min(h, nh)
    cw = min(w, nw)
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    crop = rgb_u8[y0 : y0 + ch, x0 : x0 + cw].copy()
    if crop.shape[:2] != (nh, nw):
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return scene_rgb, crop


def _render_annotated_sheet(
    rows_data: List[Tuple[str, List[Tuple[str, np.ndarray, str]]]],
    sheet_title: str,
) -> np.ndarray:
    """Render a multi-row labeled contact sheet with per-row banners, column headers, and metric footers."""
    first_img = rows_data[0][1][0][1]
    th, tw = first_img.shape[:2]
    num_cols = len(rows_data[0][1])

    pad = 8
    top_title_h = 42
    row_banner_h = 30
    col_header_h = 26
    footer_h = 24

    sheet_w = num_cols * (tw + pad) + pad
    row_block_h = row_banner_h + col_header_h + th + footer_h + pad
    sheet_h = top_title_h + len(rows_data) * row_block_h + pad

    canvas = np.full((sheet_h, sheet_w, 3), 18, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (sheet_w, top_title_h - 2), (28, 34, 42), -1)
    cv2.putText(
        canvas,
        sheet_title,
        (pad + 4, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (240, 245, 250),
        2,
        cv2.LINE_AA,
    )

    col_font = float(np.clip(tw / 640.0, 0.36, 0.48))
    sub_font = float(np.clip(tw / 700.0, 0.33, 0.44))

    y_cursor = top_title_h
    for row_title, cells in rows_data:
        cv2.rectangle(
            canvas,
            (pad, y_cursor),
            (sheet_w - pad, y_cursor + row_banner_h - 2),
            (32, 46, 42),
            -1,
        )
        cv2.putText(
            canvas,
            row_title,
            (pad + 8, y_cursor + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (130, 240, 190),
            1,
            cv2.LINE_AA,
        )
        y_col_hdr = y_cursor + row_banner_h
        y_img = y_col_hdr + col_header_h
        y_foot = y_img + th

        for c_idx, (col_label, img_rgb, metric_text) in enumerate(cells):
            x0 = pad + c_idx * (tw + pad)
            if img_rgb.shape[:2] != (th, tw):
                img_rgb = cv2.resize(img_rgb, (tw, th), interpolation=cv2.INTER_AREA)

            cv2.rectangle(canvas, (x0, y_col_hdr), (x0 + tw, y_img), (28, 30, 35), -1)
            cv2.putText(
                canvas,
                col_label,
                (x0 + 5, y_col_hdr + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                col_font,
                (235, 238, 242),
                1,
                cv2.LINE_AA,
            )

            canvas[y_img : y_img + th, x0 : x0 + tw] = img_rgb

            cv2.rectangle(canvas, (x0, y_foot), (x0 + tw, y_foot + footer_h), (25, 28, 34), -1)
            cv2.putText(
                canvas,
                metric_text,
                (x0 + 5, y_foot + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                sub_font,
                (175, 215, 250),
                1,
                cv2.LINE_AA,
            )

        y_cursor += row_block_h

    return canvas


def run_single_image_benchmark(
    image_rgb_u8: np.ndarray,
    config,
    restore_fn: Callable,
    output_dir: str,
    image_stem: str = "image",
    user_action: str = "all",
    point_xy: Optional[Tuple[float, float]] = None,
    steps: int = 1,
    benchmark_size: int = 512,
    seed: int = 1234,
    ckpt_stage: Optional[str] = None,
) -> Dict:
    """Run Raw As-Is and 4-Scenario Directional + Selective Artifact benchmarks on a single image."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    h_orig, w_orig = image_rgb_u8.shape[:2]
    click_pt = point_xy if point_xy is not None else (0.5, 0.5)
    signed_types = getattr(config, "SIGNED_ACTION_TYPES", ["exposure", "warmth"])

    if ckpt_stage == "pixel_stage":
        print(
            f"[Benchmark Note] Loaded checkpoint stage is '{ckpt_stage}' (Semantic-LoRA has not run Stage 2 yet); "
            "Pixel-Faithful and Detail columns will match until semantic_stage completes."
        )

    scene_rgb, crop_rgb = _prepare_scene_and_crop(image_rgb_u8, benchmark_size=benchmark_size)

    # =========================================================================
    # PART 1: RAW IMAGE "AS-IS" SWEEP (Zero Synthetic Degradation Added)
    # =========================================================================
    print(f"\n[1/3] Running Raw Image 'As-Is' Slider Sweep on '{image_stem}' ({w_orig}x{h_orig})...")
    raw_specs = [
        ("2. Identity (none)", "none", "none", False, 1.0, 0.0),
        ("3. Denoise+Deartifact", "denoise=1.0,deartifact=1.0", "none", False, 1.0, 0.0),
        ("4. Motion+Focus Sharp", "deblur_motion=1.0,sharpen_focus=1.0", "none", False, 1.0, 0.55),
        ("5. Brighten + Warm", "exposure=0.40,warmth=0.35", "none", False, 1.0, 0.0),
        ("6. Darken + Cool", "exposure=-0.35,warmth=-0.35", "none", False, 1.0, 0.0),
        ("7. Restore All (Pixel)", "all", "none", False, 1.0, 0.0),
        ("8. Restore All (Detail)", "all", "none", False, 1.0, 0.70),
        ("9. 1-Click Obj Mask", "all", "none", True, 1.0, 0.70),
    ]

    raw_rows_visual = []
    raw_metrics_report: Dict[str, List[Dict]] = {}
    saved_raw_mask_preview = None

    views = [("Row 1: Full Scene As-Is (No Added Artifacts)", scene_rgb)]
    if crop_rgb is not None:
        views.append(("Row 2: Native 1:1 Center Crop As-Is (True Sensor/Pixel Scale)", crop_rgb))

    for view_idx, (view_title, base_u8) in enumerate(views):
        base_sharp = max(1e-4, compute_laplacian_sharpness(base_u8))
        cells = [("1. Raw Input (As-Is)", base_u8, f"Untouched | Sharp={base_sharp:.1f}")]
        view_key = "full_scene" if view_idx == 0 else "native_1to1_crop"
        raw_metrics_report[view_key] = []

        for col_label, fg_act, bg_act, use_pt, f_sc, d_sc in raw_specs:
            pred_u8, mask_2d, _ = restore_fn(
                image_rgb_u8=base_u8,
                action=fg_act,
                bg_action=bg_act,
                point_mask=click_pt if use_pt else None,
                steps=steps,
                fidelity_scale=f_sc,
                detail_scale=d_sc,
            )
            if use_pt and mask_2d is not None and view_idx == 0:
                saved_raw_mask_preview = render_mask_overlay_preview(base_u8, mask_2d, point_xy=click_pt)

            mean_abs_px = float(np.mean(np.abs(pred_u8.astype(np.float32) - base_u8.astype(np.float32))))
            psnr_vs_raw = compute_psnr_u8(pred_u8, base_u8)
            sharp_ratio = compute_laplacian_sharpness(pred_u8) / base_sharp

            if fg_act == "none":
                max_err = float(np.max(np.abs(pred_u8.astype(np.float32) - base_u8.astype(np.float32))))
                sub = f"Max|d|={max_err:.1f}px (Locked)"
            elif use_pt and mask_2d is not None:
                fg_pct = 100.0 * float(np.mean(mask_2d >= 0.5))
                sub = f"|d|={mean_abs_px:.1f}px | FG={fg_pct:.0f}% | Shp {sharp_ratio:.2f}x"
            else:
                psnr_txt = "Inf" if psnr_vs_raw >= 99.0 else f"{psnr_vs_raw:.1f}dB"
                sub = f"|d|={mean_abs_px:.1f}px | {psnr_txt} | Shp {sharp_ratio:.2f}x"

            cells.append((col_label, pred_u8, sub))
            raw_metrics_report[view_key].append(
                {
                    "mode": col_label,
                    "action": fg_act,
                    "mean_abs_diff_u8": round(mean_abs_px, 3),
                    "psnr_vs_raw_db": round(psnr_vs_raw, 2),
                    "sharpness_ratio": round(sharp_ratio, 3),
                }
            )
        raw_rows_visual.append((view_title, cells))

    raw_sheet_u8 = _render_annotated_sheet(
        raw_rows_visual,
        sheet_title=f"Raw Image 'As-Is' Controllable Sweep — {image_stem} ({w_orig}x{h_orig})",
    )
    raw_sheet_file = out_path / f"{image_stem}_benchmark_raw_asis.png"
    cv2.imwrite(str(raw_sheet_file), cv2.cvtColor(raw_sheet_u8, cv2.COLOR_RGB2BGR))

    if saved_raw_mask_preview is not None:
        mask_preview_file = out_path / f"{image_stem}_benchmark_raw_mask.png"
        cv2.imwrite(str(mask_preview_file), cv2.cvtColor(saved_raw_mask_preview, cv2.COLOR_RGB2BGR))

    print(f"[2/3] Processing full-resolution raw image ({w_orig}x{h_orig}) with action='{user_action}'...")
    fullres_u8, _, _ = restore_fn(
        image_rgb_u8=image_rgb_u8,
        action=user_action,
        bg_action="none",
        point_mask=point_xy,
        steps=steps,
    )
    fullres_file = out_path / f"{image_stem}_raw_fullres_restored.png"
    cv2.imwrite(str(fullres_file), cv2.cvtColor(fullres_u8, cv2.COLOR_RGB2BGR))

    # =========================================================================
    # PART 2: CONTROLLED SYNTHETIC ARTIFACT & DIRECTIONAL CONTROL BENCHMARK
    # =========================================================================
    print("[3/3] Running 4-Scenario Synthetic Artifact, Directional Tone & Selective Control Benchmark...")
    pipeline = SelectiveDegradationPipeline(config)
    sh, sw = scene_rgb.shape[:2]

    split_mask = np.zeros((sh, sw), dtype=np.float32)
    split_mask[:, : sw // 2] = 1.0
    split_mask = cv2.GaussianBlur(split_mask, (0, 0), sigmaX=5.0, sigmaY=5.0)
    split_3d = split_mask[:, :, None]

    scenarios = [
        {
            "name": "Mild Everyday",
            "banner": "Scenario 1 [Mild Everyday]: Noise (0.38) + JPEG/Band (0.48) + Warm Cast (+0.55) | Selective = Keep Cast",
            "state": {"denoise": 0.38, "deartifact": 0.48, "warmth": 0.55},
            "focus_mode": 0,
            "artifact_mode": 0,
            "selective_spec": "denoise=1.0,deartifact=1.0,warmth=0.0",
            "selective_label": "4. Selective (Keep Cast)",
            "exact_spec": "denoise=1.0,deartifact=1.0,warmth=-0.55",
            "all_spec": "all,warmth=-0.55",
            "partial_spec": "denoise=0.5,deartifact=0.5,warmth=-0.275",
        },
        {
            "name": "Night Handheld",
            "banner": "Scenario 2 [Night Handheld]: Dark Exp (-0.68) + Heavy Noise (0.70) + Motion (0.70) | Selective = Keep Blur",
            "state": {"exposure": -0.68, "denoise": 0.70, "deblur_motion": 0.70},
            "focus_mode": 0,
            "artifact_mode": 0,
            "selective_spec": "denoise=1.0,exposure=0.68,deblur_motion=0.0",
            "selective_label": "4. Selective (Keep Blur)",
            "exact_spec": "denoise=1.0,exposure=0.68,deblur_motion=1.0",
            "all_spec": "all,exposure=0.68",
            "partial_spec": "denoise=0.5,exposure=0.34,deblur_motion=0.5",
        },
        {
            "name": "Washout + Bokeh",
            "banner": "Scenario 3 [Washout + Bokeh]: Bright Washout (+0.65) + Defocus (0.65) + Banding (0.55) | Selective = Keep Bokeh",
            "state": {"exposure": 0.65, "sharpen_focus": 0.65, "deartifact": 0.55},
            "focus_mode": 0,
            "artifact_mode": 1,
            "selective_spec": "exposure=-0.65,deartifact=1.0,sharpen_focus=0.0",
            "selective_label": "4. Selective (Keep Bokeh)",
            "exact_spec": "exposure=-0.65,deartifact=1.0,sharpen_focus=1.0",
            "all_spec": "all,exposure=-0.65",
            "partial_spec": "exposure=-0.325,deartifact=0.5,sharpen_focus=0.5",
        },
        {
            "name": "Web Compression",
            "banner": "Scenario 4 [Web Torture]: LowRes/SR (0.68) + Heavy JPEG (0.75) + Motion (0.45) + Cool (-0.45) | Selective = DeJPEG",
            "state": {"sharpen_focus": 0.68, "deartifact": 0.75, "deblur_motion": 0.45, "warmth": -0.45},
            "focus_mode": 1,
            "artifact_mode": 0,
            "selective_spec": "deartifact=1.0,sharpen_focus=0.0,deblur_motion=0.0,warmth=0.0",
            "selective_label": "4. Selective (DeJPEG)",
            "exact_spec": "sharpen_focus=1.0,deartifact=1.0,deblur_motion=1.0,warmth=0.45",
            "all_spec": "all,warmth=0.45",
            "partial_spec": "sharpen_focus=0.5,deartifact=0.5,deblur_motion=0.5,warmth=0.225",
        },
    ]

    synth_rows_visual = []
    synth_metrics_report: List[Dict] = []

    for s_idx, sc in enumerate(scenarios):
        rng = np.random.default_rng(seed + s_idx * 97)
        params = pipeline.sample_params(sh, sw, rng=rng)
        params.focus_mode = int(sc["focus_mode"])
        params.artifact_mode = int(sc["artifact_mode"])

        s_src = np.zeros(config.NUM_ACTIONS, dtype=np.float32)
        for deg_name, val in sc["state"].items():
            s_src[pipeline.type_to_idx[deg_name]] = float(val)

        deg_u8 = pipeline.render_with_params(scene_rgb, s_src, params)
        deg_psnr = compute_psnr_u8(deg_u8, scene_rgb)
        deg_ssim = compute_ssim_u8(deg_u8, scene_rgb)

        a_sel = parse_action_spec(sc["selective_spec"], config.DEGRADATION_TYPES, signed_types=signed_types)
        s_tgt_sel = pipeline.compute_target_state(s_src, a_sel)
        tgt_sel_u8 = pipeline.render_with_params(scene_rgb, s_tgt_sel, params)

        a_part = parse_action_spec(sc["partial_spec"], config.DEGRADATION_TYPES, signed_types=signed_types)
        s_tgt_part = pipeline.compute_target_state(s_src, a_part)
        tgt_part_u8 = pipeline.render_with_params(scene_rgb, s_tgt_part, params)

        tgt_split_u8 = np.clip(
            np.round(split_3d * scene_rgb.astype(np.float32) + (1.0 - split_3d) * deg_u8.astype(np.float32)),
            0.0,
            255.0,
        ).astype(np.uint8)

        cells = [
            ("1. Clean Reference", scene_rgb, "Ground Truth Reference"),
            ("2. Degraded Input", deg_u8, f"{deg_psnr:.2f} dB | SSIM {deg_ssim:.3f}"),
        ]

        eval_runs = [
            ("3. Identity (none)", "none", "none", None, 1.0, 0.0, deg_u8, "identity"),
            (sc["selective_label"], sc["selective_spec"], "none", None, 1.0, 0.0, tgt_sel_u8, "selective"),
            ("5. 50% Partial", sc["partial_spec"], "none", None, 1.0, 0.25, tgt_part_u8, "partial_50"),
            ("6. Targeted Exact", sc["exact_spec"], "none", None, 1.0, 0.0, scene_rgb, "targeted_exact"),
            ("7. Full (Pixel Only)", sc["all_spec"], "none", None, 1.0, 0.0, scene_rgb, "all_pixel"),
            ("8. Full (With Detail)", sc["all_spec"], "none", None, 1.0, 0.70, scene_rgb, "all_dual"),
            ("9. Split Mask (L/R)", sc["exact_spec"], "none", split_mask, 1.0, 0.45, tgt_split_u8, "split_mask"),
        ]

        row_m = {
            "scenario": sc["name"],
            "degraded_psnr": round(deg_psnr, 2),
            "degraded_ssim": round(deg_ssim, 4),
        }

        for col_lbl, fg_s, bg_s, m2d, f_sc, d_sc, ref_target_u8, m_key in eval_runs:
            pred_u8, _, _ = restore_fn(
                image_rgb_u8=deg_u8,
                action=fg_s,
                bg_action=bg_s,
                mask_2d=m2d,
                steps=steps,
                fidelity_scale=f_sc,
                detail_scale=d_sc,
            )
            if m_key == "identity":
                max_err = float(np.max(np.abs(pred_u8.astype(np.float32) - deg_u8.astype(np.float32))))
                sub = f"Max|d|={max_err:.1f}px (Exact)"
                row_m["identity_max_err_u8"] = round(max_err, 3)
            else:
                p_val = compute_psnr_u8(pred_u8, ref_target_u8)
                s_val = compute_ssim_u8(pred_u8, ref_target_u8)
                base_vs_tgt = compute_psnr_u8(deg_u8, ref_target_u8)
                gain = p_val - base_vs_tgt
                sign = "+" if gain >= 0 else ""
                sub = f"{p_val:.2f}dB ({sign}{gain:.1f}) | S {s_val:.3f}"
                row_m[f"{m_key}_psnr"] = round(p_val, 2)
                row_m[f"{m_key}_ssim"] = round(s_val, 4)

            cells.append((col_lbl, pred_u8, sub))

        synth_rows_visual.append((sc["banner"], cells))
        synth_metrics_report.append(row_m)

    synth_sheet_u8 = _render_annotated_sheet(
        synth_rows_visual,
        sheet_title=f"Synthetic Artifact & Directional Control Benchmark — {image_stem}",
    )
    synth_sheet_file = out_path / f"{image_stem}_benchmark_synthetic_sweep.png"
    cv2.imwrite(str(synth_sheet_file), cv2.cvtColor(synth_sheet_u8, cv2.COLOR_RGB2BGR))

    report = {
        "image": image_stem,
        "original_resolution": [w_orig, h_orig],
        "benchmark_panel_resolution": [int(scene_rgb.shape[1]), int(scene_rgb.shape[0])],
        "checkpoint_stage": ckpt_stage,
        "steps": steps,
        "outputs": {
            "raw_asis_sweep_png": str(raw_sheet_file),
            "synthetic_artifact_sweep_png": str(synth_sheet_file),
            "raw_fullres_restored_png": str(fullres_file),
        },
        "raw_asis_metrics": raw_metrics_report,
        "synthetic_benchmark_metrics": synth_metrics_report,
    }
    json_file = out_path / f"{image_stem}_benchmark_metrics.json"
    json_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n── Raw Image 'As-Is' Sweep Summary (Full Scene) ──")
    print(f"  {'Mode':<24} | {'Mean |d| (0-255)':>16} | {'PSNR vs Raw':>12} | {'Sharpness':>10}")
    print("  " + "-" * 70)
    for item in raw_metrics_report["full_scene"]:
        p_str = "Inf dB" if item["psnr_vs_raw_db"] >= 99.0 else f"{item['psnr_vs_raw_db']:6.2f} dB"
        print(
            f"  {item['mode']:<24} | {item['mean_abs_diff_u8']:13.2f} px | {p_str:>12} | {item['sharpness_ratio']:9.2f}x"
        )

    print("\n── Controlled Synthetic Artifact Benchmark (PSNR dB vs Exact Target) ──")
    print(
        f"  {'Scenario':<16} | {'Degraded':>8} | {'Selective':>9} | {'50% Part':>8} | "
        f"{'Targeted':>8} | {'all(Pix)':>8} | {'all(Det)':>9} | {'SplitMsk':>8}"
    )
    print("  " + "-" * 92)
    for r in synth_metrics_report:
        print(
            f"  {r['scenario']:<16} | {r['degraded_psnr']:8.2f} | {r['selective_psnr']:9.2f} | "
            f"{r['partial_50_psnr']:8.2f} | {r['targeted_exact_psnr']:8.2f} | "
            f"{r['all_pixel_psnr']:8.2f} | {r['all_dual_psnr']:9.2f} | {r['split_mask_psnr']:8.2f}"
        )

    print(f"\n✓ Saved Raw As-Is Sweep       -> {raw_sheet_file}")
    print(f"✓ Saved Synthetic Sweep       -> {synth_sheet_file}")
    print(f"✓ Saved Full-Res Raw Restore  -> {fullres_file}")
    print(f"✓ Saved Benchmark JSON Report -> {json_file}")
    return report
