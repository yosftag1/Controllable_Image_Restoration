# Controllable Pretrained Image Restoration (V5)

An **orthogonal, spatially controllable & expandable universal image restoration framework** built on **Semantic Vision Foundation Models** (`nvidia/C-RADIOv3-B` default, `nvidia/C-RADIOv2-B`, or `DINOv3 / DINOv2-Registers + SigLIP 2`) and **Pretrained Diffusion Score Priors** (`stabilityai/stable-diffusion-2-1-base` default, plus `sd-turbo`, `sd-1.5`, `bk-sdm-small`, and `tiny-sd`).

Instead of forcing every photo into an uncontrolled blind restoration or training task-specific blur estimators, V5 keeps **100% of control inside your 6-Channel Spatial Control Overlay** $\mathbf{A}(y, x)$ plus a single **Generative Texture Detail** dial ($\lambda_{\text{det}} \in [0, 1]$), and makes adding future knobs (e.g., lens distortion, lens flare) plug-and-play:

### Control Knobs

| Knob | Range | Specialist LoRA Expert | Meaning |
| :--- | :---: | :---: | :--- |
| **`denoise`** | `[0.0, 1.0]` | `clean_smooth` (`r=24`) | `0.0` = keep grain · `1.0` = full sensor luminance & chroma denoise |
| **`deartifact`** | `[0.0, 1.0]` | `clean_smooth` (`r=24`) | `0.0` = keep · `1.0` = remove JPEG 8×8 blocking, ringing & 8-bit sky banding |
| **`deblur_motion`** | `[0.0, 1.0]` | `motion_deconv` (`r=32`, `1×7+7×1` strips) | `0.0` = keep panning/motion streaks · `1.0` = deconvolve camera shake |
| **`sharpen_focus`** | `[0.0, 1.0]` | `focus_optical` (`r=32`, `3×3+5×5` discs) | `0.0` = keep portrait bokeh · `1.0` = sharpen optical defocus & low-res softness |
| **`exposure`** | `[-1.0, +1.0]` | `1×1` Radiometric Head | `-1.0` = darken / recover washout · `0.0` = keep mood lighting · `+1.0` = brighten shadows |
| **`warmth`** | `[-1.0, +1.0]` | `1×1` Radiometric Head | `-1.0` = cool / remove warm yellow cast · `0.0` = keep · `+1.0` = warm / remove cold blue cast |
| **`detail_scale`** | `[0.0, 1.0]` | `sem_branch` (`r=16`, Stage 3 `CSD`) | `0.0` = conservative pixel-faithful PSNR · `0.65–1.0` = crisp natural micro-texture |

---

## Core V5 Upgrades

1. **`SD-2.1-Base` Default + Zero-Extra-VRAM Target-Anchored `CSD` (`PiSA-SR` + `TSD-SR`)**:
   - Replaces `sd-turbo` (which had baked-in ADD oversharpening and zero negative-prompt guidance) with `stabilityai/stable-diffusion-2-1-base` (`--backbone sd-turbo | sd-1.5 | bk-sdm-small | tiny-sd` remain selectable).
   - Stage 3 temporarily disables LoRAs (`lambda_pix=0, lambda_sem=0`) on the already-loaded base UNet under `torch.no_grad()` (`0 MB` extra model weights!) to compute a target-anchored negative-prompt score direction pushing away from `"wavy fake texture, ringing, oversharpened halos"`.
2. **Expandable Slider-Routed `UIR-LoRA` Task Experts + Strip/Multi-Scale Conv-LoRA + `DoRA`**:
   - Eliminates cross-task gradient conflict between low-pass smoothing (`denoise`) and high-pass deconvolution (`deblur_motion`, `sharpen_focus`) via named experts in `Config.LORA_EXPERT_SPECS` routed directly by the user's spatial sliders (`zero blind estimator`).
   - Adding new knobs later (e.g. `lens_distortion`, `deflare`) only requires appending to `DEGRADATION_TYPES` and `LORA_EXPERT_SPECS` and passing `--allow-schema-expand` to warm-start existing trained experts.
3. **Universal `OSDD`-Style `eVAE` Spatial Skip Alignment + Slider-Driven Adaptive Timestep**:
   - `_UniversalSpatialSkipBridge` (`3×3` + `dilated 3×3` + `1×5/5×1` strip depthwise convs modulated by `action_map`) + multi-scale clean encoder skip supervision (`LOSS_EVAE_SKIP_WEIGHT`) preserves crisp high-res geometry across the frozen natural-manifold VAE decoder without leaking blur streaks.
   - Slider-driven adaptive timestep maps active structural control strength smoothly into `t ∈ [49, 279]`.
4. **Hybrid Clean + Real Camera Deblurring Loader (`GoPro` + `HIDE` + `RealBlur` + `DPDD`)**:
   - Combines Laplacian sharpness-gated clean 2K crops (2D Bezier curved camera shake + optical bokeh disc PSFs) with real paired camera datasets wrapped via a **Compositional Real-Pair Wrapper** so all 6 selective sliders train on authentic camera blur.
5. **Isolated Experimental `HYPIR` Mode (`--hypir-mode`)**:
   - Kept strictly separate from the main V5 pipeline so `HYPIR` (deterministic fixed-step `2×` high-rank score prior + Relativistic `C-RADIO` `RpGAN`) can be tested independently.

---

## Quick Start

### 1. Install & Run Smoke Tests
```bash
pip install -r requirements.txt
python tests/smoke_test.py
```

### 2. Kaggle 2× T4 Training (Clean 2K + Curated Real Deblurring Pack)
```python
!pip install -q open-clip-torch

%run train.py \
  --train-root /kaggle/input/datasets/thaihoa1476050/df2k-ost/train \
  --paired-motion-root /kaggle/input/a-curated-list-of-image-deblurring-datasets \
  --val-root /kaggle/input/datasets/thaihoa1476050/df2k-ost/test/DIV2K_valid
```

To test the **separate experimental `HYPIR` mode** on its own:
```python
%run train.py \
  --hypir-mode \
  --train-root /kaggle/input/datasets/thaihoa1476050/df2k-ost/train \
  --paired-motion-root /kaggle/input/a-curated-list-of-image-deblurring-datasets \
  --val-root /kaggle/input/datasets/thaihoa1476050/df2k-ost/test/DIV2K_valid
```

### 3. Controllable Inference & Gradio Studio
```bash
# Night Handheld Shot: Denoise + brighten shadows (+0.65), KEEP motion blur untouched!
python inference.py \
    --model checkpoints/model_best.pt \
    --image night_panning.jpg \
    --action "denoise=1.0,exposure=0.65,motion=0.0" \
    --detail-scale 0.0
```
```python
!pip install -q gradio
%run app.py --model ./checkpoints/model_best.pt
```
