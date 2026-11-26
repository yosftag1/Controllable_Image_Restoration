# Controllable Pretrained Image Restoration (V7 — SD-Turbo Crisp 7-Control)

An **orthogonal, spatially controllable image restoration & super-resolution framework** built on **Frozen DINOv2-with-Registers** (`facebook/dinov2-with-registers-base`) and **1-Step ADD-Distilled SD-Turbo** (`stabilityai/sd-turbo`, `t = 499`), with **Routed Specialist LoRA-MoE** and **Stage-2 Detail-Gated VAE Decoder LoRA**.

All control resides in a **7-Channel Spatial Control Overlay** $\mathbf{A}(y, x)$, a **Crisp Detail** dial (`detail_scale` $\in [0, 1]$), an optional **2× / 4× Super-Resolution Upscale** (`--upscale 2|4`), and optional full-frame lens distortion correction (`--lens-k1`):

### Control Knobs (Streamlined 7 Core Controls)

| Knob | Range | LoRA Expert | Stage-1 Budget | Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **`denoise`** | `[0.0, 1.0]` | `clean_atmos` | **21%** | `0.0` = keep grain · `1.0` = remove white, demosaiced ISP & post-noise JPEG grain |
| **`deartifact`** | `[0.0, 1.0]` | `clean_atmos` | 10% | `0.0` = keep · `1.0` = remove JPEG blocks, ringing, 4:2:0 subsampling & banding |
| **`deblur_motion`** | `[0.0, 1.0]` | `motion_deconv` | **26%** | `0.0` = keep panning streaks · `1.0` = deconvolve real & curved/noisy camera shake |
| **`super_resolution`** | `[0.0, 1.0]` | `sr_detail` | **25%** | `0.0` = keep original softness · `1.0` = Real-ESRGAN blind SR & crisp detail enhancement |
| **`dehaze`** | `[0.0, 1.0]` | `clean_atmos` | 7% | `0.0` = keep atmospheric mist · `1.0` = clear haze & restore contrast |
| **`exposure`** | `[-1.0, +1.0]` | `1×1` Radiometric | 6% | `-1.0` = darken / recover washout · `0.0` = keep mood lighting · `+1.0` = brighten shadows |
| **`warmth`** | `[-1.0, +1.0]` | `1×1` Radiometric | 5% | `-1.0` = cool / remove warm yellow cast · `0.0` = keep · `+1.0` = warm / remove cold blue cast |
| **`detail_scale`** | `[0.0, 1.0]` | `sem_branch` (UNet + VAE Dec) | Stage 2 (`d=1.0`) | `0.0` = Stage-1 faithful crisp · `1.0` = full Stage-2 photographic micro-detail (0% overshoot) |

---

## Core Architecture (V7)

1. **1-Step ADD-Distilled `SD-Turbo` (`t = 499`) + Routed Specialist LoRA-MoE**:
   - Uses the exact single-step `DDPMScheduler` `pred_original_sample` formulation (`z_in = z_src` at unit variance, $\hat{z}_0 = (z_{\text{src}} - \sqrt{1-\bar{\alpha}_t}\hat{\epsilon})/\sqrt{\bar{\alpha}_t}$ at SD-Turbo's native ADD midpoint `t = 499`).
   - Shared Base LoRA (`lin=16, conv=12`) + 3 additive slider-routed specialists (`motion_deconv r=12`, `sr_detail r=12`, `clean_atmos r=8`) eliminate gradient conflicts between denoising and high-frequency deconvolution/SR.
2. **Destructive-Skip Shutdown + Detail-Gated VAE Decoder LoRA**:
   - Disables blind unconditioned VAE encoder LoRA (`VAE_ENC_LORA_RANK = 0`).
   - Shuts down blurry/low-res/hazy encoder skips (`90–98%` off) for `deblur_motion`, `super_resolution`, and `dehaze` in **both** Stage 1 and Stage 2 so input blur never bleeds through.
   - Adds a Stage-2 `sem_branch`-only VAE Decoder LoRA (`r = 8`, zero at `detail_scale = 0.0` and zero on `denoise`/tone) for crisp 1–2px micro-edges.
3. **Phase-Aligned Losses + Zero-VRAM Frozen DINOv2 Feature Discriminator (S3Diff / ADD Style)**:
   - **Stage 1 (`pixel_stage`):** Charbonnier L1 + Latent L1 (`0.25`) + **un-blurred** VGG LPIPS (`0.35`) + **phase-aligned** Multi-Scale Sobel/Laplacian Edge Gradient L1 (`0.25`).
   - **Stage 2 (`semantic_stage`, fixed `detail_scale = 1.0`):** Full+2×LP L1 + **True Covariance-Aware DISTS** (`0.80`) + Multi-Scale Edge Gradient L1 (`0.35`) + **Frozen DINOv2 Multi-Layer Feature Discriminator** (`0.15`), strictly masked to 100% fully restored (`s_tgt == 0`), crisp high-res clean photography ONLY.
4. **Artifact-First, Dataset-Balanced Sampling & Single-to-Compound Curriculum**:
   - Samples by artifact budget first (Big 3 = 72%), then balances across datasets inside each artifact (`DIV2K` 45% / `Flickr2K` 35% / `OST` 20% for clean pool; per-subdataset balanced GoPro/HIDE/RealBlur + synthetic DIV2K for motion blur; `REUSE_PAIRED_CLEAN_FOR_SYNTH = False`).
   - Starts ~84% single-artifact -> ~62% single / 38% compound, boosts `denoise + deblur_motion` with demosaiced & noise-before-JPEG ISP noise, and explicitly trains the `"all"` preset (`16%` on both synthetic and real pairs).

---

## Quick Start

### 1. Install & Run Smoke Tests
```bash
pip install -r requirements.txt
python tests/smoke_test.py
```

### 2. Kaggle 2× T4 Training (Configurable Resolution `--patch-size` & `--grad-accum`)

**Default 384×384 Balanced Pilot Run (`--pilot` — 8 pixel + 4 semantic epochs @ 700 opt steps, eff batch 8):**
```python
%run train.py --pilot \
  --patch-size 384 --batch-size 4 --grad-accum 2 \
  --train-root /kaggle/input/datasets/thaihoa1476050/df2k-ost/train \
  --paired-motion-root /kaggle/input/datasets/jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets \
  --val-root /kaggle/input/datasets/thaihoa1476050/df2k-ost/test/DIV2K_valid
```

**Full Native 512×512 Run (10 pixel + 5 semantic epochs, 2/GPU × 2 grad accum = eff batch 8):**
```python
%run train.py \
  --patch-size 512 --batch-size 4 --grad-accum 2 \
  --train-root /kaggle/input/datasets/thaihoa1476050/df2k-ost/train \
  --paired-motion-root /kaggle/input/datasets/jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets \
  --val-root /kaggle/input/datasets/thaihoa1476050/df2k-ost/test/DIV2K_valid
```

### 3. Controllable Inference, 2×/4× Super-Resolution & Gradio Studio
```bash
# 2x Super-Resolution + DeJPEG with crisp Stage-2 detail:
python inference.py \
    --model checkpoints/model_best.pt \
    --image lowres_photo.jpg \
    --action "super_resolution=1.0,deartifact=0.6" \
    --upscale 2 \
    --detail-scale 0.80

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
