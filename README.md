# Controllable Pretrained Image Restoration (V4)

An **orthogonal, spatially controllable & directionally graded image restoration framework** built on **Semantic Vision Foundation Models** (`nvidia/C-RADIOv3-B` default, `nvidia/C-RADIOv2-B`, or `DINOv3 / DINOv2-Registers + SigLIP 2`) and **Fast Pretrained Diffusion Generators** (`stabilityai/sd-turbo` or `segmind/tiny-sd`).

Instead of forcing every photo to an ambiguous "auto-fixed" state or wiping out intentional motion blur, portrait bokeh, film grain, or mood lighting, you pass a **6-Channel Spatial Control Overlay** $\mathbf{A}(y, x)$ plus a single **Generative Texture Detail** dial ($\lambda_{\text{det}} \in [0, 1]$):

### Control Knobs

| Knob | Range | Meaning |
| :--- | :---: | :--- |
| **`denoise`** | `[0.0, 1.0]` | `0.0` = keep grain · `1.0` = full sensor luminance & chroma denoise |
| **`deartifact`** | `[0.0, 1.0]` | `0.0` = keep · `1.0` = remove JPEG 8×8 blocking, ringing & 8-bit sky banding |
| **`deblur_motion`** | `[0.0, 1.0]` | `0.0` = keep panning/motion streaks · `1.0` = deconvolve camera shake |
| **`sharpen_focus`** | `[0.0, 1.0]` | `0.0` = keep portrait bokeh · `1.0` = sharpen optical defocus & low-res softness |
| **`exposure`** | `[-1.0, +1.0]` | `-1.0` = darken / recover washout · `0.0` = keep mood lighting · `+1.0` = brighten shadows |
| **`warmth`** | `[-1.0, +1.0]` | `-1.0` = cool / remove warm yellow cast · `0.0` = keep · `+1.0` = warm / remove cold blue cast |
| **`detail_scale`** | `[0.0, 1.0]` | `0.0` = conservative pixel-faithful PSNR · `0.65–1.0` = crisp generative texture detail |

---

## Core V4 Architecture

1. **Calibrated Pre-Scaled `SD-Turbo` Step (`t = 249`)**:
   - Pre-scales encoded latents (`z_in = sqrt_alpha_bar * z_src`) onto `sd-turbo`'s exact native ADD amplitude manifold and predicts $\hat{z}_0 = z_{\text{src}} - \sigma_t \hat{\epsilon}_\theta$, preserving exact `1.0x` unit identity at Step 0 instead of `1.90x` unscaled gain blowup.
2. **Dense 2D `C-RADIO` Spatial Grid + Full Encoder-to-Decoder Spatial Modulation**:
   - Injects spatially aligned 2D `C-RADIOv3-B` (`FeatSharp`) feature grids directly into UNet bottleneck and mid-resolution decoder blocks while modulating `conv_in`, all `down_blocks`, `mid_block`, and `up_blocks` pointwise (`1×1` SPADE) with the 6-channel spatial control overlay.
3. **Blur-Aware `SkipConnectedVAE` + Native `H × W` `PixelSpaceRefiner`**:
   - Blur-aware skip bridges retain sharp `H/4` & `H/8` object boundaries during denoising/tone edits while suppressing blurred encoder features during deconvolution.
   - `PixelSpaceRefiner` combines a `1×1` rational-curve radiometric head (>40 dB directional `exposure` & `warmth` accuracy without `8x` VAE loss) with action-modulated depthwise-separable `3×3` edge blocks and a **Signed-Safe Zero-Action Lock** (`max(|a_k|) == 0`).
4. **Calibrated Multi-Scale Phase + Edge + Preservation Objective**:
   - Balances Charbonnier RGB L1, a 3-scale H/V/diagonal edge gradient pyramid, phase-aligned complex `rfft2` + log-amplitude frequency loss, multi-scale (`1×1` mean/std, `4×4`, `16×16`) opponent chroma/tone loss, bounded unit-normalized LPIPS, Vision-Aided PatchGAN, and an explicit **zero-DC tone lock** whenever `exposure = warmth = 0`.

---

## Quick Start

### 1. Install & Run Smoke Tests
```bash
pip install -r requirements.txt
python tests/smoke_test.py
```

### 2. Kaggle 2× T4 Training (~28k steps on DF2K-OST)
> **Note:** Start V4 training fresh (without resuming legacy 8-slider V1–V3 checkpoints).

```python
!pip install -q open-clip-torch

%run train.py \
  --train-root /kaggle/input/datasets/thaihoa1476050/df2k-ost/train \
  --val-root /kaggle/input/datasets/thaihoa1476050/df2k-ost/test/DIV2K_valid
```

To resume an interrupted **V4** session seamlessly:
```python
%run train.py \
  --train-root /kaggle/input/datasets/thaihoa1476050/df2k-ost/train \
  --val-root /kaggle/input/datasets/thaihoa1476050/df2k-ost/test/DIV2K_valid \
  --resume ./checkpoints/model_latest.pt
```

### 3. Controllable Inference Examples
```bash
# A) Night Handheld Shot: Denoise + brighten shadows (+0.65), KEEP motion blur untouched!
python inference.py \
    --model checkpoints/model_best.pt \
    --image night_panning.jpg \
    --action "denoise=1.0,exposure=0.65,motion=0.0" \
    --detail-scale 0.0

# B) Washed-Out Portrait: Darken highlights (-0.60), cool warm cast (-0.40), KEEP portrait bokeh!
python inference.py \
    --model checkpoints/model_best.pt \
    --image portrait.jpg \
    --action "denoise=0.5,deartifact=1.0,exposure=-0.60,warmth=-0.40,focus=0.0"

# C) 1-Click Localized Object Restoration: Deblur & denoise clicked subject, only denoise background!
python inference.py \
    --model checkpoints/model_best.pt \
    --image panning_car.jpg \
    --point-mask "0.5,0.45" \
    --action "denoise=1.0,motion=1.0" \
    --bg-action "denoise=1.0,motion=0.0" \
    --detail-scale 0.70
```

### 4. Interactive Gradio Studio
```python
!pip install -q gradio
%run app.py --model ./checkpoints/model_best.pt
```
