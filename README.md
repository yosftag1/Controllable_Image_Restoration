# Controllable Pretrained Image Restoration

An **orthogonal, spatially controllable image restoration framework** built on **Frozen DINOv2-with-Registers** (`facebook/dinov2-with-registers-base`) and **Stable Diffusion 2.1 Base** (`sd2-community/stable-diffusion-2-1-base` default, plus `sd-turbo`, `sd-1.5`, `bk-sdm-small`, and `tiny-sd`).

All control resides in a **10-Channel Spatial Control Overlay** $\mathbf{A}(y, x)$ plus a **Generative Texture Detail** dial (`detail_scale` $\in [0, 1]$) and an optional **Full-Frame Lens Distortion** parameter (`--lens-k1`):

### Control Knobs

| Knob | Range | Role | Meaning |
| :--- | :---: | :---: | :--- |
| **`denoise`** | `[0.0, 1.0]` | `clean_smooth` | `0.0` = keep grain · `1.0` = full sensor luminance & chroma denoise |
| **`deartifact`** | `[0.0, 1.0]` | `clean_smooth` | `0.0` = keep · `1.0` = remove JPEG blocks, ringing, 4:2:0 subsampling & banding |
| **`deblur_motion`** | `[0.0, 1.0]` | `motion_deconv` | `0.0` = keep panning streaks · `1.0` = deconvolve Bezier/tremor camera shake |
| **`sharpen_focus`** | `[0.0, 1.0]` | `focus_optical` | `0.0` = keep portrait bokeh · `1.0` = sharpen optical defocus & low-res softness |
| **`deflare`** | `[0.0, 1.0]` | `optics_atmos` | `0.0` = keep sunlight flare · `1.0` = remove FFT corona, streaks, wash & lens ghosts |
| **`dereflection`** | `[0.0, 1.0]` | `optics_atmos` | `0.0` = keep glass reflections · `1.0` = remove double-pane window reflections |
| **`dechromatic`** | `[0.0, 1.0]` | `focus_optical` | `0.0` = keep · `1.0` = remove radial purple/cyan lens chromatic aberration |
| **`dehaze`** | `[0.0, 1.0]` | `optics_atmos` | `0.0` = keep atmospheric mist · `1.0` = clear haze & restore contrast |
| **`exposure`** | `[-1.0, +1.0]` | `1×1` Radiometric Head | `-1.0` = darken / recover washout · `0.0` = keep mood lighting · `+1.0` = brighten shadows |
| **`warmth`** | `[-1.0, +1.0]` | `1×1` Radiometric Head | `-1.0` = cool / remove warm yellow cast · `0.0` = keep · `+1.0` = warm / remove cold blue cast |
| **`detail_scale`** | `[0.0, 1.0]` | `sem_branch` (`r=16`, Stage 2) | `0.0` = conservative pixel-faithful PSNR · `0.65–1.0` = crisp natural micro-texture |
| **`--lens-k1`** | `[-0.35, +0.35]` | Full-frame Brown-Conrady | Optional standalone radial barrel/pincushion correction before tiled inference |

---

## Core Architecture

1. **2-Step Latent Refinement on `SD-2.1-Base` (`t = 299 → 99`) + Shared/Specialist LoRA-MoE**:
   - Encodes once, runs Step 1 (`t=299`) for coarse deblur/denoise and Step 2 (`t=99`) for crisp fine-edge polish, then decodes once.
   - Uses an always-active Shared Base Pixel LoRA (`r=20`) + additive Slider-Routed Specialist Experts (`clean_smooth`, `motion_deconv`, `focus_optical`, `r=8` each) + Stage-2 `sem_branch` (`r=16`) with zero-drift Target-Anchored CSD and a gentle full-res subpixel anchor.
2. **2-Path Spatial Conditioning**:
   - `_ActionConditionedConvIn` + strictly `1×1` SPADE `PointwiseSpatialModulator` across UNet blocks, paired with `ActionOverlayEncoder` cross-attention tokens anchored onto OpenCLIP's 77-token prompt manifold.
3. **`SkipConnectedVAE` + `1×1` `PointwiseRadiometricHead`**:
   - Frozen natural-manifold VAE decoder, lightweight `r=8` VAE encoder LoRA, action-gated `3×3` skip bridges that fade fine skips smoothly as `detail_scale` increases, and exact zero-action bit-for-bit identity lock.
4. **Rich Hybrid Clean + Real Camera Deblurring Loader (`GoPro` + `HIDE` + `RealBlur` + `DPDD`)**:
   - Combines Laplacian sharpness-gated clean 2K crops (Bezier arcs, S-curve tremor, micro-shake motion blur, bokeh discs + chromatic aberration, 4:2:0 chroma/ringing/JPEG/banding, heteroscedastic noise, up to 4 compound degradations) with real paired camera datasets wrapped via a **Compositional Real-Pair Wrapper**.

---

## Quick Start

### 1. Install & Run Smoke Tests
```bash
pip install -r requirements.txt
python tests/smoke_test.py
```

### 2. Kaggle 2× T4 Training (Clean 2K + Curated Real Deblurring Pack)

**Balanced ~5-Hour 2-Stage Run (`--pilot` / `--5h` — `9` pixel + `3` detail/CSD epochs @ `950` steps/ep):**
```python
%run train.py --pilot \
  --train-root /kaggle/input/datasets/thaihoa1476050/df2k-ost/train \
  --paired-motion-root /kaggle/input/datasets/jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets \
  --val-root /kaggle/input/datasets/thaihoa1476050/df2k-ost/test/DIV2K_valid
```

**Full Uncapped Run (12 pixel + 3 detail/CSD epochs, or pass `--csd-weight 0.0` to ablate Stage 2 without CSD):**
```python
%run train.py \
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
