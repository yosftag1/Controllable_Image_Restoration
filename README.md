# Controllable Pretrained Image Restoration

An **action-conditioned, spatially controllable image restoration framework** that repurposes pretrained **Semantic Vision Foundation Models** (`NVIDIA C-RADIOv2.5-B` or `DINOv3 + SigLIP 2`) and **Fast Pretrained Diffusion/Flow Generators** (`stabilityai/sd-turbo` or `segmind/tiny-sd`)—without forcing images back to a single blind "base state."

Instead of wiping out intentional motion blur, bokeh, film grain, or mood lighting, you pass a **Spatial Action Overlay** $\mathbf{A} \in [0, 1]^{8 \times H \times W}$ (via global sliders, custom masks, or **1-click semantic object selection**) plus **Dual-LoRA Fidelity/Detail dials** ($\lambda_{\text{pix}}, \lambda_{\text{sem}}$) to control *exactly what* gets fixed, *where*, and *how conservatively vs. generatively*.

---

## Core Architecture

1. **Semantic "Eyes" (`SemanticEncoder` + Self-Teacher LoRA)**:
   - Supports **`nvidia/C-RADIOv2.5-B`** (~90M unified distillation of `DINO` + `SigLIP` + `SAM`, default) or **`DINOv3-Small` + `SigLIP 2-Base`**.
   - **Zero-Extra-VRAM Self-Teacher Alignment**: Frozen base weights act as the clean teacher (`LoRA OFF` on clean $x_0$) while lightweight `SwitchableLoRALinear` detours (`LoRA ON` on degraded $x_{\text{src}}$) learn to extract clean semantic tokens through heavy compound noise, dark gamma, and blur.
2. **Generative "Hands" (`SkipConnectedVAE` + Dual-LoRA `UNet2DConditionModel`)**:
   - Supports **`stabilityai/sd-turbo`** (peak 1-step generative quality) and **`segmind/tiny-sd` / `nota-ai/bk-sdm-tiny`** (~323M ultra-lightweight UNet), dropping the 123M–350M CLIP text encoder completely.
   - **Action-Gated Skip-Connected VAE**: Zero-initialized multi-scale Encoder $\to$ Decoder bridges modulated pointwise (`1×1` Conv) by the Spatial Action Overlay let untouched pixels and intentional effects bypass the `8×` latent bottleneck with zero blur.
   - **Action-Modulated Dual-LoRA (`PiSA-SR` + `S3Diff` style)**:
     - **Pixel-Faithful LoRA ($\lambda_{\text{pix}}$)**: Trained strictly on structural/frequency losses for artifact-free accuracy.
     - **Semantic-Detail LoRA ($\lambda_{\text{sem}}$)**: Trained with frozen Pixel-LoRA on multi-layer perceptual + `C-RADIO`/`DINOv3` semantic consistency losses for crisp texture synthesis on heavy blur and downsampling.
3. **Pixel Overlay Sheet (`PointwiseSpatialModulator` + Zero-Action Lock)**:
   - Uses strictly `1×1` spatial modulation (`kernel_size=1`, zero spatial receptive field across neighbors) so **whole-image sliders**, **single-object masks**, and **multi-region edits** behave identically at every pixel, backed by a **Zero-Action Lock** guaranteeing exact bit-for-bit preservation wherever all sliders equal `0.0`.
4. **Seed-Synchronized Selective Degradation Synthesis (`SelectiveDegradationPipeline`)**:
   - Samples all physical random variables (`DegradationParams`) upfront with continuous kernel interpolation, non-commutativity guards for radiometric + compression pairs, and **false-positive slider training** so pushing sliders up on already-clean attributes never causes ringing or over-smoothing.

---

## Project Layout

```text
Controllable_Image_Restoration/
├── config.py                        # Hyperparameters, backbone switches, 3-phase schedule
├── train.py                         # 3-Phase Multi-GPU FP16 trainer + controllable sweep grids
├── inference.py                     # CLI for global sliders, Dual-LoRA dials & 1-click object masks
├── requirements.txt
├── data/
│   └── selective_dataset.py         # Seed-synchronized partial & spatially localized synthesis
├── models/
│   ├── lora_layers.py               # Switchable Self-Teacher LoRA & Action-Modulated Dual-LoRA
│   ├── semantic_encoder.py          # C-RADIOv2.5-B & DINOv3+SigLIP2 wrapper
│   ├── action_overlay.py            # Action tokens + 1x1 PointwiseSpatialModulator (SPADE)
│   ├── skip_vae.py                  # DataParallel-safe SkipConnectedVAE
│   └── controllable_restorer.py     # End-to-end 1-to-4 step Latent Rectified Flow restorer
├── losses/
│   └── controllable_losses.py       # Charbonnier + Gradient + 2D FFT + Multi-Layer VGG + Semantic Consistency
├── utils/
│   └── semantic_mask.py             # Action string parser & 1-click semantic object mask extractor
└── tests/
    └── smoke_test.py                # Syntax, physics, Dual-LoRA isolation & overlay verification
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python tests/smoke_test.py
```

### 2. 3-Phase Training (Supports Single-GPU & Multi-GPU e.g. Kaggle 2× T4)
Checkpoints only store the lightweight trainable LoRA / Skip-VAE / Overlay weights (**~45 MB** instead of 4 GB) as `model_latest.pt` and `model_best.pt`:

```bash
# Kaggle DF2K-OST notebook workflow (safe on both fresh Epoch 1 runs & resumed sessions)
%run train.py \
  --train-root /kaggle/input/datasets/thaihoa1476050/df2k-ost/train \
  --val-root /kaggle/input/datasets/thaihoa1476050/df2k-ost/test/DIV2K_valid \
  --resume ./checkpoints/model_latest.pt

# Default local DIV2K: C-RADIOv2.5-B + sd-turbo (3-phase staged training)
python train.py --data-root ./data/DIV2K

# Ultra-lightweight generator + DINOv3+SigLIP2 vision encoders
python train.py --data-root ./data/DIV2K --backbone tiny-sd --vision-encoder dinov3_siglip2

# Fast debug check (3 epochs: 1 Warmup + 1 Pixel Stage + 1 Semantic Stage)
python train.py --debug --data-root ./data/DIV2K
```

### 3. Controllable Inference Examples

```bash
# A) Selective Global Control: Remove 100% noise & brighten exposure 70%, KEEP motion blur untouched!
python inference.py \
    --model checkpoints/model_best.pt \
    --image sample.jpg \
    --action "noise=1.0,exposure=0.7,motion=0.0" \
    --detail-scale 0.75

# B) 1-Click Localized Object Control: Click subject at (x=0.5, y=0.45) to deblur & denoise subject,
#    while keeping background panning motion blur intact (only denoising background)!
python inference.py \
    --model checkpoints/model_best.pt \
    --image panning_car.jpg \
    --point-mask "0.5,0.45" \
    --action "noise=1.0,motion=1.0" \
    --bg-action "noise=1.0,motion=0.0" \
    --detail-scale 1.0

# C) Conservative Pixel-Exact Mode vs. High-Detail Mode (PiSA-SR Dual-LoRA dial, zero retraining)
python inference.py --model checkpoints/model_best.pt --image blurry.jpg --action "all" --detail-scale 0.0
python inference.py --model checkpoints/model_best.pt --image blurry.jpg --action "all" --detail-scale 1.2 --steps 2
```
