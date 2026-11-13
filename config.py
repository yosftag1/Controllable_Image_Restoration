"""Configuration for Controllable Pretrained Image Restoration."""


class Config:
    """Main configuration for Action-Conditioned Dual-LoRA + Skip-VAE + Semantic Prior Restoration."""

    PROJECT_NAME = "controllable_image_restoration"

    # -------------------------------------------------------------------------
    # 1. Pretrained Semantic Vision Encoder ("The Eyes" + Self-Teacher LoRA)
    # -------------------------------------------------------------------------
    # Options: "c_radio" (default), "dinov3_siglip2", or "mock_tiny" (offline unit tests)
    SEMANTIC_ENCODER_TYPE = "c_radio"

    # Hugging Face repository IDs (100% public & ungated defaults)
    # Default: newest "nvidia/C-RADIOv3-B" (90M FeatSharp dense distillation; fallback: "nvidia/C-RADIOv2-B")
    RADIO_MODEL_ID = "nvidia/C-RADIOv3-B"
    RADIO_ALT_MODEL_ID = "nvidia/C-RADIOv2-B"
    DINOV3_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
    DINOV2_FALLBACK_MODEL_ID = "facebook/dinov2-with-registers-small"
    SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"

    # Self-Teacher LoRA settings on frozen semantic vision encoder(s)
    VISION_LORA_RANK = 8
    VISION_LORA_ALPHA = 16.0
    VISION_LORA_DROPOUT = 0.0

    # -------------------------------------------------------------------------
    # 2. Pretrained Diffusion/Flow Generator ("The Hands" + Skip-VAE + Dual-LoRA)
    # -------------------------------------------------------------------------
    # Options:
    #   - "sd-turbo": stabilityai/sd-turbo (1-step native, ~860M frozen UNet, peak quality)
    #   - "tiny-sd":  segmind/tiny-sd (~323M pruned UNet, ultra-lightweight & fast)
    #   - "bk-sdm-tiny": nota-ai/bk-sdm-tiny (~323M block-pruned UNet)
    #   - "mock_tiny": tiny randomly initialized UNet+VAE for fast offline testing
    GENERATOR_BACKBONE = "sd-turbo"

    GENERATOR_REPO_MAP = {
        "sd-turbo": "stabilityai/sd-turbo",
        "tiny-sd": "segmind/tiny-sd",
        "bk-sdm-tiny": "nota-ai/bk-sdm-tiny",
    }

    # Action-Modulated Dual-LoRA (PiSA-SR Pixel vs. Semantic + S3Diff Action Gating)
    UNET_LORA_RANK = 16
    UNET_LORA_ALPHA = 16.0
    VAE_ENC_LORA_RANK = 8
    VAE_ENC_LORA_ALPHA = 8.0
    VAE_DEC_LORA_RANK = 8
    VAE_DEC_LORA_ALPHA = 8.0

    # Native OpenCLIP quality prompt anchor (precomputed once on CPU -> 0 MB text-encoder GPU VRAM)
    DEFAULT_POS_PROMPT = (
        "a high-resolution, sharp, clean, natural photograph, detailed texture, "
        "smooth gradients, accurate colors, crisp focus"
    )
    PROMPT_SEQ_LEN = 77

    # Default inference Dual-LoRA scales
    DEFAULT_FIDELITY_SCALE = 1.0   # lambda_pix: strict structural & color accuracy
    DEFAULT_DETAIL_SCALE = 0.75    # lambda_sem: perceptual/semantic texture synthesis (0.0 = conservative)

    # SD-Turbo Native ADD Midpoint (t=499) Generative Formulation & High-Res Tiled Inference:
    # t=499 is one of SD-Turbo's 4 exact ADD-distilled timesteps [999, 749, 499, 249]
    # (sqrt_alpha_bar=0.5269, sigma=1.613: avoids t=999's 14.65x hallucination blowup & t=249's timid 0.69x step)
    DEFAULT_INFERENCE_STEPS = 1    # 1 step default; supports 2-4 iterative refinement steps
    RESTORATION_MAX_TIMESTEP = 499 # Native SD-Turbo ADD mid-timestep balancing strong synthesis & layout anchor
    INFERENCE_TILE_SIZE = 512      # Seamless overlapping tile size for 2K/4K photos (0 = untiled)
    INFERENCE_TILE_OVERLAP = 96    # Raised-cosine feather overlap in pixels between adjacent tiles
    MULTI_STEP_TRAIN_PROB = 0.0    # Pure 1-step t=499 training (avoids mixed-t conditional blur)
    INITIAL_VELOCITY_SCALE = 1.0   # Native 1.0x DDPM eps prediction scale
    SKIP_DROPOUT_PROB = 0.50       # Drop encoder skips on 50% of train samples so UNet masters all 8 tasks directly

    # -------------------------------------------------------------------------
    # 3. Controllable Action Space & Spatial Pixel Overlay (1x1 SPADE)
    # -------------------------------------------------------------------------
    DEGRADATION_TYPES = [
        "gaussian_noise",
        "motion_blur",
        "jpeg_compression",
        "color_degradation",
        "exposure_issue",
        "color_banding",
        "downsample_upsample",
        "defocus_blur",
    ]
    NUM_ACTIONS = len(DEGRADATION_TYPES)
    ACTION_HIDDEN_DIM = 128
    SPATIAL_MOD_HIDDEN_DIM = 64

    # Below this max action value at a pixel (x, y), lock residual to exact 0.0 preservation
    ZERO_ACTION_LOCK_THRESHOLD = 0.03

    # -------------------------------------------------------------------------
    # 4. Seed-Synchronized Selective Degradation Pipeline
    # -------------------------------------------------------------------------
    MIN_SEVERITY = 0.30
    MAX_COMPOUND_DEGRADATIONS = 3
    COMPOUND_PROB = 0.75

    # Spatial localization mix during training
    SPATIAL_MASK_PROB = 0.25       # 25% localized smooth/object masks, 75% uniform full-patch restoration

    # Action sampling distribution (must sum to 1.0; avoids flooding inactive sliders with no-op labels)
    ACTION_DIST_SELECTIVE = 0.24   # Fix subset of active degradations, strictly preserve others (a_j = 0)
    ACTION_DIST_PARTIAL = 0.14     # Fractional removal sliders a_k in [0.25, 0.90]
    ACTION_DIST_FULL = 0.52        # Strong full-restoration supervision (80% active-only=1.0, 20% global 'all'=1.0)
    ACTION_DIST_FALSE_POS = 0.05   # Non-zero slider on clean attributes (teaches safe no-op on clean areas)
    ACTION_DIST_IDENTITY = 0.05    # All-zero action vector -> exact identity preservation

    # Physical camera degradation parameter ranges
    GAUSSIAN_NOISE_STD_RANGE = (8.0, 50.0)
    MOTION_BLUR_KERNEL_RANGE = (5, 27)
    MOTION_BLUR_ANGLE_RANGE = (0.0, 180.0)
    JPEG_QUALITY_RANGE = (10, 55)
    COLOR_SHIFT_MAX = 0.35
    SATURATION_LOSS_MAX = 0.55
    HUE_SHIFT_MAX = 3.0
    BRIGHTNESS_MAX = 0.25
    CONTRAST_MAX = 0.35
    GAMMA_RANGE = (0.55, 1.85)
    DOWNSAMPLE_SCALE_RANGE = (0.25, 0.75)
    DEFOCUS_SIGMA_RANGE = (0.8, 5.5)
    BANDING_LEVELS_RANGE = (6, 26)

    # -------------------------------------------------------------------------
    # 5. Dataset, Hardware & 3-Phase Training Schedule
    # -------------------------------------------------------------------------
    DATASET_NAME = "DIV2K"
    DATASET_PATH = "./data/DIV2K"
    TRAIN_DATASET_PATH = None
    VAL_DATASET_PATH = None
    TRAIN_IMAGES_COUNT = 0         # 0 = use all available images
    VAL_IMAGES_COUNT = 0
    MAX_STEPS_PER_EPOCH = 0        # 0 = full dataset per epoch; >0 = cap steps/epoch (reshuffled slice each epoch)

    OUTPUT_DIR = "./outputs"
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"

    PATCH_SIZE = 256
    BATCH_SIZE = 8                 # 4 per T4 GPU on 2x T4 (~8 GB/15 GB VRAM safe across all 3 phases)
    NUM_WORKERS = 4
    DATALOADER_PREFETCH_FACTOR = 4
    DECODE_CACHE_SIZE = 16         # Per-worker staggered 2K-PNG decode cache (cuts FUSE/libpng CPU cost ~6x)
    CROPS_PER_DECODE = 6
    DEVICE = "cuda"
    USE_AMP = True
    GRADIENT_CHECKPOINTING = True  # Cuts UNet + Skip-VAE activation VRAM by ~7 GB on 15 GB T4 GPUs
    MULTI_GPU = True
    SEED = 42

    # 3-Phase Staged Training ("staged" or "joint"):
    #   Phase 1 (Epochs 1..ALIGN_WARMUP_EPOCHS): Self-Teacher Vision LoRA warmup on L_align
    #   Phase 2 (Next PIXEL_STAGE_EPOCHS):       Train Pixel-LoRA + Skip-VAE + Action Overlay (lambda_sem=0)
    #   Phase 3 (Remaining epochs):              Freeze Pixel-LoRA & Skip-VAE; train Semantic-LoRA for crisp detail
    TRAINING_MODE = "staged"
    EPOCHS = 16
    ALIGN_WARMUP_EPOCHS = 1
    PIXEL_STAGE_EPOCHS = 5         # Epochs 2..6 = Pixel-Faithful Stage; Epochs 7..16 = Semantic-Detail + GAN Stage

    LEARNING_RATE = 8e-5           # SD-Turbo LoRA sweet spot (prevents 2e-4 attention overshoot)
    VISION_LORA_LR = 1e-4
    DISC_LR = 8e-5
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 1.0
    CHECKPOINT_EVERY_N_EPOCHS = 5

    # Change-Focused + Contrastive Perceptual + C-RADIO Semantic + Vision-Aided GAN Loss Weights:
    # Prevents lazy-identity minimum (pred ~= src) and blurry conditional-mean L1 collapse
    LOSS_L1_WEIGHT = 0.55
    LOSS_SEMANTIC_STAGE_L1_WEIGHT = 0.25  # Relax pixelwise L1 in Phase 3 so Semantic-LoRA + GAN synthesize crisp detail
    LOSS_GRADIENT_WEIGHT = 0.30
    LOSS_FREQUENCY_WEIGHT = 0.15   # Shift-invariant log-amplitude spectrum matching (restores high-freq deblur/SR power)
    LOSS_COLOR_TONE_WEIGHT = 0.45  # Multi-scale (4x4, 16x16, 32x32) pooled RGB + chroma for cast, exposure & debanding
    LOSS_ALIGN_WEIGHT = 0.15       # Self-Teacher vision encoder feature alignment
    LOSS_PIXEL_STAGE_PERCEPTUAL_WEIGHT = 1.25  # Spatially-focused LPIPS-VGG sharpness in Phase 2
    LOSS_PERCEPTUAL_WEIGHT = 1.50  # Spatially-focused LPIPS-VGG sharpness in Phase 3 (Semantic-LoRA)
    LOSS_CONTRAST_WEIGHT = 0.45    # Contrastive VGG ratio d(pred,tgt)/(d(pred,src)+0.5*d(src,tgt)) defeating pred==src
    LOSS_PIXEL_STAGE_SEM_WEIGHT = 0.25  # Dense C-RADIO patch cosine consistency active right from Phase 2
    LOSS_SEM_CONSISTENCY_WEIGHT = 0.40  # Dense C-RADIO patch cosine consistency in Phase 3
    LOSS_GAN_WEIGHT = 0.20         # Vision-Aided Spectral-Norm PatchGAN loss in Phase 3 for realistic crisp textures
    PERCEPTUAL_LAYERS = ["relu1_2", "relu2_2", "relu3_4", "relu4_4"]

    # Logging & Validation Sweeps
    # Clean line-based logging by default so Kaggle background runs / piped logs never flood with \r bars
    USE_TQDM = False
    LOG_INTERVAL = 100
    SAMPLE_EVERY_N_STEPS = 500     # Save mid-epoch visual sweep every N steps (0 = end-of-epoch only)
    DISPLAY_INLINE_IMAGES = True   # Render saved sweep grids inline in Kaggle/Jupyter cell logs (use `%run train.py`)
    INLINE_IMAGE_MAX_WIDTH = 1800  # Crisp JPEG width (~220 KB/sweep) so background `.ipynb` commits stay fast & light
    VAL_EPOCH_INTERVAL = 1
    EVAL_IMAGE_COUNT = 3


class DebugConfig(Config):
    """Lightweight debug configuration for rapid end-to-end verification."""

    EPOCHS = 3
    ALIGN_WARMUP_EPOCHS = 1
    PIXEL_STAGE_EPOCHS = 1         # Epoch 1: Align, Epoch 2: Pixel-LoRA, Epoch 3: Semantic-LoRA
    BATCH_SIZE = 2
    TRAIN_IMAGES_COUNT = 8
    VAL_IMAGES_COUNT = 2
    LOG_INTERVAL = 2
    SAMPLE_EVERY_N_STEPS = 0
    EVAL_IMAGE_COUNT = 1
