"""Configuration for Controllable Pretrained Image Restoration (V4 Orthogonal + Directional)."""


class Config:
    """Main configuration for Action-Conditioned Dual-LoRA + Skip-VAE + Pixel-Refiner Restoration."""

    PROJECT_NAME = "controllable_image_restoration"
    MODEL_VERSION = "v4_orthogonal_directional"

    # -------------------------------------------------------------------------
    # 1. Pretrained Semantic Vision Encoder ("The Eyes" + Self-Teacher LoRA)
    # -------------------------------------------------------------------------
    # Options: "c_radio" (default), "dinov3_siglip2", or "mock_tiny" (offline unit tests)
    SEMANTIC_ENCODER_TYPE = "c_radio"

    # Hugging Face repository IDs (100% public & ungated defaults)
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
    # 2. Pretrained Diffusion/Flow Generator ("The Hands" + Skip-VAE + Pixel Refiner)
    # -------------------------------------------------------------------------
    # Options: "sd-turbo" (default), "tiny-sd", "bk-sdm-tiny", or "mock_tiny"
    GENERATOR_BACKBONE = "sd-turbo"

    GENERATOR_REPO_MAP = {
        "sd-turbo": "stabilityai/sd-turbo",
        "tiny-sd": "segmind/tiny-sd",
        "bk-sdm-tiny": "nota-ai/bk-sdm-tiny",
    }

    # Action-Modulated Dual-LoRA (PiSA-SR Structural vs. Generative-Detail + S3Diff Action Gating)
    UNET_LORA_RANK = 16
    UNET_LORA_ALPHA = 16.0
    VAE_ENC_LORA_RANK = 8
    VAE_ENC_LORA_ALPHA = 8.0
    VAE_DEC_LORA_RANK = 8
    VAE_DEC_LORA_ALPHA = 8.0
    PIXEL_REFINER_CHANNELS = 40

    # Native OpenCLIP quality prompt anchor (precomputed once on CPU -> 0 MB text-encoder GPU VRAM)
    DEFAULT_POS_PROMPT = (
        "a high-resolution, sharp, clean, natural photograph, detailed texture, "
        "smooth gradients, accurate colors, crisp focus"
    )
    PROMPT_SEQ_LEN = 77

    # Single user-facing Generative Detail dial (`detail_scale` in [0, 1]; `fidelity_scale` stays locked at 1.0)
    DEFAULT_FIDELITY_SCALE = 1.0
    DEFAULT_DETAIL_SCALE = 0.65

    # Calibrated Pre-Scaled Residual Latent Step at SD-Turbo Native ADD t=249:
    # z_in = sqrt_alpha_bar * z_src (0.8218x) matches SD-Turbo's exact training amplitude manifold,
    # and hat_z_0 = z_src - (sqrt_one_minus / sqrt_alpha_bar) * eps_pred preserves 1.0x identity at Step 0!
    DEFAULT_INFERENCE_STEPS = 1
    RESTORATION_MAX_TIMESTEP = 249
    INFERENCE_TILE_SIZE = 512      # Seamless overlapping tile size for 2K/4K photos (0 = untiled)
    INFERENCE_TILE_OVERLAP = 96    # Raised-cosine feather overlap in pixels between adjacent tiles
    MULTI_STEP_TRAIN_PROB = 0.0
    INITIAL_VELOCITY_SCALE = 1.0
    SKIP_DROPOUT_PROB = 0.0        # Zero train/eval skip mismatch; blur-aware gating handles skip suppression

    # -------------------------------------------------------------------------
    # 3. Orthogonal Control Space (4 Unsigned [0, 1] Fixes + 2 Signed [-1, +1] Grades)
    # -------------------------------------------------------------------------
    # Channels 0..3 in [0.0, +1.0]: objective artifact/blur restoration (0 = preserve, 1 = full fix)
    # Channels 4..5 in [-1.0, +1.0]: directional artistic/radiometric controls
    #   - exposure: -1.0 (Darken / recover washout) <-> 0.0 (Keep) <-> +1.0 (Brighten / lift shadows)
    #   - warmth:   -1.0 (Cool / remove warm cast)  <-> 0.0 (Keep) <-> +1.0 (Warm / remove cold cast)
    DEGRADATION_TYPES = [
        "denoise",
        "deartifact",
        "deblur_motion",
        "sharpen_focus",
        "exposure",
        "warmth",
    ]
    UNSIGNED_ACTION_TYPES = [
        "denoise",
        "deartifact",
        "deblur_motion",
        "sharpen_focus",
    ]
    SIGNED_ACTION_TYPES = [
        "exposure",
        "warmth",
    ]
    NUM_ACTIONS = len(DEGRADATION_TYPES)
    ACTION_HIDDEN_DIM = 128
    SPATIAL_MOD_HIDDEN_DIM = 64

    # Below this max |action| magnitude at pixel (x, y), lock residual to exact 0.0 preservation
    ZERO_ACTION_LOCK_THRESHOLD = 0.03

    # -------------------------------------------------------------------------
    # 4. Seed-Synchronized Selective & Directional Synthesis Pipeline
    # -------------------------------------------------------------------------
    MIN_SEVERITY = 0.28
    MAX_COMPOUND_DEGRADATIONS = 3
    COMPOUND_PROB = 0.72
    CURRICULUM_WARMUP_EPOCHS = 2   # Early pixel-stage epochs emphasize clean 1-to-2 slider credit assignment

    # Spatial localization mix during training
    SPATIAL_MASK_PROB = 0.25       # 25% localized smooth/object masks, 75% uniform full-patch control

    # Action sampling distribution (must sum to 1.0):
    ACTION_DIST_SELECTIVE = 0.22   # Fix subset of active degradations, strictly preserve others (a_j = 0)
    ACTION_DIST_PARTIAL = 0.14     # Smooth fractional restoration in (0, 1) / directional tone steps
    ACTION_DIST_FULL = 0.44        # Full restoration of all active degradations (exact clean target)
    ACTION_DIST_FALSE_POS = 0.14   # 50% false-positive no-op safety on clean objective sliders + 50% pure artistic exposure/warmth grading
    ACTION_DIST_IDENTITY = 0.06    # All-zero action vector -> exact identity preservation

    # Physical camera & compression parameter ranges
    GAUSSIAN_NOISE_STD_RANGE = (8.0, 48.0)
    MOTION_BLUR_KERNEL_RANGE = (5, 25)
    MOTION_BLUR_ANGLE_RANGE = (0.0, 180.0)
    DEFOCUS_SIGMA_RANGE = (0.8, 4.8)
    DOWNSAMPLE_SCALE_RANGE = (0.28, 0.72)
    JPEG_QUALITY_RANGE = (10, 52)
    BANDING_LEVELS_RANGE = (7, 26)
    EXPOSURE_STRENGTH_KAPPA = 2.15 # Smooth rational EV curve strength: g = 2^(kappa * e), e in [-1, 1]
    WARMTH_R_KAPPA = 1.15          # Planckian R/G/B rational white-balance curve strengths
    WARMTH_G_KAPPA = 0.12
    WARMTH_B_KAPPA = 1.30

    # -------------------------------------------------------------------------
    # 5. Dataset, Hardware & Kaggle 2x T4 Training Schedule (~28k steps / ~9 hrs)
    # -------------------------------------------------------------------------
    DATASET_NAME = "DIV2K"
    DATASET_PATH = "./data/DIV2K"
    TRAIN_DATASET_PATH = None
    VAL_DATASET_PATH = None
    TRAIN_IMAGES_COUNT = 0         # 0 = use all available images
    VAL_IMAGES_COUNT = 0
    MIN_STEPS_PER_EPOCH = 1000     # Prevents small folders (e.g. 800-image DIV2K) from starving at 100 steps/epoch
    MAX_STEPS_PER_EPOCH = 0        # 0 = full dataset per epoch; >0 = cap steps/epoch (reshuffled slice each epoch)

    OUTPUT_DIR = "./outputs"
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"

    PATCH_SIZE = 256
    BATCH_SIZE = 8                 # 4 per T4 GPU on 2x T4 (~8.5 GB/15 GB VRAM safe across all 3 phases)
    NUM_WORKERS = 4
    DATALOADER_PREFETCH_FACTOR = 4
    DECODE_CACHE_SIZE = 16
    CROPS_PER_DECODE = 6
    DEVICE = "cuda"
    USE_AMP = True
    GRADIENT_CHECKPOINTING = True
    MULTI_GPU = True
    SEED = 42

    # 3-Phase Staged Training ("staged" or "joint"):
    #   Phase 1 (Epoch 1):      Self-Teacher Vision LoRA warmup on L_align
    #   Phase 2 (Epochs 2..12): Train Pixel-LoRA + Skip-VAE + PixelRefiner + Action/Semantic Overlays (~73% of run!)
    #   Phase 3 (Epochs 13..16): Freeze structural base; train Semantic-Detail LoRA + GAN with high L1 anchor
    TRAINING_MODE = "staged"
    EPOCHS = 16
    ALIGN_WARMUP_EPOCHS = 1
    PIXEL_STAGE_EPOCHS = 11

    LEARNING_RATE = 1e-4
    VISION_LORA_LR = 1e-4
    DISC_LR = 6e-5
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 1.0
    CHECKPOINT_EVERY_N_EPOCHS = 4

    # Calibrated Loss Weights (RGB/Edge/Phase/Tone fidelity anchored on par with bounded LPIPS):
    LOSS_L1_WEIGHT = 1.00
    LOSS_SEMANTIC_STAGE_L1_WEIGHT = 0.75  # Keep strong pixel anchor in Stage 3 so GAN never washes out PSNR/tone
    LOSS_GRADIENT_WEIGHT = 0.55           # 3-scale (1x, 2x, 4x) H/V/diag edge pyramid for motion & defocus deblur
    LOSS_FREQUENCY_WEIGHT = 0.25          # Phase-aligned complex rfft2 + focal log-amplitude spectrum loss
    LOSS_COLOR_TONE_WEIGHT = 0.65         # Multi-scale (1x1 mean/std, 4x4, 16x16) RGB + opponent chroma loss
    LOSS_PRESERVE_WEIGHT = 0.50           # Explicit zero-action & zero-DC tone preservation when exposure=warmth=0
    LOSS_ALIGN_WEIGHT = 0.15              # Self-Teacher vision encoder feature alignment
    LOSS_PIXEL_STAGE_PERCEPTUAL_WEIGHT = 0.35  # Strictly unit-normalized LPIPS-VGG in Stage 2
    LOSS_PERCEPTUAL_WEIGHT = 0.55              # Strictly unit-normalized LPIPS-VGG in Stage 3
    LOSS_CONTRAST_WEIGHT = 0.15                # Bounded contrastive VGG ratio defeating lazy pred==src early on
    LOSS_PIXEL_STAGE_SEM_WEIGHT = 0.15         # Dense C-RADIO patch cosine consistency in Stage 2
    LOSS_SEM_CONSISTENCY_WEIGHT = 0.25         # Dense C-RADIO patch cosine consistency in Stage 3
    LOSS_GAN_WEIGHT = 0.08                     # Subtle Vision-Aided PatchGAN crispness in Stage 3
    PERCEPTUAL_LAYERS = ["relu1_2", "relu2_2", "relu3_4", "relu4_4"]

    # Logging & Validation Sweeps
    USE_TQDM = False
    LOG_INTERVAL = 100
    SAMPLE_EVERY_N_STEPS = 500
    DISPLAY_INLINE_IMAGES = True
    INLINE_IMAGE_MAX_WIDTH = 1800
    VAL_EPOCH_INTERVAL = 1
    EVAL_IMAGE_COUNT = 3


class DebugConfig(Config):
    """Lightweight debug configuration for rapid end-to-end verification."""

    EPOCHS = 3
    ALIGN_WARMUP_EPOCHS = 1
    PIXEL_STAGE_EPOCHS = 1         # Epoch 1: Align, Epoch 2: Pixel Stage, Epoch 3: Semantic Stage
    CURRICULUM_WARMUP_EPOCHS = 0
    MIN_STEPS_PER_EPOCH = 0
    BATCH_SIZE = 2
    TRAIN_IMAGES_COUNT = 8
    VAL_IMAGES_COUNT = 2
    LOG_INTERVAL = 2
    SAMPLE_EVERY_N_STEPS = 0
    EVAL_IMAGE_COUNT = 1
