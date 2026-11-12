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

    # Default inference Dual-LoRA scales
    DEFAULT_FIDELITY_SCALE = 1.0   # lambda_pix: strict structural & color accuracy
    DEFAULT_DETAIL_SCALE = 0.75    # lambda_sem: perceptual/semantic texture synthesis (0.0 = conservative)

    # Latent Rectified Flow / Turbo sampling
    DEFAULT_INFERENCE_STEPS = 1    # 1 step default; supports 2-4 Euler steps for severe compound cases
    MULTI_STEP_TRAIN_PROB = 0.30   # 70% t=1.0 (1-step direct), 30% t~U(0.25, 1.0) (enables 2-4 step trajectory)
    INITIAL_VELOCITY_SCALE = 0.15  # Smooth initial latent step before LoRA converges

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
    MIN_SEVERITY = 0.20
    MAX_COMPOUND_DEGRADATIONS = 3
    COMPOUND_PROB = 0.65

    # Spatial localization mix during training
    SPATIAL_MASK_PROB = 0.40       # 40% localized smooth/object masks, 60% uniform whole-image sliders

    # Action sampling distribution (must sum to 1.0)
    ACTION_DIST_SELECTIVE = 0.30   # Fix subset of active degradations, strictly preserve others (a_j = 0)
    ACTION_DIST_PARTIAL = 0.25     # Fractional removal sliders a_k in [0.15, 0.85]
    ACTION_DIST_FULL = 0.20        # Full removal of active degradations
    ACTION_DIST_FALSE_POS = 0.15   # Non-zero slider on clean attributes (teaches safe no-op on clean areas)
    ACTION_DIST_IDENTITY = 0.10    # All-zero action vector -> exact identity preservation

    # Physical camera degradation parameter ranges
    GAUSSIAN_NOISE_STD_RANGE = (8.0, 50.0)
    MOTION_BLUR_KERNEL_RANGE = (5, 27)
    MOTION_BLUR_ANGLE_RANGE = (0.0, 180.0)
    JPEG_QUALITY_RANGE = (10, 55)
    COLOR_SHIFT_MAX = 0.35
    SATURATION_LOSS_MAX = 0.55
    HUE_SHIFT_MAX = 10.0
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

    OUTPUT_DIR = "./outputs"
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"

    PATCH_SIZE = 256
    BATCH_SIZE = 8                 # 4 per T4 GPU on 2x T4 (16GB VRAM each) in FP16
    NUM_WORKERS = 4
    DEVICE = "cuda"
    USE_AMP = True
    MULTI_GPU = True
    SEED = 42

    # 3-Phase Staged Training ("staged" or "joint"):
    #   Phase 1 (Epochs 1..ALIGN_WARMUP_EPOCHS): Self-Teacher Vision LoRA warmup on L_align
    #   Phase 2 (Next PIXEL_STAGE_EPOCHS):       Train Pixel-LoRA + Skip-VAE + Action Overlay (lambda_sem=0)
    #   Phase 3 (Remaining epochs):              Freeze Pixel-LoRA & Skip-VAE; train Semantic-LoRA for crisp detail
    TRAINING_MODE = "staged"
    EPOCHS = 40
    ALIGN_WARMUP_EPOCHS = 2
    PIXEL_STAGE_EPOCHS = 22        # Epochs 3..24 = Pixel-Faithful Stage; Epochs 25..40 = Semantic-Detail Stage

    LEARNING_RATE = 2e-4
    VISION_LORA_LR = 1e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 1.0
    CHECKPOINT_EVERY_N_EPOCHS = 5

    # Loss weights
    LOSS_L1_WEIGHT = 1.0
    LOSS_GRADIENT_WEIGHT = 0.20
    LOSS_FREQUENCY_WEIGHT = 0.08
    LOSS_ALIGN_WEIGHT = 0.15       # Self-Teacher vision encoder feature alignment
    LOSS_PERCEPTUAL_WEIGHT = 0.25  # Active in Phase 3 (or joint mode)
    LOSS_SEM_CONSISTENCY_WEIGHT = 0.20  # Semantic feature match between restored & target in Phase 3
    PERCEPTUAL_LAYERS = ["relu2_2", "relu3_4", "relu4_4"]

    # Logging & Validation Sweeps
    LOG_INTERVAL = 50
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
    EVAL_IMAGE_COUNT = 1
