"""Configuration for Controllable Pretrained Image Restoration (2-Step SD-2.1 + Shared/Specialist LoRA-MoE)."""


class Config:
    """Clean 2-stage baseline: 2-Step SD-2.1-Base (t=299->99) + Shared/Specialist LoRA-MoE + Frozen DINOv2-Reg."""

    PROJECT_NAME = "controllable_image_restoration"
    MODEL_VERSION = "v5_2step_moe"

    # -------------------------------------------------------------------------
    # 1. Frozen Pretrained Semantic Vision Encoder (Cross-Attn Context & 1-Click Masks)
    # -------------------------------------------------------------------------
    SEMANTIC_ENCODER_TYPE = "dinov2_reg"
    DINOV2_REG_MODEL_ID = "facebook/dinov2-with-registers-base"
    DINOV2_FALLBACK_MODEL_ID = "facebook/dinov2-with-registers-small"

    # -------------------------------------------------------------------------
    # 2. Pretrained Generator Backbone & Shared+Specialist LoRA-MoE
    # -------------------------------------------------------------------------
    GENERATOR_BACKBONE = "sd-2.1-base"

    GENERATOR_REPO_MAP = {
        "sd-2.1-base": "sd2-community/stable-diffusion-2-1-base",
        "sd-turbo": "stabilityai/sd-turbo",
        "sd-1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "bk-sdm-small": "nota-ai/bk-sdm-small",
        "tiny-sd": "segmind/tiny-sd",
        "bk-sdm-tiny": "nota-ai/bk-sdm-tiny",
    }
    GENERATOR_FALLBACK_REPOS = {
        "sd-2.1-base": [
            "sd2-community/stable-diffusion-2-1-base",
            "Manojb/stable-diffusion-2-1-base",
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
        ],
        "sd-1.5": [
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "runwayml/stable-diffusion-v1-5",
        ],
    }

    # Shared Base Pixel LoRA (always active across 100% of batches) + Lightweight Specialist Experts
    UNET_LORA_RANK = 20
    UNET_LORA_ALPHA = 20.0
    LORA_EXPERT_SPECS = [
        {"name": "clean_smooth", "sliders": ["denoise", "deartifact"], "rank": 8},
        {"name": "motion_deconv", "sliders": ["deblur_motion"], "rank": 8},
        {"name": "focus_optical", "sliders": ["sharpen_focus", "dechromatic"], "rank": 8},
        {"name": "optics_atmos", "sliders": ["deflare", "dereflection", "dehaze"], "rank": 8},
    ]
    SEM_DETAIL_LORA_RANK = 16
    SEM_DETAIL_LORA_ALPHA = 16.0

    # Lightweight VAE Encoder LoRA + 100% Frozen Natural-Manifold Decoder
    VAE_ENC_LORA_RANK = 8
    VAE_ENC_LORA_ALPHA = 8.0
    TONE_HEAD_HIDDEN_DIM = 32

    DEFAULT_POS_PROMPT = (
        "a high-resolution, sharp, clean, natural photograph, realistic micro-texture, "
        "smooth gradients, accurate colors, crisp optical focus"
    )
    DEFAULT_NEG_PROMPT = (
        "blurry, motion blur, out of focus, camera shake, noisy, jpeg artifacts, "
        "oversharpened halos, ringing, wavy fake texture, checkerboard, plastic"
    )
    PROMPT_SEQ_LEN = 77

    # Single user-facing Generative Detail dial (`detail_scale` in [0, 1])
    DEFAULT_FIDELITY_SCALE = 1.0
    DEFAULT_DETAIL_SCALE = 0.35

    # 2-Step Latent Refinement schedule on SD-2.1-base (Step 1 coarse deblur/denoise -> Step 2 crisp polish)
    DEFAULT_INFERENCE_STEPS = 1
    RESTORATION_TIMESTEPS = (299, 99)
    INFERENCE_TILE_SIZE = 512
    INFERENCE_TILE_OVERLAP = 96
    INITIAL_VELOCITY_SCALE = 1.0
    # Drop encoder skips on 15% of training steps so UNet `hat_z_0` carries full sharp deblurring on its own
    SKIP_DROPOUT_PROB = 0.15

    # -------------------------------------------------------------------------
    # 3. Orthogonal Control Space (8 Unsigned [0, 1] Fixes + 2 Signed [-1, +1] Grades)
    # -------------------------------------------------------------------------
    DEGRADATION_TYPES = [
        "denoise",
        "deartifact",
        "deblur_motion",
        "sharpen_focus",
        "deflare",
        "dereflection",
        "dechromatic",
        "dehaze",
        "exposure",
        "warmth",
    ]
    UNSIGNED_ACTION_TYPES = [
        "denoise",
        "deartifact",
        "deblur_motion",
        "sharpen_focus",
        "deflare",
        "dereflection",
        "dechromatic",
        "dehaze",
    ]
    SIGNED_ACTION_TYPES = [
        "exposure",
        "warmth",
    ]
    NUM_ACTIONS = len(DEGRADATION_TYPES)
    ACTION_HIDDEN_DIM = 128
    SPATIAL_MOD_HIDDEN_DIM = 64

    ZERO_ACTION_LOCK_THRESHOLD = 0.03

    # -------------------------------------------------------------------------
    # 4. Linear-Light Curved/Tremor Motion, Optical Bokeh/CA & Richer Synthesis
    # -------------------------------------------------------------------------
    MIN_SEVERITY = 0.25
    MAX_COMPOUND_DEGRADATIONS = 4
    COMPOUND_PROB = 0.75
    CURRICULUM_WARMUP_EPOCHS = 2

    SPATIAL_MASK_PROB = 0.25

    # Clean-Patch Laplacian Sharpness Gate
    CLEAN_PATCH_SHARPNESS_CANDIDATES = 3
    CLEAN_PATCH_MIN_LAPLACIAN_VAR = 18.0
    CLEAN_PATCH_SMOOTH_ALLOW_PROB = 0.15

    # Action sampling distribution (must sum to 1.0):
    ACTION_DIST_SELECTIVE = 0.22
    ACTION_DIST_PARTIAL = 0.14
    ACTION_DIST_FULL = 0.50
    ACTION_DIST_FALSE_POS = 0.08
    ACTION_DIST_IDENTITY = 0.06

    # Physical camera & compression parameter ranges
    GAUSSIAN_NOISE_STD_RANGE = (8.0, 48.0)
    MOTION_BLUR_KERNEL_RANGE = (6, 27)
    MOTION_BLUR_ANGLE_RANGE = (0.0, 180.0)
    MOTION_CURVE_MAX_BEND = 0.38
    DEFOCUS_SIGMA_RANGE = (0.8, 4.4)
    BOKEH_DISC_RADIUS_RANGE = (1.2, 6.5)
    DOWNSAMPLE_SCALE_RANGE = (0.30, 0.75)
    JPEG_QUALITY_RANGE = (10, 52)
    BANDING_LEVELS_RANGE = (7, 26)
    EXPOSURE_STRENGTH_KAPPA = 2.15
    WARMTH_R_KAPPA = 1.15
    WARMTH_G_KAPPA = 0.12
    WARMTH_B_KAPPA = 1.30

    # -------------------------------------------------------------------------
    # 5. Hybrid Multi-Dataset Paths, Hardware & 2-Stage Training Schedule
    # -------------------------------------------------------------------------
    DATASET_NAME = "DIV2K"
    DATASET_PATH = "./data/DIV2K"
    TRAIN_DATASET_PATH = None
    VAL_DATASET_PATH = None

    PAIRED_MOTION_DATASET_PATH = None
    PAIRED_DEFOCUS_DATASET_PATH = None
    PAIRED_REAL_PROB = 0.42
    REUSE_PAIRED_CLEAN_FOR_SYNTH = True

    TRAIN_IMAGES_COUNT = 0
    VAL_IMAGES_COUNT = 0
    MIN_STEPS_PER_EPOCH = 1000
    MAX_STEPS_PER_EPOCH = 0

    OUTPUT_DIR = "./outputs"
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"

    PATCH_SIZE = 256
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    DATALOADER_PREFETCH_FACTOR = 4
    DECODE_CACHE_SIZE = 16
    CROPS_PER_DECODE = 8
    DEVICE = "cuda"
    USE_AMP = True
    GRADIENT_CHECKPOINTING = True
    MULTI_GPU = True
    SEED = 42

    # 2-Stage Training:
    #   Stage 1 (Epochs 1..12):  Pixel Stage (Spatial Control + 2-Step Pixel/Latent Fidelity)
    #   Stage 2 (Epochs 13..15): Semantic Stage (DISTS Texture + Subpixel Anchor + Zero-Drift CSD)
    TRAINING_MODE = "staged"
    EPOCHS = 15
    PIXEL_STAGE_EPOCHS = 12

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 1.0
    CHECKPOINT_EVERY_N_EPOCHS = 4

    # Clean Loss Objective:
    LOSS_L1_WEIGHT = 1.0
    LOSS_LATENT_WEIGHT = 0.50
    LOSS_PIXEL_STAGE_PERCEPTUAL_WEIGHT = 0.10
    LOSS_PRESERVE_WEIGHT = 0.50

    LOSS_SEMANTIC_STAGE_LP_L1_WEIGHT = 0.35
    LOSS_SEMANTIC_STAGE_FULL_L1_WEIGHT = 0.50  # Strong subpixel anchor preventing fine-detail drift
    LOSS_SEMANTIC_STAGE_LATENT_LP_WEIGHT = 0.25
    LOSS_PERCEPTUAL_WEIGHT = 0.60
    LOSS_CSD_WEIGHT = 0.15
    CSD_CFG_SCALE = 5.0
    CSD_STEP_SCALE = 0.20
    CSD_TIMESTEP_RANGE = (50, 400)
    PERCEPTUAL_LAYERS = ["relu1_2", "relu2_2", "relu3_4", "relu4_4"]

    USE_TQDM = False
    LOG_INTERVAL = 100
    SAMPLE_EVERY_N_STEPS = 500
    DISPLAY_INLINE_IMAGES = True
    INLINE_IMAGE_MAX_WIDTH = 1800
    VAL_EPOCH_INTERVAL = 1
    EVAL_IMAGE_COUNT = 3


class PilotConfig(Config):
    """Balanced ~5-hour 2-stage preset (`--pilot` / `--5h`) for Kaggle 2x T4 (9 pixel + 3 semantic epochs)."""

    EPOCHS = 12
    PIXEL_STAGE_EPOCHS = 9
    CURRICULUM_WARMUP_EPOCHS = 2
    LEARNING_RATE = 1.15e-4
    MAX_STEPS_PER_EPOCH = 950
    MIN_STEPS_PER_EPOCH = 950
    VAL_IMAGES_COUNT = 60
    LOG_INTERVAL = 100
    SAMPLE_EVERY_N_STEPS = 475
    CHECKPOINT_EVERY_N_EPOCHS = 3


class DebugConfig(Config):
    """Lightweight debug configuration for rapid end-to-end verification."""

    EPOCHS = 2
    PIXEL_STAGE_EPOCHS = 1
    CURRICULUM_WARMUP_EPOCHS = 0
    MIN_STEPS_PER_EPOCH = 0
    BATCH_SIZE = 2
    TRAIN_IMAGES_COUNT = 8
    VAL_IMAGES_COUNT = 2
    LOG_INTERVAL = 2
    SAMPLE_EVERY_N_STEPS = 0
    EVAL_IMAGE_COUNT = 1
