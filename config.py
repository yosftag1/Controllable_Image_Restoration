"""Configuration for Controllable Pretrained Image Restoration (1-Step SD-Turbo + Routed LoRA-MoE)."""


class Config:
    """Crisp 2-stage baseline: 1-Step SD-Turbo (t=499) + Routed Specialist LoRA-MoE + Frozen DINOv2-Reg."""

    PROJECT_NAME = "controllable_image_restoration"
    MODEL_VERSION = "v7_sdturbo_crisp7"

    # -------------------------------------------------------------------------
    # 1. Frozen Pretrained Semantic Vision Encoder (Context, 1-Click Masks & Stage-2 DINOv2 Disc)
    # -------------------------------------------------------------------------
    SEMANTIC_ENCODER_TYPE = "dinov2_reg"
    DINOV2_REG_MODEL_ID = "facebook/dinov2-with-registers-base"
    DINOV2_FALLBACK_MODEL_ID = "facebook/dinov2-with-registers-small"

    # -------------------------------------------------------------------------
    # 2. Pretrained 1-Step Generator Backbone & Routed Specialist LoRA-MoE
    # -------------------------------------------------------------------------
    GENERATOR_BACKBONE = "sd-turbo"

    GENERATOR_REPO_MAP = {
        "sd-turbo": "stabilityai/sd-turbo",
        "sd-2.1-base": "sd2-community/stable-diffusion-2-1-base",
        "sd-1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "bk-sdm-small": "nota-ai/bk-sdm-small",
        "tiny-sd": "segmind/tiny-sd",
        "bk-sdm-tiny": "nota-ai/bk-sdm-tiny",
    }
    GENERATOR_FALLBACK_REPOS = {
        "sd-turbo": [
            "stabilityai/sd-turbo",
            "sd2-community/stable-diffusion-2-1-base",
        ],
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

    # Shared Base Pixel LoRA + Additive Slider-Routed Specialist Experts (no task interference!)
    UNET_LORA_RANK = 16
    UNET_LORA_ALPHA = 16.0
    UNET_CONV_LORA_RANK = 12
    UNET_CONV_LORA_ALPHA = 12.0
    LORA_EXPERT_SPECS = [
        {"name": "motion_deconv", "sliders": ["deblur_motion"], "rank": 12},
        {"name": "sr_detail", "sliders": ["super_resolution"], "rank": 12},
        {"name": "clean_atmos", "sliders": ["denoise", "deartifact", "dehaze"], "rank": 8},
    ]

    # Stage-2 Semantic-Detail LoRA (spatially routed strictly to DETAIL_ACTION_TYPES — never fires on denoise/tone)
    SEM_DETAIL_LORA_RANK = 16
    SEM_DETAIL_LORA_ALPHA = 16.0
    SEM_DETAIL_CONV_LORA_RANK = 12
    SEM_DETAIL_CONV_LORA_ALPHA = 12.0

    # VAE: Disable blind unconditioned encoder LoRA; use detail-gated Stage-2 Decoder LoRA for crisp 1-2px edges!
    VAE_ENC_LORA_RANK = 0
    VAE_ENC_LORA_ALPHA = 0.0
    VAE_DEC_LORA_RANK = 8
    VAE_DEC_LORA_ALPHA = 8.0
    TONE_HEAD_HIDDEN_DIM = 32

    DEFAULT_POS_PROMPT = (
        "a sharp, high-resolution, crisp natural photograph, fine optical detail, "
        "clean micro-texture, accurate colors, natural contrast"
    )
    DEFAULT_NEG_PROMPT = (
        "blurry, motion blur, low resolution, hazy, washed out, noisy, jpeg artifacts, "
        "smeared, painterly, plastic, wavy fake texture, ringing, oversharpened halos"
    )
    PROMPT_SEQ_LEN = 77

    # Single user-facing Generative Detail dial (`detail_scale` in [0, 1]) — gated to DETAIL_ACTION_TYPES only
    DEFAULT_FIDELITY_SCALE = 1.0
    DEFAULT_DETAIL_SCALE = 0.50

    # Native 1-Step ADD-distilled SD-Turbo midpoint (`t=499`: 1/sqrt_ab=1.90, strong crisp prior, stable FP16)
    DEFAULT_INFERENCE_STEPS = 1
    RESTORATION_TIMESTEPS = (499,)
    INFERENCE_TILE_SIZE = 512
    INFERENCE_TILE_OVERLAP = 96
    INITIAL_VELOCITY_SCALE = 1.0
    # Drop encoder skips on 15% of training steps so UNet `hat_z_0` carries full sharp restoration on its own
    SKIP_DROPOUT_PROB = 0.15

    # -------------------------------------------------------------------------
    # 3. Streamlined 7-Control Orthogonal Space (5 Unsigned [0, 1] Fixes + 2 Signed [-1, +1] Grades)
    # -------------------------------------------------------------------------
    DEGRADATION_TYPES = [
        "denoise",
        "deartifact",
        "deblur_motion",
        "super_resolution",
        "dehaze",
        "exposure",
        "warmth",
    ]
    UNSIGNED_ACTION_TYPES = [
        "denoise",
        "deartifact",
        "deblur_motion",
        "super_resolution",
        "dehaze",
    ]
    SIGNED_ACTION_TYPES = [
        "exposure",
        "warmth",
    ]
    # Tasks allowed to activate Stage-2 `sem_branch` (UNet + VAE Dec LoRA) & DINOv2 Disc (keeps `denoise`/tone 100% faithful)
    DETAIL_ACTION_TYPES = [
        "deblur_motion",
        "super_resolution",
        "deartifact",
        "dehaze",
    ]
    # Destructive tasks where input encoder skips are blurry/low-res/veiled and MUST be shut down (~95%+) in Stage 1 & 2
    DESTRUCTIVE_SKIP_ACTIONS = [
        "deblur_motion",
        "super_resolution",
        "dehaze",
    ]
    NUM_ACTIONS = len(DEGRADATION_TYPES)
    ACTION_HIDDEN_DIM = 128
    SPATIAL_MOD_HIDDEN_DIM = 64

    ZERO_ACTION_LOCK_THRESHOLD = 0.03

    # -------------------------------------------------------------------------
    # 4. Artifact-First Balanced Sampling, Single-to-Compound Curriculum & Crisp Synthesis
    # -------------------------------------------------------------------------
    # Primary artifact sampling weights (balanced around artifacts first, then datasets inside each artifact!)
    # Big 3 (`deblur_motion`, `super_resolution`, `denoise`) get ~72% of Stage-1 training budget.
    ARTIFACT_SAMPLING_WEIGHTS = {
        "deblur_motion": 0.26,
        "super_resolution": 0.25,
        "denoise": 0.21,
        "deartifact": 0.10,
        "dehaze": 0.07,
        "exposure": 0.06,
        "warmth": 0.05,
    }
    # Stage-2 artifact sampling weights (heavily focused on crisp detail synthesis tasks)
    STAGE2_ARTIFACT_SAMPLING_WEIGHTS = {
        "super_resolution": 0.42,
        "deblur_motion": 0.40,
        "deartifact": 0.10,
        "dehaze": 0.05,
        "denoise": 0.03,
        "exposure": 0.00,
        "warmth": 0.00,
    }

    # Within `deblur_motion`, fraction using real paired datasets (GoPro/HIDE/RealBlur) vs synthetic DIV2K/DF2K
    MOTION_REAL_PAIR_FRAC = 0.50
    # Within `deblur_motion` compounds, explicit probability of pairing with realistic camera noise (`denoise`)
    MOTION_NOISE_COMPOUND_PROB = 0.45

    MIN_SEVERITY = 0.25
    MAX_COMPOUND_DEGRADATIONS = 3
    # Lower steady-state compound ratio (~38% compound / ~62% single-artifact) + strong single-artifact curriculum!
    COMPOUND_PROB = 0.38
    CURRICULUM_WARMUP_EPOCHS = 4

    SPATIAL_MASK_PROB = 0.20

    # Strict Clean-Patch Laplacian Sharpness Gate (filters out soft/bokeh patches from Flickr2K/OST!)
    CLEAN_PATCH_SHARPNESS_CANDIDATES = 5
    CLEAN_PATCH_MIN_LAPLACIAN_VAR = 65.0
    CLEAN_PATCH_SMOOTH_ALLOW_PROB = 0.08

    # Clean dataset source balancing weights (so 800 crisp DIV2K images aren't drowned out by 10k OST!)
    CLEAN_DATASET_WEIGHTS = {
        "div2k": 0.45,
        "flickr2k": 0.35,
        "lsdir": 0.40,
        "ost": 0.20,
    }

    # Action sampling distribution (must sum to 1.0):
    # Note: `ACTION_DIST_BLIND_ALL` explicitly trains `[1, 1, 1, 1, 1, 0, 0]` on BOTH synth & real pairs!
    ACTION_DIST_FULL = 0.50
    ACTION_DIST_BLIND_ALL = 0.16
    ACTION_DIST_SELECTIVE = 0.16
    ACTION_DIST_PARTIAL = 0.10
    ACTION_DIST_FALSE_POS = 0.04
    ACTION_DIST_IDENTITY = 0.04

    # Physical camera, Real-ESRGAN SR & compression parameter ranges
    GAUSSIAN_NOISE_STD_RANGE = (8.0, 48.0)
    MOTION_BLUR_KERNEL_RANGE = (7, 39)
    MOTION_BLUR_ANGLE_RANGE = (0.0, 180.0)
    MOTION_CURVE_MAX_BEND = 0.38
    MOTION_NONUNIFORM_PROB = 0.25
    MOTION_HDR_STREAK_PROB = 0.55
    PAIRED_NATIVE_CROP_PROB = 0.85
    PAIRED_MIN_MOTION_DIFF = 9.0
    SR_BLUR_SIGMA_RANGE = (0.4, 2.2)
    SR_DOWNSAMPLE_SCALE_RANGE = (0.25, 0.65)
    JPEG_QUALITY_RANGE = (12, 55)
    BANDING_LEVELS_RANGE = (8, 26)
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
    PAIRED_REAL_PROB = 0.50  # Legacy alias synced to `MOTION_REAL_PAIR_FRAC`
    # NEVER dump soft 720p GoPro/HIDE/RealBlur frames into the clean synthetic GT / SR / GAN-real pool!
    REUSE_PAIRED_CLEAN_FOR_SYNTH = False

    TRAIN_IMAGES_COUNT = 0
    VAL_IMAGES_COUNT = 0
    MIN_STEPS_PER_EPOCH = 800   # Optimizer steps per epoch (x GRAD_ACCUM_STEPS micro-batches)
    MAX_STEPS_PER_EPOCH = 0

    OUTPUT_DIR = "./outputs"
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"

    # Train near SD-Turbo's native resolution (configurable via --patch-size 256/384/512 and --grad-accum)
    PATCH_SIZE = 384
    BATCH_SIZE = 4        # Per-optimizer micro-batch across all GPUs (2/GPU on 2x T4 -> fits 384px & 512px cleanly!)
    GRAD_ACCUM_STEPS = 2  # Effective global batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 8
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
    #   Stage 1 (Epochs 1..10):  Pixel Stage (1-Step SD-Turbo + LoRA-MoE: L1 + Latent + Un-blurred LPIPS + Edge Grad)
    #   Stage 2 (Epochs 11..15): Semantic Stage (Fixed d=1.0: True DISTS + Edge Grad + Clean DINOv2 Feature Discriminator)
    TRAINING_MODE = "staged"
    EPOCHS = 15
    PIXEL_STAGE_EPOCHS = 10

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 1.0
    CHECKPOINT_EVERY_N_EPOCHS = 4

    # Clean Phase-Aligned Loss Objective (ZERO phase-blind FFT magnitude, ZERO covariance-free VGG-std!):
    LOSS_L1_WEIGHT = 1.0
    LOSS_LATENT_WEIGHT = 0.25
    LOSS_PIXEL_STAGE_PERCEPTUAL_WEIGHT = 0.35  # Un-blurred full-res VGG perceptual in Stage 1 -> crisp, non-hazy base!
    LOSS_PIXEL_STAGE_EDGE_WEIGHT = 0.25        # Phase-aligned Sobel/Laplacian edge gradient L1 in Stage 1
    LOSS_PRESERVE_WEIGHT = 0.50

    # Stage 2 (`semantic_stage`, trained at fixed detail_scale = 1.0 -> zero overshoot!)
    LOSS_SEMANTIC_STAGE_LP_L1_WEIGHT = 0.35
    LOSS_SEMANTIC_STAGE_FULL_L1_WEIGHT = 0.20  # Anchors exact geometry & color without blurring crisp edges
    LOSS_SEMANTIC_STAGE_LATENT_LP_WEIGHT = 0.10
    LOSS_PERCEPTUAL_WEIGHT = 0.80              # Un-blurred VGG + True Gaussian-Windowed DISTS (with covariance!)
    LOSS_EDGE_GRAD_WEIGHT = 0.35               # Phase-aligned multi-scale Sobel + Laplacian gradient L1
    LOSS_GAN_WEIGHT = 0.15                     # Frozen DINOv2 Multi-Layer Feature Discriminator (clean HR targets ONLY!)
    GAN_DISC_LR = 1.5e-4
    GAN_WARMUP_STEPS = 50
    DINO_DISC_MAX_SIZE = 252                   # 18x18 DINOv2 patch grid (fast & sharp on T4 VRAM)
    LOSS_CSD_WEIGHT = 0.0
    CSD_EVERY_N_STEPS = 4
    CSD_CFG_SCALE = 5.0
    CSD_STEP_SCALE = 0.30
    CSD_TIMESTEP_RANGE = (150, 400)
    PERCEPTUAL_LAYERS = ["relu1_2", "relu2_2", "relu3_4", "relu4_4"]

    USE_TQDM = False
    LOG_INTERVAL = 100
    SAMPLE_EVERY_N_STEPS = 500
    DISPLAY_INLINE_IMAGES = True
    INLINE_IMAGE_MAX_WIDTH = 2000
    VAL_EPOCH_INTERVAL = 1
    EVAL_IMAGE_COUNT = 4
    EVAL_SWEEP_CROP_SIZE = 384


class PilotConfig(Config):
    """Balanced ~3.5-hour 1-step SD-Turbo preset (`--pilot` / `--5h`) for Kaggle 2x T4 (8 pixel + 4 semantic epochs)."""

    EPOCHS = 12
    PIXEL_STAGE_EPOCHS = 8
    CURRICULUM_WARMUP_EPOCHS = 3
    LEARNING_RATE = 1.15e-4
    MAX_STEPS_PER_EPOCH = 700
    MIN_STEPS_PER_EPOCH = 700
    VAL_IMAGES_COUNT = 60
    LOG_INTERVAL = 100
    SAMPLE_EVERY_N_STEPS = 350
    CHECKPOINT_EVERY_N_EPOCHS = 3


class DebugConfig(Config):
    """Lightweight debug configuration for rapid end-to-end verification."""

    EPOCHS = 2
    PIXEL_STAGE_EPOCHS = 1
    CURRICULUM_WARMUP_EPOCHS = 0
    MIN_STEPS_PER_EPOCH = 0
    PATCH_SIZE = 128
    BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 1
    TRAIN_IMAGES_COUNT = 8
    VAL_IMAGES_COUNT = 2
    LOG_INTERVAL = 2
    SAMPLE_EVERY_N_STEPS = 0
    EVAL_IMAGE_COUNT = 1
    EVAL_SWEEP_CROP_SIZE = 128
