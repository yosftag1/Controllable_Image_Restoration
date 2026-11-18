"""Configuration for Controllable Pretrained Image Restoration (V5 Universal Slider-Routed Experts)."""


class Config:
    """Main V5 configuration: SD-2.1-Base + Expandable Slider-Routed UIR-LoRA Experts + Universal eVAE."""

    PROJECT_NAME = "controllable_image_restoration"
    MODEL_VERSION = "v5_universal_experts"

    # -------------------------------------------------------------------------
    # 1. Pretrained Semantic Vision Encoder ("The Eyes" + Self-Teacher LoRA)
    # -------------------------------------------------------------------------
    SEMANTIC_ENCODER_TYPE = "c_radio"

    RADIO_MODEL_ID = "nvidia/C-RADIOv3-B"
    RADIO_ALT_MODEL_ID = "nvidia/C-RADIOv2-B"
    DINOV3_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
    DINOV2_FALLBACK_MODEL_ID = "facebook/dinov2-with-registers-small"
    SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"

    VISION_LORA_RANK = 8
    VISION_LORA_ALPHA = 16.0
    VISION_LORA_DROPOUT = 0.0

    # -------------------------------------------------------------------------
    # 2. Pretrained Generator Backbones & Expandable Slider-Routed UIR-LoRA
    # -------------------------------------------------------------------------
    # Default switched from `sd-turbo` to `sd-2.1-base`:
    # - Eliminates SD-Turbo's baked-in ADD high-contrast/oversharpening artifacts.
    # - Unlocks zero-extra-VRAM Target-Anchored Classifier Score Distillation (CSD/TSD)
    #   with negative prompts in Stage 3 (`LoRA OFF` on the same frozen base UNet!).
    GENERATOR_BACKBONE = "sd-2.1-base"

    # Note: Stability AI retired `stabilityai/stable-diffusion-2-1-base`; we default to the
    # verified diffusers mirror `sd2-community/stable-diffusion-2-1-base` with automatic mirror fallback!
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

    # Expandable Task-Routed Pixel LoRA Experts (UIR-LoRA + Strip/Multi-Scale Conv-LoRA):
    # - Each expert is routed deterministically by its assigned user sliders (zero estimator,
    #   zero cross-task gradient conflict between smoothing noise vs. deconvolving blur!).
    # - Future-proof: to add new knobs later (e.g. `lens_distortion`, `deflare`), append the
    #   knob to `DEGRADATION_TYPES` and append a new expert spec dict below. Existing named
    #   experts load seamlessly from checkpoints without retraining from scratch.
    LORA_EXPERT_SPECS = [
        {
            "name": "clean_smooth",
            "sliders": ["denoise", "deartifact"],
            "rank": 24,
            "conv_mode": "standard_3x3",
        },
        {
            "name": "motion_deconv",
            "sliders": ["deblur_motion"],
            "rank": 32,
            "conv_mode": "strip_1x7_7x1",
        },
        {
            "name": "focus_optical",
            "sliders": ["sharpen_focus"],
            "rank": 32,
            "conv_mode": "multiscale_5x5",
        },
    ]

    # Fallback / base rank & decoupled Stage-3 Semantic-Detail LoRA rank
    UNET_LORA_RANK = 28
    UNET_LORA_ALPHA = 28.0
    SEM_DETAIL_LORA_RANK = 16
    SEM_DETAIL_LORA_ALPHA = 16.0
    USE_DORA = True
    # Keep base UNet norms frozen in main V5 so LoRA-OFF exposes exact `sd-2.1-base` for Stage-3 CSD!
    UNFREEZE_UNET_NORMS = False

    # Universal eVAE Encoder LoRA + 100% Frozen Natural-Manifold Decoder (`VAE_DEC_LORA_RANK = 0`)
    VAE_ENC_LORA_RANK = 16
    VAE_ENC_LORA_ALPHA = 16.0
    VAE_DEC_LORA_RANK = 0
    VAE_DEC_LORA_ALPHA = 0.0
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

    # Single user-facing Generative Detail dial (`detail_scale` in [0, 1]; conservative natural default)
    DEFAULT_FIDELITY_SCALE = 1.0
    DEFAULT_DETAIL_SCALE = 0.35

    # Slider-Driven Adaptive Timestep Schedule (`FideDiff` principle without a blind estimator!):
    # Maps active structural slider strength smoothly into [RESTORATION_MIN_TIMESTEP, RESTORATION_MAX_TIMESTEP].
    DEFAULT_INFERENCE_STEPS = 1
    ADAPTIVE_TIMESTEP = True
    RESTORATION_MIN_TIMESTEP = 49
    RESTORATION_MAX_TIMESTEP = 279
    INFERENCE_TILE_SIZE = 512
    INFERENCE_TILE_OVERLAP = 96
    MULTI_STEP_TRAIN_PROB = 0.0
    INITIAL_VELOCITY_SCALE = 1.0
    SKIP_DROPOUT_PROB = 0.0

    # -------------------------------------------------------------------------
    # Separate Experimental HYPIR Mode (Disabled by default; opt-in via --hypir-mode)
    # -------------------------------------------------------------------------
    # Kept strictly isolated from the main V5 pipeline so HYPIR (deterministic fixed-step
    # score-prior + Relativistic Vision RpGAN without CSD) can be benchmarked separately.
    HYPIR_MODE = False
    HYPIR_FIXED_TIMESTEP = 200
    HYPIR_LORA_RANK_SCALE = 2.0
    LOSS_RPGAN_WEIGHT = 0.0

    # -------------------------------------------------------------------------
    # 3. Orthogonal Control Space (4 Unsigned [0, 1] Fixes + 2 Signed [-1, +1] Grades)
    # -------------------------------------------------------------------------
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

    ZERO_ACTION_LOCK_THRESHOLD = 0.03

    # -------------------------------------------------------------------------
    # 4. Linear-Light Curved Motion, Optical Bokeh Disc & Directional Synthesis
    # -------------------------------------------------------------------------
    MIN_SEVERITY = 0.25
    MAX_COMPOUND_DEGRADATIONS = 3
    COMPOUND_PROB = 0.70
    CURRICULUM_WARMUP_EPOCHS = 2

    SPATIAL_MASK_PROB = 0.25

    # Clean-Patch Laplacian Sharpness Gate (purges blurry background bokeh & flat web JPEGs from GT!)
    CLEAN_PATCH_SHARPNESS_CANDIDATES = 3
    CLEAN_PATCH_MIN_LAPLACIAN_VAR = 18.0
    CLEAN_PATCH_SMOOTH_ALLOW_PROB = 0.15

    # Action sampling distribution (must sum to 1.0):
    ACTION_DIST_SELECTIVE = 0.22
    ACTION_DIST_PARTIAL = 0.14
    ACTION_DIST_FULL = 0.42
    ACTION_DIST_FALSE_POS = 0.16   # Strong no-op & clean-bokeh safety when blur/sharpen sliders are high
    ACTION_DIST_IDENTITY = 0.06

    # Physical camera & compression parameter ranges
    GAUSSIAN_NOISE_STD_RANGE = (8.0, 48.0)
    MOTION_BLUR_KERNEL_RANGE = (5, 25)
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
    # 5. Hybrid Multi-Dataset Paths, Hardware & Kaggle 2x T4 Training Schedule
    # -------------------------------------------------------------------------
    DATASET_NAME = "DIV2K"
    DATASET_PATH = "./data/DIV2K"
    TRAIN_DATASET_PATH = None
    VAL_DATASET_PATH = None

    # Optional real paired camera deblurring roots (e.g. GoPro/HIDE/RealBlur pack + DPDD defocus):
    # Wrapped via Compositional Real-Pair Wrapper so all 6 selective sliders train on REAL camera blur!
    PAIRED_MOTION_DATASET_PATH = None
    PAIRED_DEFOCUS_DATASET_PATH = None
    PAIRED_REAL_PROB = 0.35

    TRAIN_IMAGES_COUNT = 0
    VAL_IMAGES_COUNT = 0
    MIN_STEPS_PER_EPOCH = 1000
    MAX_STEPS_PER_EPOCH = 0
    MAX_ALIGN_STEPS = 0

    OUTPUT_DIR = "./outputs"
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"

    PATCH_SIZE = 256
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    DATALOADER_PREFETCH_FACTOR = 4
    DECODE_CACHE_SIZE = 16
    CROPS_PER_DECODE = 6
    DEVICE = "cuda"
    USE_AMP = True
    GRADIENT_CHECKPOINTING = True
    MULTI_GPU = True
    SEED = 42

    # 3-Phase Staged Training:
    #   Phase 1 (Epoch 1):       Self-Teacher Vision LoRA warmup
    #   Phase 2 (Epochs 2..13):  Universal Slider-Routed Experts + Universal eVAE Skip Alignment
    #   Phase 3 (Epochs 14..16): Target-Anchored CSD (`PiSA-SR` / `TSD-SR`, 0 MB extra VRAM!) + LPIPS
    TRAINING_MODE = "staged"
    EPOCHS = 16
    ALIGN_WARMUP_EPOCHS = 1
    PIXEL_STAGE_EPOCHS = 12

    LEARNING_RATE = 1e-4
    VISION_LORA_LR = 1e-4
    DISC_LR = 5e-5
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 1.0
    CHECKPOINT_EVERY_N_EPOCHS = 4

    # V5 Artifact-Free Latent + Universal eVAE + Laplacian Pyramid + Target-Anchored CSD Weights:
    LOSS_LATENT_WEIGHT = 0.85             # Direct clean-latent target supervision (hat_z_0 vs frozen_vae(tgt))
    LOSS_EVAE_SKIP_WEIGHT = 0.35          # Universal OSDD-style eVAE skip feature alignment vs clean encoder!
    LOSS_L1_WEIGHT = 1.00
    LOSS_SEMANTIC_STAGE_L1_WEIGHT = 0.85  # Strong pixel anchor in Stage 3 prevents fake texture drift
    LOSS_PYRAMID_WEIGHT = 0.60            # 4-level anti-aliased Laplacian pyramid (local sharpness, zero FFT ringing)
    LOSS_GRADIENT_WEIGHT = 0.35           # Multi-scale edge gradient alignment
    LOSS_HF_GUARD_WEIGHT = 0.50           # Asymmetric hinge penalizing fake high-freq energy > ground truth
    LOSS_FREQUENCY_WEIGHT = 0.10          # Pure complex (real, imag) phase+amp L1 ONLY
    LOSS_COLOR_TONE_WEIGHT = 0.65         # Multi-scale RGB + opponent chroma accuracy
    LOSS_PRESERVE_WEIGHT = 0.50           # Zero-action & zero-DC tone lock when exposure=warmth=0
    LOSS_ALIGN_WEIGHT = 0.15              # Self-Teacher vision encoder alignment
    LOSS_PIXEL_STAGE_PERCEPTUAL_WEIGHT = 0.08
    LOSS_PERCEPTUAL_WEIGHT = 0.22
    LOSS_CSD_WEIGHT = 0.20                # Stage-3 zero-extra-VRAM Target-Anchored CSD (`sd-2.1-base` teacher!)
    CSD_CFG_SCALE = 4.5                   # Negative-prompt suppression strength in Target-Anchored CSD
    CSD_TIMESTEP_RANGE = (40, 320)        # TSD-SR DASM low-to-mid noise band for texture crispness without drift
    LOSS_CONTRAST_WEIGHT = 0.0
    LOSS_PIXEL_STAGE_SEM_WEIGHT = 0.10
    LOSS_SEM_CONSISTENCY_WEIGHT = 0.22
    LOSS_GAN_WEIGHT = 0.0
    PERCEPTUAL_LAYERS = ["relu2_2", "relu3_4", "relu4_4"]

    USE_TQDM = False
    LOG_INTERVAL = 100
    SAMPLE_EVERY_N_STEPS = 500
    DISPLAY_INLINE_IMAGES = True
    INLINE_IMAGE_MAX_WIDTH = 1800
    VAL_EPOCH_INTERVAL = 1
    EVAL_IMAGE_COUNT = 3


class HYPIRConfig(Config):
    """Separate experimental HYPIR configuration (opt-in via `--hypir-mode` for isolated testing).

    Implements HYPIR (SIGGRAPH Asia 2025): deterministic fixed-step score-prior restoration
    with high-rank LoRA (`2x` rank scale) + pretrained C-RADIO Relativistic GAN (`RpGAN`:
    `D(pred) - D(gt)`), disabling diffusion timestep variation and CSD teacher distillation.
    """

    MODEL_VERSION = "v5_hypir_experimental"
    HYPIR_MODE = True
    UNFREEZE_UNET_NORMS = True
    ADAPTIVE_TIMESTEP = False
    RESTORATION_MIN_TIMESTEP = 200
    RESTORATION_MAX_TIMESTEP = 200
    HYPIR_FIXED_TIMESTEP = 200
    HYPIR_LORA_RANK_SCALE = 2.0
    LOSS_CSD_WEIGHT = 0.0
    LOSS_RPGAN_WEIGHT = 0.08


class PilotConfig(Config):
    """Fast ~45-minute end-to-end 3-stage pilot preset (`--pilot` / `--quick`) for large datasets.
    """Balanced ~5-hour end-to-end 3-stage preset (`--pilot` / `--5h`) for Kaggle 2x T4.

    Rotates random non-overlapping slices across your full clean + paired datasets each epoch so
    you get a complete end-to-end preview of all 3 stages (Align -> Pixel Experts/eVAE -> CSD Detail)
    and can later expand seamlessly into a full uncapped run via `--resume checkpoints/model_best_pixel.pt`.
    Runs 13 epochs (`1` align @ `350` steps + `9` pixel @ `950` steps/ep + `3` CSD detail @ `950`
    steps/ep = `11,750` total steps, ~94k patches rotating across your full clean + paired datasets)
    so all 3 stages converge solidly in ~5 hours, and can still be resumed into a longer uncapped run.
    """

    EPOCHS = 8
    EPOCHS = 13
    ALIGN_WARMUP_EPOCHS = 1
    PIXEL_STAGE_EPOCHS = 5
    CURRICULUM_WARMUP_EPOCHS = 1
    LEARNING_RATE = 1.35e-4
    MAX_ALIGN_STEPS = 200
    MAX_STEPS_PER_EPOCH = 350
    MIN_STEPS_PER_EPOCH = 350
    VAL_IMAGES_COUNT = 40
    LOG_INTERVAL = 35
    SAMPLE_EVERY_N_STEPS = 175
    CHECKPOINT_EVERY_N_EPOCHS = 2
    PIXEL_STAGE_EPOCHS = 9
    CURRICULUM_WARMUP_EPOCHS = 2
    LEARNING_RATE = 1.15e-4
    MAX_ALIGN_STEPS = 350
    MAX_STEPS_PER_EPOCH = 950
    MIN_STEPS_PER_EPOCH = 950
    VAL_IMAGES_COUNT = 60
    LOG_INTERVAL = 100
    SAMPLE_EVERY_N_STEPS = 475
    CHECKPOINT_EVERY_N_EPOCHS = 3


class DebugConfig(Config):
    """Lightweight debug configuration for rapid end-to-end verification."""

    EPOCHS = 3
    ALIGN_WARMUP_EPOCHS = 1
    PIXEL_STAGE_EPOCHS = 1
    CURRICULUM_WARMUP_EPOCHS = 0
    MIN_STEPS_PER_EPOCH = 0
    BATCH_SIZE = 2
    TRAIN_IMAGES_COUNT = 8
    VAL_IMAGES_COUNT = 2
    LOG_INTERVAL = 2
    SAMPLE_EVERY_N_STEPS = 0
    EVAL_IMAGE_COUNT = 1
