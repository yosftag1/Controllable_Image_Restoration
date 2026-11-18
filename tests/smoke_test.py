"""End-to-end verification suite for Controllable_Image_Restoration (V5 Universal Experts + Hybrid Data).

Runs:
1. AST syntax check across every Python file in the repository.
2. V5 config invariants (`sd-2.1-base` default, expandable `LORA_EXPERT_SPECS`, isolated `HYPIRConfig`),
   2D Bezier curved motion blur + optical bokeh disc PSF continuity, Lie-group directional Exposure/Warmth,
   Laplacian clean-patch sharpness gate, and auto-discovery + Compositional Real-Pair Wrapper across
   GoPro / RealBlur / DPDD folder & `.txt` layouts.
3. Full 3-phase forward/backward, Slider-Routed UIR-LoRA expert zero-interference routing, Universal eVAE
   skip alignment, Zero-Extra-VRAM Target-Anchored CSD, and isolated HYPIR Relativistic RpGAN checks
   (when PyTorch is installed in the active environment).
"""

import ast
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_syntax_all_files():
    py_files = sorted(PROJECT_ROOT.rglob("*.py"))
    assert py_files, "No Python files found!"
    for path in py_files:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
    print(f"[PASS] AST syntax verified across {len(py_files)} Python files.")


def test_selective_pipeline():
    import cv2
    import numpy as np
    import tempfile
    from config import Config, HYPIRConfig
    from data.selective_dataset import (
        SelectiveDegradationPipeline,
        create_controllable_dataset,
        discover_paired_blur_datasets,
    )
    from utils.image_benchmark import run_single_image_benchmark
    from utils.semantic_mask import build_spatial_action_overlay, parse_action_spec

    cfg = Config()
    dist_sum = (
        cfg.ACTION_DIST_SELECTIVE
        + cfg.ACTION_DIST_PARTIAL
        + cfg.ACTION_DIST_FULL
        + cfg.ACTION_DIST_FALSE_POS
        + cfg.ACTION_DIST_IDENTITY
    )
    assert abs(dist_sum - 1.0) < 1e-6, f"Action distribution must sum to 1.0, got {dist_sum}"
    assert cfg.NUM_ACTIONS == 6, f"Expected 6 orthogonal controls, got {cfg.NUM_ACTIONS}"
    assert cfg.GENERATOR_BACKBONE == "sd-2.1-base", "Default V5 backbone must be sd-2.1-base!"
    assert "sd-turbo" in cfg.GENERATOR_REPO_MAP and "bk-sdm-small" in cfg.GENERATOR_REPO_MAP
    assert cfg.ADAPTIVE_TIMESTEP is True and 10 <= cfg.RESTORATION_MIN_TIMESTEP < cfg.RESTORATION_MAX_TIMESTEP
    assert cfg.VAE_DEC_LORA_RANK == 0, "VAE decoder must remain frozen to protect SD's natural image manifold!"
    assert cfg.HYPIR_MODE is False and cfg.LOSS_RPGAN_WEIGHT == 0.0 and cfg.LOSS_CSD_WEIGHT > 0.0
    assert len(cfg.LORA_EXPERT_SPECS) == 3

    # Verify HYPIR mode stays cleanly isolated in HYPIRConfig
    hyp_cfg = HYPIRConfig()
    assert hyp_cfg.HYPIR_MODE is True and hyp_cfg.LOSS_CSD_WEIGHT == 0.0 and hyp_cfg.LOSS_RPGAN_WEIGHT > 0.0
    assert hyp_cfg.ADAPTIVE_TIMESTEP is False and hyp_cfg.UNFREEZE_UNET_NORMS is True

    # Verify CLI action parser: 'all' keeps signed artistic channels at 0.0; signed & legacy aliases work
    all_vec = parse_action_spec("all", cfg.DEGRADATION_TYPES, signed_types=cfg.SIGNED_ACTION_TYPES)
    assert np.allclose(all_vec, [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]), f"Unexpected 'all' vector: {all_vec}"

    legacy_vec = parse_action_spec(
        "noise=0.8,jpeg=0.9,motion=0.7,sr=0.6,darken=0.5,cool=0.4",
        cfg.DEGRADATION_TYPES,
        signed_types=cfg.SIGNED_ACTION_TYPES,
    )
    assert np.allclose(legacy_vec, [0.8, 0.9, 0.7, 0.6, -0.5, -0.4]), f"Unexpected alias parse: {legacy_vec}"

    overlay_neg = build_spatial_action_overlay(16, 16, legacy_vec)
    assert float(overlay_neg.min()) == -0.5, "Spatial overlay must preserve negative signed controls!"

    pipe = SelectiveDegradationPipeline(cfg)
    rng = np.random.default_rng(123)

    yy, xx = np.meshgrid(np.linspace(12, 242, 128), np.linspace(12, 242, 128), indexing="ij")
    clean = np.stack([xx, yy, 128.0 + 0.3 * xx - 0.3 * yy], axis=-1).clip(0, 255).astype(np.uint8)
    cv2.circle(clean, (64, 64), 28, (235, 70, 55), -1)
    cv2.rectangle(clean, (20, 20), (46, 102), (45, 210, 195), -1)
    params = pipe.sample_params(128, 128, rng=rng)

    # 1. Exact identity at state = 0
    zero_state = np.zeros(cfg.NUM_ACTIONS, dtype=np.float32)
    ident = pipe.render_with_params(clean, zero_state, params)
    assert np.array_equal(clean, ident), "State=0 must be exact identity!"

    # 2. Seed-synchronized selective removal: curved motion_blur + denoise -> remove noise, keep curved motion blur
    s_src = np.zeros(cfg.NUM_ACTIONS, dtype=np.float32)
    s_src[pipe.type_to_idx["deblur_motion"]] = 0.80
    s_src[pipe.type_to_idx["denoise"]] = 0.70
    x_src = pipe.render_with_params(clean, s_src, params)

    a_keep_blur = parse_action_spec("denoise=1.0,motion=0.0", cfg.DEGRADATION_TYPES, signed_types=cfg.SIGNED_ACTION_TYPES)
    s_tgt = pipe.compute_target_state(s_src, a_keep_blur)
    x_tgt = pipe.render_with_params(clean, s_tgt, params)

    s_only_blur = np.zeros(cfg.NUM_ACTIONS, dtype=np.float32)
    s_only_blur[pipe.type_to_idx["deblur_motion"]] = 0.80
    x_pure_blur = pipe.render_with_params(clean, s_only_blur, params)
    assert np.array_equal(x_tgt, x_pure_blur), "Preserved curved motion blur target must match pure motion blur!"
    assert not np.array_equal(x_src, x_tgt), "Denoised target should differ from noisy source!"

    # 3. Signed directional Exposure & Warmth Lie-group inversion
    for signed_name in ("exposure", "warmth"):
        idx = pipe.type_to_idx[signed_name]
        for sign in (-1.0, 1.0):
            s_tone = np.zeros(cfg.NUM_ACTIONS, dtype=np.float32)
            s_tone[idx] = sign * 0.72
            deg_tone = pipe.render_with_params(clean, s_tone, params)
            assert not np.array_equal(clean, deg_tone)

            act_inv = pipe.compute_full_restore_action(s_tone)
            s_rec = pipe.compute_target_state(s_tone, act_inv)
            rec_tone = pipe.render_with_params(clean, s_rec, params)
            assert np.array_equal(clean, rec_tone), f"{signed_name}: inverse signed slider must reach exact clean target!"

    # 4. Continuity across both optical bokeh disc PSF & Gaussian focus modes + 2D Bezier motion curves
    for use_disc in (True, False):
        params.use_bokeh_disc = use_disc
        for mode_id in (0, 1, 2):
            params.focus_mode = mode_id
            params.artifact_mode = mode_id
            for u_idx in pipe.unsigned_indices:
                deg_name = cfg.DEGRADATION_TYPES[u_idx]
                s_one = np.zeros(cfg.NUM_ACTIONS, dtype=np.float32)
                s_one[u_idx] = 0.85
                d_full = pipe.render_with_params(clean, s_one, params).astype(np.float32)
                d_half = pipe.render_with_params(clean, 0.5 * s_one, params).astype(np.float32)
                d_zero = pipe.render_with_params(clean, 0.0 * s_one, params).astype(np.float32)
                err_full = float(np.mean((d_full - clean.astype(np.float32)) ** 2))
                err_half = float(np.mean((d_half - clean.astype(np.float32)) ** 2))
                err_zero = float(np.mean((d_zero - clean.astype(np.float32)) ** 2))
                assert err_zero == 0.0, f"{deg_name} (disc={use_disc}, mode={mode_id}): a=1.0 must reach clean!"
                assert err_full > err_half > 0.0, f"{deg_name} (disc={use_disc}, mode={mode_id}): must be monotonic!"

    # 5. Compositional Real-Pair Wrapper + GoPro / RealBlur / DPDD Auto-Discovery Verification
    real_blur_sim = cv2.GaussianBlur(clean, (9, 9), 2.4)
    for _ in range(25):
        rx_src, rx_tgt, ra_fg, ra_bg, rmask, rs_src = pipe.synthesize_real_pair_components(
            real_blur_sim, clean, blur_channel_name="deblur_motion", rng=rng
        )
        assert rx_src.shape == clean.shape and rx_tgt.shape == clean.shape
        assert -1.0 <= float(ra_fg.min()) and float(ra_fg.max()) <= 1.0
        for u_idx in pipe.unsigned_indices:
            assert float(ra_fg[u_idx]) >= 0.0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Clean DF2K + symlinked folder
        real_ost = tmp_path / "external_ost" / "outdoor"
        real_ost.mkdir(parents=True)
        train_root = tmp_path / "df2k-ost" / "train"
        (train_root / "DIV2K").mkdir(parents=True)
        (train_root / "Flickr2K").mkdir(parents=True)
        val_root = tmp_path / "df2k-ost" / "test" / "DIV2K_valid"
        val_root.mkdir(parents=True)

        big_clean = cv2.resize(clean, (300, 300), interpolation=cv2.INTER_NEAREST)
        big_blur = cv2.GaussianBlur(big_clean, (11, 11), 3.0)
        cv2.imwrite(str(train_root / "DIV2K" / "0001.png"), big_clean)
        cv2.imwrite(str(train_root / "Flickr2K" / "0002.jpg"), big_clean)
        cv2.imwrite(str(real_ost / "0003.png"), big_clean)
        os.symlink(str(real_ost), str(train_root / "OST_link"))
        cv2.imwrite(str(val_root / "0801.png"), big_clean)

        # Mock Curated Motion Pack (GoPro scene layout + RealBlur .txt list layout) & DPDD Defocus layout
        curated_motion = tmp_path / "curated_deblur"
        (curated_motion / "GoPro" / "train" / "GOPR0372" / "blur").mkdir(parents=True)
        (curated_motion / "GoPro" / "train" / "GOPR0372" / "sharp").mkdir(parents=True)
        cv2.imwrite(str(curated_motion / "GoPro" / "train" / "GOPR0372" / "blur" / "0001.png"), big_blur)
        cv2.imwrite(str(curated_motion / "GoPro" / "train" / "GOPR0372" / "sharp" / "0001.png"), big_clean)

        rb_dir = curated_motion / "RealBlur" / "scene_001"
        (rb_dir / "blur").mkdir(parents=True)
        (rb_dir / "gt").mkdir(parents=True)
        cv2.imwrite(str(rb_dir / "blur" / "blur_1.png"), big_blur)
        cv2.imwrite(str(rb_dir / "gt" / "gt_1.png"), big_clean)
        (curated_motion / "RealBlur_J_train_list.txt").write_text(
            "RealBlur/scene_001/gt/gt_1.png RealBlur/scene_001/blur/blur_1.png\n", encoding="utf-8"
        )

        dpdd_root = tmp_path / "DPDD" / "train"
        (dpdd_root / "inputC").mkdir(parents=True)
        (dpdd_root / "target").mkdir(parents=True)
        cv2.imwrite(str(dpdd_root / "inputC" / "1P0A0001.png"), big_blur)
        cv2.imwrite(str(dpdd_root / "target" / "1P0A0001.png"), big_clean)

        found_motion = discover_paired_blur_datasets(curated_motion, "deblur_motion", is_train=True)
        found_defocus = discover_paired_blur_datasets(dpdd_root, "sharpen_focus", is_train=True)
        assert len(found_motion) == 2, f"Expected 2 GoPro+RealBlur motion pairs, got {len(found_motion)}"
        assert len(found_defocus) == 1 and found_defocus[0][2] == "sharpen_focus"

        ds_cfg = Config()
        ds_cfg.TRAIN_DATASET_PATH = str(train_root)
        ds_cfg.PAIRED_MOTION_DATASET_PATH = str(curated_motion)
        ds_cfg.PAIRED_DEFOCUS_DATASET_PATH = str(dpdd_root)
        ds_cfg.VAL_DATASET_PATH = str(val_root)
        train_ds = create_controllable_dataset(ds_cfg, is_train=True, verbose=False)
        val_ds = create_controllable_dataset(ds_cfg, is_train=False, verbose=False)
        assert len(train_ds.image_paths) == 3 and len(train_ds.paired_records) == 3 and len(val_ds) == 1

        bench_out = tmp_path / "bench_out"
        test_photo = cv2.resize(clean.copy(), (256, 200), interpolation=cv2.INTER_LINEAR)

        def _mock_restore(
            image_rgb_u8,
            action="all",
            bg_action="none",
            point_mask=None,
            mask_2d=None,
            steps=1,
            fidelity_scale=1.0,
            detail_scale=0.0,
        ):
            fg = parse_action_spec(action, ds_cfg.DEGRADATION_TYPES, signed_types=ds_cfg.SIGNED_ACTION_TYPES)
            if np.allclose(fg, 0.0):
                return image_rgb_u8.copy(), None, None
            smoothed = cv2.GaussianBlur(image_rgb_u8, (3, 3), 0.7)
            if point_mask is not None and mask_2d is None:
                mh, mw = image_rgb_u8.shape[:2]
                mask_2d = np.zeros((mh, mw), dtype=np.float32)
                cv2.circle(mask_2d, (mw // 2, mh // 2), min(mh, mw) // 4, 1.0, -1)
            if mask_2d is not None:
                m3 = mask_2d[:, :, None]
                out = np.clip(np.round(m3 * smoothed + (1.0 - m3) * image_rgb_u8), 0, 255).astype(np.uint8)
                return out, mask_2d, None
            return smoothed, None, None

        report = run_single_image_benchmark(
            image_rgb_u8=test_photo,
            config=ds_cfg,
            restore_fn=_mock_restore,
            output_dir=str(bench_out),
            image_stem="demo_photo",
            benchmark_size=128,
            ckpt_stage="pixel_stage",
        )
        assert Path(report["outputs"]["raw_asis_sweep_png"]).exists()
        assert Path(report["outputs"]["synthetic_artifact_sweep_png"]).exists()
        assert Path(report["outputs"]["raw_fullres_restored_png"]).exists()
        assert len(report["synthetic_benchmark_metrics"]) == 4

    print("[PASS] V5 curved Bezier + bokeh disc physics, sharpness gate, GoPro/RealBlur/DPDD loader & benchmark verified.")


def test_model_and_stages_mock():
    import threading
    import torch
    from config import DebugConfig, HYPIRConfig
    from losses.controllable_losses import ControllableRestorationLoss, VisionAidedDiscriminator
    from models.controllable_restorer import ControllableRestorer
    from models.lora_layers import resolve_expert_specs
    from utils.semantic_mask import extract_one_click_semantic_mask

    cfg = DebugConfig()
    cfg.GENERATOR_BACKBONE = "mock_tiny"
    cfg.SEMANTIC_ENCODER_TYPE = "mock_tiny"
    cfg.PATCH_SIZE = 64

    # Verify automatic future-knob expansion in resolve_expert_specs (e.g. adding lens_distortion & deflare)
    expanded_specs = resolve_expert_specs(
        cfg.LORA_EXPERT_SPECS,
        cfg.DEGRADATION_TYPES + ["lens_distortion", "deflare"],
        signed_types=cfg.SIGNED_ACTION_TYPES,
    )
    assert any(sp["name"] == "extra_universal" and len(sp["indices"]) == 2 for sp in expanded_specs)

    model = ControllableRestorer(cfg)
    loss_fn = ControllableRestorationLoss(cfg, use_pretrained_vgg=False)
    disc = VisionAidedDiscriminator(sem_dim=model.semantic_encoder.spatial_dim, num_actions=cfg.NUM_ACTIONS)

    assert len(model.skip_vae.vae_pix_lora_params) > 0 and len(model.skip_vae.vae_sem_lora_params) > 0
    assert all(not p.requires_grad for p in model.skip_vae.vae.decoder.parameters()), (
        "VAE decoder base must have zero trainable LoRAs to prevent pixel-space ripple artifacts!"
    )
    for mod in model.pixel_refiner.modules():
        if isinstance(mod, torch.nn.Conv2d):
            assert tuple(mod.kernel_size) == (1, 1), "PointwiseRadiometricHead must be strictly 1x1!"

    clean = torch.rand(2, 3, 64, 64)
    src = torch.rand(2, 3, 64, 64)
    tgt = torch.rand(2, 3, 64, 64)
    act_map = torch.rand(2, cfg.NUM_ACTIONS, 64, 64)
    act_map[:, 4:, :, :] = act_map[:, 4:, :, :] * 2.0 - 1.0

    # Stage 1: Self-Teacher Align Warmup
    model.set_training_stage("align_warmup")
    loss_s1 = model(src, clean_01=clean, align_only=True)
    loss_s1.backward()
    groups = model.get_stage_parameter_groups()
    assert any(p.grad is not None for p in groups["vision_lora"])
    assert all(p.grad is None for p in groups["pixel_stage"])

    def _backward_on_worker_thread(loss_tensor):
        err_holder = []

        def _worker():
            try:
                loss_tensor.backward()
            except BaseException as exc:
                err_holder.append(exc)

        worker = threading.Thread(target=_worker)
        worker.start()
        worker.join()
        if err_holder:
            raise err_holder[0]

    # Stage 2: Pixel Stage (Slider-Routed UIR-LoRA Experts + Universal eVAE + Dense C-RADIO + 1x1 Tone Head)
    # Test even when one expert's slider (`deblur_motion`) is 0.0 across the whole batch (DDP safety + zero grad conflict!)
    act_zero_motion = act_map.clone()
    act_zero_motion[:, 2, :, :] = 0.0
    model.zero_grad(set_to_none=True)
    model.set_training_stage("pixel_stage")
    out_s2 = model(src, act_zero_motion, tgt_01=tgt, clean_01=clean, fidelity_scale=1.0, detail_scale=0.0)
    loss_s2, metrics_s2 = loss_fn(
        out_s2["pred"],
        tgt,
        act_zero_motion,
        out_s2["align_loss"],
        stage="pixel_stage",
        t_vec=out_s2["t"],
        src_01=src,
        hat_z_0=out_s2["hat_z_0"],
        z_tgt=out_s2["z_tgt"],
        evae_loss=out_s2["evae_loss"],
        restorer_model=model,
        semantic_encoder=model.semantic_encoder,
    )
    assert metrics_s2["evae"] >= 0.0 and metrics_s2["lat"] > 0.0 and metrics_s2["pyr"] > 0.0
    _backward_on_worker_thread(loss_s2)
    model.clear_runtime_contexts()
    missing_pix_grads = [
        n for n, p in model.named_parameters()
        if any(p is gp for gp in groups["pixel_stage"]) and p.grad is None
    ]
    assert not missing_pix_grads, f"Unused pixel_stage params (breaks DDP!): {missing_pix_grads}"
    # Confirm motion_deconv expert received exact 0.0 gradient magnitude when deblur_motion == 0.0!
    for name, param in model.named_parameters():
        if "pix_experts.motion_deconv" in name:
            assert float(param.grad.abs().max().item()) == 0.0, f"Inactive expert {name} leaked gradient!"
    assert all(p.grad is None for p in groups["semantic_stage"])

    # Stage 3: Semantic-Detail Stage with Zero-Extra-VRAM Target-Anchored CSD
    model.zero_grad(set_to_none=True)
    model.set_training_stage("semantic_stage")
    out_s3 = model(src, act_map, tgt_01=tgt, clean_01=clean, fidelity_scale=1.0, detail_scale=0.8)
    loss_s3, metrics_s3, aux_s3 = loss_fn(
        out_s3["pred"],
        tgt,
        act_map,
        out_s3["align_loss"],
        stage="semantic_stage",
        t_vec=out_s3["t"],
        src_01=src,
        hat_z_0=out_s3["hat_z_0"],
        z_tgt=out_s3["z_tgt"],
        evae_loss=out_s3["evae_loss"],
        restorer_model=model,
        semantic_encoder=model.semantic_encoder,
        return_aux=True,
    )
    assert metrics_s3["csd"] >= 0.0
    _backward_on_worker_thread(loss_s3)
    model.clear_runtime_contexts()
    missing_sem_grads = [
        n for n, p in model.named_parameters()
        if any(p is gp for gp in groups["semantic_stage"]) and p.grad is None
    ]
    assert not missing_sem_grads, f"Unused semantic_stage params (breaks DDP!): {missing_sem_grads}"
    assert all(p.grad is None for p in groups["pixel_stage"])

    # Isolated HYPIR Mode Relativistic RpGAN Check
    cfg.HYPIR_MODE = True
    cfg.LOSS_CSD_WEIGHT = 0.0
    cfg.LOSS_RPGAN_WEIGHT = 0.08
    disc.eval()
    disc.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    out_hyp = model(src, act_map, tgt_01=tgt, clean_01=clean, fidelity_scale=1.0, detail_scale=0.8)
    loss_hyp, metrics_hyp, aux_hyp = loss_fn(
        out_hyp["pred"],
        tgt,
        act_map,
        out_hyp["align_loss"],
        stage="semantic_stage",
        t_vec=out_hyp["t"],
        src_01=src,
        hat_z_0=out_hyp["hat_z_0"],
        z_tgt=out_hyp["z_tgt"],
        evae_loss=out_hyp["evae_loss"],
        restorer_model=model,
        semantic_encoder=model.semantic_encoder,
        discriminator=disc,
        return_aux=True,
    )
    assert metrics_hyp["csd"] == 0.0 and aux_hyp.get("use_rpgan") is True
    loss_hyp.backward()
    model.clear_runtime_contexts()

    disc.requires_grad_(True)
    disc.train()
    disc.zero_grad(set_to_none=True)
    d_loss = VisionAidedDiscriminator.discriminator_hinge_loss(
        disc, tgt, out_hyp["pred"].detach(), aux_hyp["tgt_grid"], aux_hyp["pred_grid"], act_map, relativistic=True
    )
    d_loss.backward()
    assert all(p.grad is not None for p in disc.parameters())

    # Signed Negative Slider Unlock & Zero-Action Exact Preservation Check
    cfg.HYPIR_MODE = False
    model.eval()
    zero_act = torch.zeros(1, cfg.NUM_ACTIONS, 64, 64)
    ident_out = model.sample(src[:1], zero_act, steps=1)
    assert torch.allclose(ident_out, src[:1], atol=1e-7), "Zero action must preserve input bit-for-bit!"

    neg_act = torch.zeros(1, cfg.NUM_ACTIONS, 64, 64)
    neg_act[:, 4, :, :32] = -0.80
    with torch.no_grad():
        model.pixel_refiner.tone_head[-1].bias.fill_(0.25)
    neg_out = model.sample(src[:1], neg_act, steps=1, fidelity_scale=1.0, detail_scale=0.8)
    assert not torch.allclose(neg_out[:, :, :, :32], src[:1, :, :, :32], atol=1e-4)
    assert torch.allclose(neg_out[:, :, :, 32:], src[:1, :, :, 32:], atol=1e-7)

    big_src = torch.rand(1, 3, 100, 110)
    big_act = torch.zeros(1, cfg.NUM_ACTIONS, 100, 110)
    big_act[:, 5, :, :50] = -0.75
    tiled_out, tiled_traj = model.sample(
        big_src, big_act, steps=2, tile_size=64, tile_overlap=16, return_trajectory=True
    )
    assert tiled_out.shape == (1, 3, 100, 110) and len(tiled_traj) == 3
    assert torch.allclose(tiled_out[:, :, :, 50:], big_src[:, :, :, 50:], atol=1e-6)

    mask_2d = extract_one_click_semantic_mask(model.semantic_encoder, src[:1], point_xy=(0.5, 0.5))
    assert mask_2d.shape == (64, 64) and 0.0 <= float(mask_2d.min()) and float(mask_2d.max()) <= 1.0
    print("[PASS] V5 Slider-Routed UIR-LoRA Experts, Universal eVAE, Target-Anchored CSD & isolated HYPIR verified.")


if __name__ == "__main__":
    test_syntax_all_files()
    try:
        import numpy  # noqa: F401
        import cv2    # noqa: F401

        test_selective_pipeline()
    except ImportError as exc:
        print(f"[SKIP] NumPy/OpenCV check skipped in current env ({exc}).")

    try:
        import torch  # noqa: F401

        test_model_and_stages_mock()
    except ImportError as exc:
        print(f"[SKIP] PyTorch runtime check skipped in current env ({exc}).")
