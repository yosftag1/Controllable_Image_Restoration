"""End-to-end verification suite for Controllable_Image_Restoration (V7 SD-Turbo Crisp 7-Control).

Runs:
1. AST syntax check across every Python file in the repository.
2. Config invariants (`sd-turbo` default, `RESTORATION_TIMESTEPS=(499,)`, 7 core controls,
   3 routed specialist LoRA experts, `VAE_ENC_LORA_RANK=0`, `VAE_DEC_LORA_RANK=8`,
   `REUSE_PAIRED_CLEAN_FOR_SYNTH=False`, `PATCH_SIZE=384`, `GRAD_ACCUM_STEPS=2`),
   artifact-first + dataset-balanced sampling, monotonic Real-ESRGAN SR & 3-mode ISP noise,
   Lie-group Exposure/Warmth, strict `gan_clean_eligible` hygiene, and curated paired loader.
3. Full 1-step / 2-stage forward/backward (`pixel_stage` & `semantic_stage`), Routed LoRA-MoE +
   Stage-2 VAE Decoder LoRA DDP gradient completeness & zero-leakage on pure `denoise`,
   zero-VRAM `DINOv2FeatureDiscriminator` G/D hinge gating, zero-action identity, tiled inference,
   and 1-click semantic mask extraction.
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
    from config import Config, DebugConfig, PilotConfig
    from data.selective_dataset import (
        SelectiveDegradationPipeline,
        create_controllable_dataset,
        discover_paired_blur_datasets,
    )
    from utils.image_benchmark import run_single_image_benchmark
    from utils.semantic_mask import build_spatial_action_overlay, parse_action_spec

    cfg = Config()
    dist_sum = (
        cfg.ACTION_DIST_FULL
        + cfg.ACTION_DIST_BLIND_ALL
        + cfg.ACTION_DIST_SELECTIVE
        + cfg.ACTION_DIST_PARTIAL
        + cfg.ACTION_DIST_FALSE_POS
        + cfg.ACTION_DIST_IDENTITY
    )
    assert abs(dist_sum - 1.0) < 1e-6, f"Action distribution must sum to 1.0, got {dist_sum}"
    assert cfg.NUM_ACTIONS == 7, f"Expected 7 streamlined controls, got {cfg.NUM_ACTIONS}"
    assert "super_resolution" in cfg.DEGRADATION_TYPES and "sharpen_focus" not in cfg.DEGRADATION_TYPES
    assert cfg.SEMANTIC_ENCODER_TYPE == "dinov2_reg", "Default vision encoder must be dinov2_reg!"
    assert cfg.GENERATOR_BACKBONE == "sd-turbo", "Default backbone must be sd-turbo!"
    assert cfg.GENERATOR_REPO_MAP["sd-turbo"] == "stabilityai/sd-turbo"
    assert tuple(cfg.RESTORATION_TIMESTEPS) == (499,)
    assert len(cfg.LORA_EXPERT_SPECS) == 3
    assert cfg.VAE_ENC_LORA_RANK == 0 and cfg.VAE_DEC_LORA_RANK == 8
    assert cfg.REUSE_PAIRED_CLEAN_FOR_SYNTH is False
    assert cfg.PATCH_SIZE == 384 and cfg.GRAD_ACCUM_STEPS == 2
    assert "denoise" not in cfg.DETAIL_ACTION_TYPES and "super_resolution" in cfg.DETAIL_ACTION_TYPES
    assert "deblur_motion" in cfg.DESTRUCTIVE_SKIP_ACTIONS and "super_resolution" in cfg.DESTRUCTIVE_SKIP_ACTIONS
    assert cfg.LOSS_PIXEL_STAGE_PERCEPTUAL_WEIGHT >= 0.30 and cfg.LOSS_PIXEL_STAGE_EDGE_WEIGHT > 0.0
    assert cfg.LOSS_EDGE_GRAD_WEIGHT > 0.0 and cfg.LOSS_GAN_WEIGHT > 0.0

    pil_cfg = PilotConfig()
    assert pil_cfg.EPOCHS == 12 and pil_cfg.PIXEL_STAGE_EPOCHS == 8
    assert pil_cfg.MAX_STEPS_PER_EPOCH == 700 and pil_cfg.MIN_STEPS_PER_EPOCH == 700

    dbg_cfg = DebugConfig()
    assert dbg_cfg.EPOCHS == 2 and dbg_cfg.PIXEL_STAGE_EPOCHS == 1

    all_vec = parse_action_spec("all", cfg.DEGRADATION_TYPES, signed_types=cfg.SIGNED_ACTION_TYPES)
    assert np.allclose(all_vec, [1.0] * 5 + [0.0, 0.0]), f"Unexpected 'all' vector: {all_vec}"

    alias_vec = parse_action_spec(
        "noise=0.8,jpeg=0.9,motion=0.7,sr=0.6,haze=0.4,darken=0.5,cool=0.3",
        cfg.DEGRADATION_TYPES,
        signed_types=cfg.SIGNED_ACTION_TYPES,
    )
    assert np.allclose(alias_vec, [0.8, 0.9, 0.7, 0.6, 0.4, -0.5, -0.3]), f"Unexpected alias parse: {alias_vec}"

    overlay_neg = build_spatial_action_overlay(16, 16, alias_vec)
    assert float(overlay_neg.min()) == -0.5, "Spatial overlay must preserve negative signed controls!"

    pipe = SelectiveDegradationPipeline(cfg)
    rng = np.random.default_rng(123)

    yy, xx = np.meshgrid(np.linspace(12, 242, 128), np.linspace(12, 242, 128), indexing="ij")
    clean = np.stack([xx, yy, 128.0 + 0.3 * xx - 0.3 * yy], axis=-1).clip(0, 255).astype(np.uint8)
    cv2.circle(clean, (64, 64), 28, (235, 70, 55), -1)
    cv2.rectangle(clean, (20, 20), (46, 102), (45, 210, 195), -1)
    # Add crisp high-frequency checkerboard strip so SR/blur monotonicity is strongly measurable
    clean[80:112, 80:112] = (((np.arange(32)[:, None] + np.arange(32)[None, :]) % 2) * 180 + 35)[:, :, None].astype(np.uint8)
    params = pipe.sample_params(128, 128, rng=rng)

    # 1. Exact identity at state = 0 & custom motion blur helper
    zero_state = np.zeros(cfg.NUM_ACTIONS, dtype=np.float32)
    ident = pipe.render_with_params(clean, zero_state, params)
    assert np.array_equal(clean, ident), "State=0 must be exact identity!"

    custom_id = pipe.render_custom_motion_blur(clean, length_px=0.0, angle_deg=35.0)
    assert np.array_equal(clean, custom_id), "Custom motion blur with length=0 must be identity!"
    custom_blur = pipe.render_custom_motion_blur(
        clean, length_px=15.0, angle_deg=35.0, curve_bend=0.25, shake_jitter=0.30, pattern=1
    )
    assert custom_blur.shape == clean.shape and not np.array_equal(clean, custom_blur)

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
    assert np.array_equal(x_tgt, x_pure_blur), "Preserved motion blur target must match pure motion blur!"
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

    # 4. Monotonicity across ALL 5 unsigned controls (incl. all 3 SR modes & all 3 ISP noise modes!)
    params.motion_shake_jitter = 0.28
    params.sr_scale_factor = 0.35
    params.sr_sigma_max = 1.6
    for mode_id in (0, 1, 2, 3):
        params.sr_mode = mode_id % 3
        params.noise_mode = mode_id % 3
        params._noise_cache = None
        params.heteroscedastic_noise = bool(mode_id % 2 == 0)
        params.motion_pattern = mode_id % 3
        params.motion_spatial_mode = mode_id % 4
        params.motion_hdr_boost = 0.30 if (mode_id % 2 == 0) else 0.0
        params._motion_mask_cache = None
        params.artifact_mode = mode_id % 4
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
            assert err_zero == 0.0, f"{deg_name} (mode={mode_id}): a=1.0 must reach clean!"
            assert err_full > err_half > 0.0, f"{deg_name} (mode={mode_id}): must be monotonic ({err_full} vs {err_half})!"

    # Verify Brown-Conrady lens distortion helper
    assert np.array_equal(clean, pipe.render_lens_distortion(clean, k1=0.0, k2=0.0))
    assert not np.array_equal(clean, pipe.render_lens_distortion(clean, k1=0.25))

    # 5. Stage-2 Detail Focus, Artifact-First Budget, Blind 'all' Coverage & GAN-Clean Hygiene
    pipe.detail_stage_focus = True
    det_hits = 0
    gan_clean_hits = 0
    for _ in range(40):
        _, x_t, a_fg, a_bg, _, _, gan_ok = pipe.synthesize_components(clean, rng=rng)
        if any(float(a_fg[i]) >= 0.25 for i in pipe.detail_indices):
            det_hits += 1
        if gan_ok:
            gan_clean_hits += 1
            assert np.array_equal(x_t, clean), "gan_clean_eligible=True MUST correspond to 100% clean GT target!"
    assert det_hits >= 34, f"Expected >=85% active detail actions in Stage 2, got {det_hits}/40"
    assert gan_clean_hits >= 20, f"Expected strong Stage-2 GAN-clean target ratio, got {gan_clean_hits}/40"
    pipe.detail_stage_focus = False

    # Verify Real-Pair Wrapper NEVER marks 720p pairs as gan_clean_eligible & covers blind 'all' preset!
    real_blur_sim = cv2.GaussianBlur(clean, (9, 9), 2.4)
    all_hits = 0
    for _ in range(80):
        rx_src, rx_tgt, ra_fg, ra_bg, rmask, rs_src, r_gan_ok = pipe.synthesize_real_pair_components(
            real_blur_sim, clean, blur_channel_name="deblur_motion", rng=rng
        )
        assert rx_src.shape == clean.shape and rx_tgt.shape == clean.shape
        assert r_gan_ok is False, "Real 720p camera pairs must NEVER be marked gan_clean_eligible!"
        if all(float(ra_fg[i]) >= 0.99 for i in pipe.unsigned_indices):
            all_hits += 1
    assert all_hits >= 4, f"Expected explicit blind 'all' preset coverage on real pairs, got {all_hits}/80"

    # 6. Clean dataset bucket isolation (REUSE_PAIRED_CLEAN_FOR_SYNTH=False) & per-subdataset paired balancing
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_root = tmp_path / "df2k-ost" / "train"
        (train_root / "DIV2K").mkdir(parents=True)
        (train_root / "Flickr2K").mkdir(parents=True)
        (train_root / "OST").mkdir(parents=True)
        val_root = tmp_path / "df2k-ost" / "test" / "DIV2K_valid"
        val_root.mkdir(parents=True)

        big_clean = cv2.resize(clean, (420, 420), interpolation=cv2.INTER_NEAREST)
        big_blur = cv2.GaussianBlur(big_clean, (13, 13), 3.4)
        cv2.imwrite(str(train_root / "DIV2K" / "0001.png"), big_clean)
        cv2.imwrite(str(train_root / "Flickr2K" / "0002.jpg"), big_clean)
        for oi in range(5):
            cv2.imwrite(str(train_root / "OST" / f"ost_{oi:02d}.png"), big_clean)
        cv2.imwrite(str(val_root / "0801.png"), big_clean)

        curated_motion = tmp_path / "curated_deblur"
        (curated_motion / "GoPro" / "train" / "GOPR0372" / "blur").mkdir(parents=True)
        (curated_motion / "GoPro" / "train" / "GOPR0372" / "sharp").mkdir(parents=True)
        cv2.imwrite(str(curated_motion / "GoPro" / "train" / "GOPR0372" / "blur" / "0001.png"), big_blur)
        cv2.imwrite(str(curated_motion / "GoPro" / "train" / "GOPR0372" / "sharp" / "0001.png"), big_clean)

        (curated_motion / "HIDE" / "test" / "blur").mkdir(parents=True)
        (curated_motion / "HIDE" / "test" / "sharp").mkdir(parents=True)
        cv2.imwrite(str(curated_motion / "HIDE" / "test" / "blur" / "hide_01.png"), big_blur)
        cv2.imwrite(str(curated_motion / "HIDE" / "test" / "sharp" / "hide_01.png"), big_clean)

        ds_cfg = Config()
        ds_cfg.PATCH_SIZE = 128
        ds_cfg.TRAIN_DATASET_PATH = str(train_root)
        ds_cfg.PAIRED_MOTION_DATASET_PATH = str(curated_motion)
        ds_cfg.VAL_DATASET_PATH = str(val_root)
        train_ds = create_controllable_dataset(ds_cfg, is_train=True, verbose=False)
        val_ds = create_controllable_dataset(ds_cfg, is_train=False, verbose=False)
        assert len(train_ds.paired_records) == 2 and len(val_ds) == 1
        assert len(train_ds.image_paths) == 7, f"Paired GoPro/HIDE sharp frames must NOT leak into clean pool! Got {len(train_ds.image_paths)}"
        assert set(train_ds.clean_buckets.keys()) == {"div2k", "flickr2k", "ost"}
        assert set(train_ds.paired_buckets.keys()) == {"gopro", "hide"}
        # DIV2K must have higher sampling probability than OST despite 5x fewer OST images!
        div_i = train_ds._clean_bucket_names.index("div2k")
        ost_i = train_ds._clean_bucket_names.index("ost")
        assert train_ds._clean_bucket_probs[div_i] > train_ds._clean_bucket_probs[ost_i]

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
        assert len(report["synthetic_benchmark_metrics"]) == 4

    print("[PASS] Artifact-first budget, dataset-balanced buckets, Real-ESRGAN SR, ISP noise & GAN-clean hygiene verified.")


def test_model_and_stages_mock():
    import threading
    import torch
    from config import DebugConfig
    from losses.controllable_losses import ControllableRestorationLoss, DINOv2FeatureDiscriminator
    from models.controllable_restorer import ControllableRestorer
    from utils.semantic_mask import extract_one_click_semantic_mask

    cfg = DebugConfig()
    cfg.GENERATOR_BACKBONE = "mock_tiny"
    cfg.SEMANTIC_ENCODER_TYPE = "mock_tiny"
    cfg.PATCH_SIZE = 64

    model = ControllableRestorer(cfg)
    loss_fn = ControllableRestorationLoss(cfg, use_pretrained_vgg=False)

    assert len(model.skip_vae.vae_pix_lora_params) == 0, "Blind VAE encoder LoRA must be disabled by default!"
    assert len(model.skip_vae.vae_sem_lora_params) > 0, "Stage-2 detail-gated VAE Decoder LoRA must be active!"
    assert all(not p.requires_grad for p in model.semantic_encoder.parameters()), (
        "Semantic encoder must remain frozen!"
    )
    for mod in model.pixel_refiner.modules():
        if isinstance(mod, torch.nn.Conv2d):
            assert tuple(mod.kernel_size) == (1, 1), "PointwiseRadiometricHead must be strictly 1x1!"

    src = torch.rand(2, 3, 64, 64)
    tgt = torch.rand(2, 3, 64, 64)
    act_map = torch.rand(2, cfg.NUM_ACTIONS, 64, 64)
    act_map[:, 5:, :, :] = act_map[:, 5:, :, :] * 2.0 - 1.0

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

    groups = model.get_stage_parameter_groups()

    # Stage 1: Pixel Stage (1-Step SD-Turbo + 3 Specialist LoRA Experts + Un-blurred VGG + Edge Grad)
    act_zero_motion = act_map.clone()
    act_zero_motion[:, 2, :, :] = 0.0  # motion=0 -> verifies DDP still receives zero-grad tensors for motion_deconv!
    model.zero_grad(set_to_none=True)
    model.set_training_stage("pixel_stage")
    out_s1 = model(src, act_zero_motion, tgt_01=tgt, fidelity_scale=1.0, detail_scale=0.0)
    assert "hat_z_1" in out_s1 and out_s1["hat_z_1"] is out_s1["hat_z_0"]
    loss_s1, metrics_s1 = loss_fn(
        out_s1["pred"],
        tgt,
        act_zero_motion,
        stage="pixel_stage",
        src_01=src,
        hat_z_0=out_s1["hat_z_0"],
        hat_z_1=out_s1["hat_z_1"],
        z_tgt=out_s1["z_tgt"],
        restorer_model=model,
    )
    assert metrics_s1["l1"] > 0.0 and metrics_s1["lat"] > 0.0 and metrics_s1["edge"] > 0.0 and metrics_s1["csd"] == 0.0
    _backward_on_worker_thread(loss_s1)
    model.clear_runtime_contexts()
    missing_pix_grads = [
        n for n, p in model.named_parameters()
        if any(p is gp for gp in groups["pixel_stage"]) and p.grad is None
    ]
    assert not missing_pix_grads, f"Unused pixel_stage params (breaks DDP!): {missing_pix_grads}"
    assert all(p.grad is None for p in groups["semantic_stage"])

    # Stage 2: Semantic-Detail Stage (Fixed d=1.0, UNet + VAE Decoder sem_branch, True DISTS + Edge Grad)
    model.zero_grad(set_to_none=True)
    model.set_training_stage("semantic_stage")
    out_s2 = model(src, act_map, tgt_01=tgt, fidelity_scale=1.0, detail_scale=1.0)
    loss_s2, metrics_s2 = loss_fn(
        out_s2["pred"],
        tgt,
        act_map,
        stage="semantic_stage",
        src_01=src,
        hat_z_0=out_s2["hat_z_0"],
        hat_z_1=out_s2["hat_z_1"],
        z_tgt=out_s2["z_tgt"],
        restorer_model=model,
        run_csd=False,
    )
    assert metrics_s2["perc"] > 0.0 and metrics_s2["edge"] > 0.0
    _backward_on_worker_thread(loss_s2)
    model.clear_runtime_contexts()
    missing_sem_grads = [
        n for n, p in model.named_parameters()
        if any(p is gp for gp in groups["semantic_stage"]) and p.grad is None
    ]
    assert not missing_sem_grads, f"Unused semantic_stage params (breaks DDP!): {missing_sem_grads}"
    assert all(p.grad is None for p in groups["pixel_stage"])

    # Verify BOTH UNet & VAE Decoder `sem_branch` zero-leakage on pure `denoise`!
    act_denoise_only = torch.zeros_like(act_map)
    act_denoise_only[:, 0, :, :] = 1.0  # denoise only
    model.zero_grad(set_to_none=True)
    model.set_training_stage("semantic_stage")
    out_dn = model(src, act_denoise_only, tgt_01=tgt, fidelity_scale=1.0, detail_scale=1.0)
    loss_dn, _ = loss_fn(
        out_dn["pred"],
        tgt,
        act_denoise_only,
        stage="semantic_stage",
        src_01=src,
        hat_z_0=out_dn["hat_z_0"],
        z_tgt=out_dn["z_tgt"],
        restorer_model=model,
        run_csd=False,
    )
    _backward_on_worker_thread(loss_dn)
    model.clear_runtime_contexts()
    for n, p in model.named_parameters():
        if any(p is gp for gp in groups["semantic_stage"]):
            assert p.grad is not None, f"DDP requires zero grad tensor (not None) on {n}!"
            assert float(p.grad.abs().max().item()) == 0.0, f"sem_branch leaked gradient on pure denoise ({n})!"

    # Inference invariance: even with non-zero UNet & VAE-Dec sem_branch weights, pure denoise is identical at d=0 vs d=1
    model.eval()
    with torch.no_grad():
        for p in groups["semantic_stage"]:
            p.fill_(0.05)
        dn_d0 = model.sample(src[:1], act_denoise_only[:1], steps=1, fidelity_scale=1.0, detail_scale=0.0)
        dn_d1 = model.sample(src[:1], act_denoise_only[:1], steps=1, fidelity_scale=1.0, detail_scale=1.0)
        assert torch.allclose(dn_d0, dn_d1, atol=1e-6), "Pure denoise must be 100% invariant to detail_scale!"
        for p in groups["semantic_stage"]:
            p.zero_()
    model.train()

    # Verify Frozen-DINOv2 Multi-Layer Feature Discriminator (zero grad when clean_mask=0 or detail_mask=0!)
    disc = DINOv2FeatureDiscriminator(feat_dim=model.semantic_encoder.spatial_dim, num_scales=3, hidden_dim=32)
    fake_req = torch.rand(2, 3, 64, 64, requires_grad=True)
    one_det = torch.ones(2, 1, 64, 64)
    zero_clean = torch.zeros(2)
    g_zero = disc.compute_g_loss(model.semantic_encoder, fake_req, detail_mask=one_det, clean_sample_mask=zero_clean)
    g_zero.backward()
    assert float(fake_req.grad.abs().max().item()) == 0.0, "Non-clean target must produce zero DINOv2-Disc G gradient!"

    fake_req2 = torch.rand(2, 3, 64, 64, requires_grad=True)
    one_clean = torch.ones(2)
    g_act = disc.compute_g_loss(model.semantic_encoder, fake_req2, detail_mask=one_det, clean_sample_mask=one_clean)
    g_act.backward()
    assert float(fake_req2.grad.abs().max().item()) > 0.0, "Clean active detail target must produce non-zero G gradient!"

    disc.zero_grad(set_to_none=True)
    d_loss, _, _ = DINOv2FeatureDiscriminator.compute_d_loss_with_module(
        disc, model.semantic_encoder, tgt, fake_req2.detach(), detail_mask=one_det, clean_sample_mask=one_clean
    )
    d_loss.backward()
    assert all(p.grad is not None for p in disc.parameters()), "All DINOv2-Disc head params must receive grads!"

    # Signed Negative Slider & Zero-Action Exact Preservation Check
    model.eval()
    zero_act = torch.zeros(1, cfg.NUM_ACTIONS, 64, 64)
    ident_out = model.sample(src[:1], zero_act, steps=1)
    assert torch.allclose(ident_out, src[:1], atol=1e-7), "Zero action must preserve input bit-for-bit!"

    neg_act = torch.zeros(1, cfg.NUM_ACTIONS, 64, 64)
    neg_act[:, 5, :, :32] = -0.80  # exposure index = 5
    with torch.no_grad():
        model.pixel_refiner.tone_head[-1].bias.fill_(0.25)
    neg_out = model.sample(src[:1], neg_act, steps=1, fidelity_scale=1.0, detail_scale=0.65)
    assert not torch.allclose(neg_out[:, :, :, :32], src[:1, :, :, :32], atol=1e-4)
    assert torch.allclose(neg_out[:, :, :, 32:], src[:1, :, :, 32:], atol=1e-7)

    big_src = torch.rand(1, 3, 100, 110)
    big_act = torch.zeros(1, cfg.NUM_ACTIONS, 100, 110)
    big_act[:, 6, :, :50] = -0.75  # warmth index = 6
    tiled_out, tiled_traj = model.sample(
        big_src, big_act, steps=2, tile_size=64, tile_overlap=16, return_trajectory=True
    )
    assert tiled_out.shape == (1, 3, 100, 110) and len(tiled_traj) == 3
    assert torch.allclose(tiled_out[:, :, :, 50:], big_src[:, :, :, 50:], atol=1e-6)

    mask_2d = extract_one_click_semantic_mask(model.semantic_encoder, src[:1], point_xy=(0.5, 0.5))
    assert mask_2d.shape == (64, 64) and 0.0 <= float(mask_2d.min()) and float(mask_2d.max()) <= 1.0
    print("[PASS] 1-Step SD-Turbo, Routed LoRA-MoE, VAE-Dec LoRA, DINOv2-Disc & DDP gradients verified.")


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
