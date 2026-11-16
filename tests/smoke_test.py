"""End-to-end verification suite for Controllable_Image_Restoration (V4 Orthogonal + Directional).

Runs:
1. AST syntax check across every Python file in the repository.
2. 6-channel orthogonal + signed Lie-group degradation synthesis, directional Exposure/Warmth
   group inversion, legacy CLI alias compatibility, and benchmark contact-sheet verification.
3. Full 3-phase forward/backward, Self-Teacher LoRA switch, Dual-LoRA + PixelSpaceRefiner
   isolation, signed negative slider Zero-Action unlock, Spatial Overlay exact preservation,
   and 1-Click Semantic Mask tests (using offline mock_tiny backbones if PyTorch is installed).
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
    import numpy as np
    from config import Config
    from data.selective_dataset import SelectiveDegradationPipeline
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
    assert cfg.RESTORATION_MAX_TIMESTEP == 249 and cfg.SKIP_DROPOUT_PROB == 0.0
    assert cfg.PIXEL_STAGE_EPOCHS >= 10 and cfg.LOSS_L1_WEIGHT >= 0.8

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
    params = pipe.sample_params(128, 128, rng=rng)

    # 1. Exact identity at state = 0
    zero_state = np.zeros(cfg.NUM_ACTIONS, dtype=np.float32)
    ident = pipe.render_with_params(clean, zero_state, params)
    assert np.array_equal(clean, ident), "State=0 must be exact identity!"

    # 2. Seed-synchronized selective removal: motion_blur + denoise -> remove noise, keep motion blur
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
    assert np.array_equal(x_tgt, x_pure_blur), "Preserved motion blur target must match pure motion blur exactly!"
    assert not np.array_equal(x_src, x_tgt), "Denoised target should differ from noisy source!"

    # 3. Signed directional Exposure & Warmth Lie-group inversion (both negative & positive states)
    for signed_name in ("exposure", "warmth"):
        idx = pipe.type_to_idx[signed_name]
        for sign in (-1.0, 1.0):
            s_tone = np.zeros(cfg.NUM_ACTIONS, dtype=np.float32)
            s_tone[idx] = sign * 0.72
            deg_tone = pipe.render_with_params(clean, s_tone, params)
            assert not np.array_equal(clean, deg_tone), f"{signed_name} ({sign}) should visibly shift image!"

            act_inv = pipe.compute_full_restore_action(s_tone)
            assert abs(float(act_inv[idx]) + sign * 0.72) < 1e-6
            s_rec = pipe.compute_target_state(s_tone, act_inv)
            rec_tone = pipe.render_with_params(clean, s_rec, params)
            assert np.array_equal(clean, rec_tone), f"{signed_name}: inverse signed slider must reach exact clean target!"

            s_half = pipe.compute_target_state(s_tone, pipe.compute_partial_restore_action(s_tone, 0.5))
            half_tone = pipe.render_with_params(clean, s_half, params).astype(np.float32)
            err_full = float(np.mean((deg_tone.astype(np.float32) - clean.astype(np.float32)) ** 2))
            err_half = float(np.mean((half_tone - clean.astype(np.float32)) ** 2))
            assert err_full > err_half > 0.0, f"{signed_name} ({sign}): 50% partial recovery must be strictly monotonic!"

    # 4. Synthesize pair bounds & identity check across 30 random trials (covering signed & artistic modes)
    for _ in range(30):
        src_u8, act_map, tgt_u8, _ = pipe.synthesize_pair(clean, rng=rng)
        assert src_u8.shape == clean.shape and tgt_u8.shape == clean.shape
        assert act_map.shape == (cfg.NUM_ACTIONS, 128, 128)
        assert -1.0 <= float(act_map.min()) and float(act_map.max()) <= 1.0
        for u_idx in pipe.unsigned_indices:
            assert float(act_map[u_idx].min()) >= 0.0, "Unsigned restoration channels must stay >= 0!"
        if float(np.max(np.abs(act_map))) == 0.0:
            assert np.array_equal(src_u8, tgt_u8), "All-zero action map must yield exact x_src == x_tgt!"

    # 5. False-positive safety on objective sliders: setting [0..3]=1.0 on clean attributes is exact no-op
    s_only_noise = np.zeros(cfg.NUM_ACTIONS, dtype=np.float32)
    s_only_noise[pipe.type_to_idx["denoise"]] = 0.65
    x_noisy = pipe.render_with_params(clean, s_only_noise, params)
    a_fp = parse_action_spec(
        "motion=1.0,focus=1.0,deartifact=1.0,denoise=0.0",
        cfg.DEGRADATION_TYPES,
        signed_types=cfg.SIGNED_ACTION_TYPES,
    )
    x_fp_tgt = pipe.render_with_params(clean, pipe.compute_target_state(s_only_noise, a_fp), params)
    assert np.array_equal(x_noisy, x_fp_tgt), "False-positive objective sliders on clean attributes must be no-ops!"

    # 6. Continuity across all 4 unsigned channels across all focus/artifact sub-modes
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
            assert err_zero == 0.0, f"{deg_name} (mode={mode_id}): a=1.0 must reach exact clean image!"
            assert err_full > err_half > 0.0, f"{deg_name} (mode={mode_id}): partial a=0.5 must be monotonic!"

    # 7. Kaggle DF2K-OST nested subfolder + symlink dataset discovery & benchmark suite verification
    import cv2
    import tempfile
    from data.selective_dataset import create_controllable_dataset
    from utils.image_benchmark import run_single_image_benchmark

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        real_ost = tmp_path / "external_ost" / "outdoor"
        real_ost.mkdir(parents=True)
        train_root = tmp_path / "df2k-ost" / "train"
        (train_root / "DIV2K").mkdir(parents=True)
        (train_root / "Flickr2K").mkdir(parents=True)
        val_root = tmp_path / "df2k-ost" / "test" / "DIV2K_valid"
        val_root.mkdir(parents=True)

        cv2.imwrite(str(train_root / "DIV2K" / "0001.png"), clean)
        cv2.imwrite(str(train_root / "Flickr2K" / "0002.jpg"), clean)
        cv2.imwrite(str(real_ost / "0003.png"), clean)
        os.symlink(str(real_ost), str(train_root / "OST_link"))
        cv2.imwrite(str(val_root / "0801.png"), clean)

        ds_cfg = Config()
        ds_cfg.TRAIN_DATASET_PATH = str(train_root)
        ds_cfg.VAL_DATASET_PATH = str(val_root)
        train_ds = create_controllable_dataset(ds_cfg, is_train=True)
        val_ds = create_controllable_dataset(ds_cfg, is_train=False)
        assert len(train_ds) == 3 and len(val_ds) == 1

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

    print("[PASS] 6-channel orthogonal + directional synthesis, Lie-group tone/WB & benchmark suite verified.")


def test_model_and_stages_mock():
    import threading
    import torch
    from config import DebugConfig
    from losses.controllable_losses import ControllableRestorationLoss, VisionAidedDiscriminator
    from models.controllable_restorer import ControllableRestorer
    from utils.semantic_mask import extract_one_click_semantic_mask

    cfg = DebugConfig()
    cfg.GENERATOR_BACKBONE = "mock_tiny"
    cfg.SEMANTIC_ENCODER_TYPE = "mock_tiny"
    cfg.PATCH_SIZE = 64

    model = ControllableRestorer(cfg)
    loss_fn = ControllableRestorationLoss(cfg, use_pretrained_vgg=False)
    disc = VisionAidedDiscriminator(sem_dim=model.semantic_encoder.spatial_dim, num_actions=cfg.NUM_ACTIONS)

    assert len(model.skip_vae.vae_pix_lora_params) > 0 and len(model.skip_vae.vae_sem_lora_params) > 0

    clean = torch.rand(2, 3, 64, 64)
    src = torch.rand(2, 3, 64, 64)
    tgt = torch.rand(2, 3, 64, 64)
    act_map = torch.rand(2, cfg.NUM_ACTIONS, 64, 64)
    # Exercise negative signed controls on channels 4 & 5
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

    # Stage 2: Pixel Stage (LoRA + SkipVAE + Dense C-RADIO + PixelSpaceRefiner)
    model.zero_grad(set_to_none=True)
    model.set_training_stage("pixel_stage")
    out_s2 = model(src, act_map, tgt_01=tgt, clean_01=clean, fidelity_scale=1.0, detail_scale=0.0)
    loss_s2, metrics_s2 = loss_fn(
        out_s2["pred"],
        tgt,
        act_map,
        out_s2["align_loss"],
        stage="pixel_stage",
        t_vec=out_s2["t"],
        src_01=src,
        semantic_encoder=model.semantic_encoder,
    )
    assert metrics_s2["pres"] >= 0.0 and metrics_s2["color"] >= 0.0
    _backward_on_worker_thread(loss_s2)
    model.clear_runtime_contexts()
    missing_pix_grads = [
        n for n, p in model.named_parameters()
        if any(p is gp for gp in groups["pixel_stage"]) and p.grad is None
    ]
    assert not missing_pix_grads, f"Unused pixel_stage params (breaks DDP!): {missing_pix_grads}"
    assert all(p.grad is None for p in groups["semantic_stage"])

    # Stage 3: Semantic-Detail + Vision-Aided GAN Stage
    model.zero_grad(set_to_none=True)
    model.set_training_stage("semantic_stage")
    disc.eval()
    disc.requires_grad_(False)
    out_s3 = model(src, act_map, tgt_01=tgt, clean_01=clean, fidelity_scale=1.0, detail_scale=0.8)
    loss_s3, metrics_s3, aux_s3 = loss_fn(
        out_s3["pred"],
        tgt,
        act_map,
        out_s3["align_loss"],
        stage="semantic_stage",
        t_vec=out_s3["t"],
        src_01=src,
        semantic_encoder=model.semantic_encoder,
        discriminator=disc,
        return_aux=True,
    )
    _backward_on_worker_thread(loss_s3)
    model.clear_runtime_contexts()
    missing_sem_grads = [
        n for n, p in model.named_parameters()
        if any(p is gp for gp in groups["semantic_stage"]) and p.grad is None
    ]
    assert not missing_sem_grads, f"Unused semantic_stage params (breaks DDP!): {missing_sem_grads}"
    assert all(p.grad is None for p in groups["pixel_stage"])
    assert "gan" in metrics_s3 and "pred_grid" in aux_s3

    # Discriminator single-concat DDP backward check
    disc.requires_grad_(True)
    disc.train()
    disc.zero_grad(set_to_none=True)
    d_loss = VisionAidedDiscriminator.discriminator_hinge_loss(
        disc, tgt, out_s3["pred"].detach(), aux_s3["tgt_grid"], aux_s3["pred_grid"], act_map
    )
    d_loss.backward()
    assert all(p.grad is not None for p in disc.parameters())

    # Signed Negative Slider Unlock & Zero-Action Exact Preservation Check
    model.eval()
    zero_act = torch.zeros(1, cfg.NUM_ACTIONS, 64, 64)
    ident_out = model.sample(src[:1], zero_act, steps=1)
    assert torch.allclose(ident_out, src[:1], atol=1e-7), "Zero action must preserve input bit-for-bit!"

    # Pure negative slider (e.g. exposure = -0.8) must NOT be treated as zero by _apply_zero_action_lock!
    neg_act = torch.zeros(1, cfg.NUM_ACTIONS, 64, 64)
    neg_act[:, 4, :, :32] = -0.80
    # Perturb tone_head output bias slightly to confirm negative gate unlocks left half while locking right half
    with torch.no_grad():
        model.pixel_refiner.tone_head[-1].bias.fill_(0.25)
    neg_out = model.sample(src[:1], neg_act, steps=1, fidelity_scale=1.0, detail_scale=0.8)
    assert not torch.allclose(neg_out[:, :, :, :32], src[:1, :, :, :32], atol=1e-4), (
        "Negative signed slider (-0.8) must unlock restoration on active region!"
    )
    assert torch.allclose(neg_out[:, :, :, 32:], src[:1, :, :, 32:], atol=1e-7), (
        "Zero-action right half must remain bit-for-bit identical!"
    )

    # Seamless High-Res Tiled Inference Check
    big_src = torch.rand(1, 3, 100, 110)
    big_act = torch.zeros(1, cfg.NUM_ACTIONS, 100, 110)
    big_act[:, 5, :, :50] = -0.75  # Negative cool WB on left half only
    tiled_out, tiled_traj = model.sample(
        big_src, big_act, steps=2, tile_size=64, tile_overlap=16, return_trajectory=True
    )
    assert tiled_out.shape == (1, 3, 100, 110) and len(tiled_traj) == 3
    assert torch.allclose(tiled_out[:, :, :, 50:], big_src[:, :, :, 50:], atol=1e-6)

    mask_2d = extract_one_click_semantic_mask(model.semantic_encoder, src[:1], point_xy=(0.5, 0.5))
    assert mask_2d.shape == (64, 64) and 0.0 <= float(mask_2d.min()) and float(mask_2d.max()) <= 1.0
    print("[PASS] V4 3-phase Dual-LoRA, Dense C-RADIO, PixelRefiner, and Signed Zero-Action Lock verified.")


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
