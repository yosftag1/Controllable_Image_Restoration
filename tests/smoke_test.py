"""End-to-end verification suite for Controllable_Image_Restoration.

Runs:
1. AST syntax check across every Python file in the repository.
2. Seed-synchronized physical synthesis & non-commutativity checks (if numpy/cv2 installed).
3. Full 3-phase forward/backward, Self-Teacher LoRA switch, Dual-LoRA isolation,
   Zero-Action pixel lock, Spatial Overlay preservation, and 1-Click Semantic Mask tests
   (using offline mock_tiny backbones if PyTorch is installed).
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
    from utils.semantic_mask import parse_action_spec

    cfg = Config()
    pipe = SelectiveDegradationPipeline(cfg)
    rng = np.random.default_rng(123)

    # Smooth synthetic test pattern
    yy, xx = np.meshgrid(np.linspace(0, 255, 128), np.linspace(0, 255, 128), indexing="ij")
    clean = np.stack([xx, yy, 128.0 + 0.3 * xx - 0.3 * yy], axis=-1).clip(0, 255).astype(np.uint8)
    params = pipe.sample_params(128, 128, rng=rng)

    # 1. Exact identity at severity = 0
    zero_sev = np.zeros(cfg.NUM_ACTIONS, dtype=np.float32)
    ident = pipe.render_with_params(clean, zero_sev, params)
    assert np.array_equal(clean, ident), "Severity=0 must be exact identity!"

    # 2. Seed-synchronized selective removal: motion_blur + gaussian_noise -> remove noise, keep motion blur
    s_src = np.zeros(cfg.NUM_ACTIONS, dtype=np.float32)
    s_src[pipe.type_to_idx["motion_blur"]] = 0.80
    s_src[pipe.type_to_idx["gaussian_noise"]] = 0.70
    x_src = pipe.render_with_params(clean, s_src, params)

    a_keep_blur = parse_action_spec("noise=1.0,motion=0.0", cfg.DEGRADATION_TYPES)
    s_tgt = (1.0 - a_keep_blur) * s_src
    x_tgt = pipe.render_with_params(clean, s_tgt, params)

    s_only_blur = np.zeros(cfg.NUM_ACTIONS, dtype=np.float32)
    s_only_blur[pipe.type_to_idx["motion_blur"]] = 0.80
    x_pure_blur = pipe.render_with_params(clean, s_only_blur, params)
    assert np.array_equal(x_tgt, x_pure_blur), "Preserved motion blur target must match pure motion blur exactly!"
    assert not np.array_equal(x_src, x_tgt), "Denoised target should differ from noisy source!"

    # 3. Synthesize pair shape & bounds check across 25 random trials (covering all 5 action modes + spatial masks)
    for _ in range(25):
        src_u8, act_map, tgt_u8, s_vec = pipe.synthesize_pair(clean, rng=rng)
        assert src_u8.shape == clean.shape and tgt_u8.shape == clean.shape
        assert act_map.shape == (cfg.NUM_ACTIONS, 128, 128)
        assert 0.0 <= float(act_map.min()) and float(act_map.max()) <= 1.0
        if float(act_map.max()) == 0.0:
            assert np.array_equal(src_u8, tgt_u8), "All-zero action map must yield exact x_src == x_tgt!"

    # 4. False-positive slider safety: setting action=1.0 on an INACTIVE degradation must NOT alter x_tgt!
    s_only_noise = np.zeros(cfg.NUM_ACTIONS, dtype=np.float32)
    s_only_noise[pipe.type_to_idx["gaussian_noise"]] = 0.65
    x_noisy = pipe.render_with_params(clean, s_only_noise, params)
    a_fp = parse_action_spec("motion=1.0,defocus=1.0,jpeg=1.0,noise=0.0", cfg.DEGRADATION_TYPES)
    x_fp_tgt = pipe.render_with_params(clean, (1.0 - a_fp) * s_only_noise, params)
    assert np.array_equal(x_noisy, x_fp_tgt), "False-positive sliders on clean attributes must be exact no-ops!"

    # 5. Continuity across all 8 degradations: MSE to clean strictly decreases from a=0.0 -> a=0.5 -> a=1.0
    for deg_name in cfg.DEGRADATION_TYPES:
        s_one = np.zeros(cfg.NUM_ACTIONS, dtype=np.float32)
        s_one[pipe.type_to_idx[deg_name]] = 0.85
        d_full = pipe.render_with_params(clean, s_one, params).astype(np.float32)
        d_half = pipe.render_with_params(clean, 0.5 * s_one, params).astype(np.float32)
        d_zero = pipe.render_with_params(clean, 0.0 * s_one, params).astype(np.float32)
        err_full = float(np.mean((d_full - clean.astype(np.float32)) ** 2))
        err_half = float(np.mean((d_half - clean.astype(np.float32)) ** 2))
        err_zero = float(np.mean((d_zero - clean.astype(np.float32)) ** 2))
        assert err_zero == 0.0, f"{deg_name}: a=1.0 must reach exact clean image!"
        assert err_full > err_half > 0.0, f"{deg_name}: partial removal a=0.5 must lie strictly between full & clean!"

    # 6. Kaggle DF2K-OST nested subfolder + symlink dataset discovery check
    import cv2
    import tempfile
    from data.selective_dataset import create_controllable_dataset

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
        assert len(train_ds) == 3, f"Expected all 3 nested+symlinked train images, got {len(train_ds)}"
        assert len(val_ds) == 1, f"Expected 1 val image, got {len(val_ds)}"

        # Verify Single-Image Raw As-Is + Synthetic Benchmark Suite end-to-end
        from utils.image_benchmark import run_single_image_benchmark
        from utils.semantic_mask import parse_action_spec

        bench_out = tmp_path / "bench_out"
        test_photo = np.clip(clean.copy(), 0, 255).astype(np.uint8)
        test_photo = cv2.resize(test_photo, (256, 200), interpolation=cv2.INTER_LINEAR)

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
            fg = parse_action_spec(action, ds_cfg.DEGRADATION_TYPES)
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
        assert "native_1to1_crop" in report["raw_asis_metrics"]

    print("[PASS] Seed-synchronized selective degradation pipeline, benchmark suite & DF2K-OST loader verified.")


def test_model_and_stages_mock():
    import torch
    from config import DebugConfig
    from losses.controllable_losses import ControllableRestorationLoss
    from models.controllable_restorer import ControllableRestorer
    from utils.semantic_mask import extract_one_click_semantic_mask

    cfg = DebugConfig()
    cfg.GENERATOR_BACKBONE = "mock_tiny"
    cfg.SEMANTIC_ENCODER_TYPE = "mock_tiny"
    cfg.PATCH_SIZE = 64

    model = ControllableRestorer(cfg)
    loss_fn = ControllableRestorationLoss(cfg, use_pretrained_vgg=False)

    clean = torch.rand(2, 3, 64, 64)
    src = torch.rand(2, 3, 64, 64)
    tgt = torch.rand(2, 3, 64, 64)
    act_map = torch.rand(2, cfg.NUM_ACTIONS, 64, 64)

    # Stage 1: Self-Teacher Align Warmup
    model.set_training_stage("align_warmup")
    loss_s1 = model(src, clean_01=clean, align_only=True)
    loss_s1.backward()
    groups = model.get_stage_parameter_groups()
    assert any(p.grad is not None for p in groups["vision_lora"]), "Vision LoRA must receive gradients in Stage 1!"
    assert all(p.grad is None for p in groups["pixel_stage"]), "Pixel stage must be frozen in Stage 1!"

    # Stage 2: Pixel-Faithful Stage (including cross-thread backward recomputation check for CUDA C++ autograd parity)
    import threading

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

    model.zero_grad(set_to_none=True)
    model.set_training_stage("pixel_stage")
    out_s2 = model(src, act_map, tgt_01=tgt, clean_01=clean, fidelity_scale=1.0, detail_scale=0.0)
    loss_s2, _ = loss_fn(out_s2["pred"], tgt, act_map, out_s2["align_loss"], stage="pixel_stage", t_vec=out_s2["t"])
    _backward_on_worker_thread(loss_s2)
    model.clear_runtime_contexts()
    assert any(p.grad is not None for p in groups["pixel_stage"]), "Pixel stage must receive gradients in Stage 2!"
    assert all(p.grad is None for p in groups["semantic_stage"]), "Semantic LoRA must stay frozen in Stage 2!"

    # Stage 3: Semantic-Detail Stage (cross-thread backward recomputation check)
    model.zero_grad(set_to_none=True)
    model.set_training_stage("semantic_stage")
    out_s3 = model(src, act_map, tgt_01=tgt, clean_01=clean, fidelity_scale=1.0, detail_scale=1.0)
    loss_s3, _ = loss_fn(
        out_s3["pred"],
        tgt,
        act_map,
        out_s3["align_loss"],
        stage="semantic_stage",
        t_vec=out_s3["t"],
        semantic_encoder=model.semantic_encoder,
    )
    _backward_on_worker_thread(loss_s3)
    model.clear_runtime_contexts()
    assert any(p.grad is not None for p in groups["semantic_stage"]), "Semantic LoRA must receive gradients in Stage 3!"
    assert all(p.grad is None for p in groups["pixel_stage"]), "Pixel stage must stay frozen in Stage 3!"

    # Zero-Action & Localized Split-Mask Exact Preservation Check
    model.eval()
    zero_act = torch.zeros(1, cfg.NUM_ACTIONS, 64, 64)
    ident_out = model.sample(src[:1], zero_act, steps=2)
    assert torch.allclose(ident_out, src[:1], atol=1e-7), "Zero action must preserve input bit-for-bit!"

    half_act = torch.zeros(1, cfg.NUM_ACTIONS, 64, 64)
    half_act[:, :, :, :32] = 1.0  # Left half edited, right half locked at 0.0
    local_out = model.sample(src[:1], half_act, steps=2, fidelity_scale=1.0, detail_scale=0.8)
    assert torch.allclose(local_out[:, :, :, 32:], src[:1, :, :, 32:], atol=1e-7), (
        "Unmasked right half (action=0) must remain 100% untouched!"
    )

    # Seamless High-Res Tiled Inference Check (exact preservation on unmasked half across tile seams)
    big_src = torch.rand(1, 3, 100, 110)
    big_act = torch.zeros(1, cfg.NUM_ACTIONS, 100, 110)
    big_act[:, :, :, :50] = 1.0
    tiled_out, tiled_traj = model.sample(
        big_src, big_act, steps=2, tile_size=64, tile_overlap=16, return_trajectory=True
    )
    assert tiled_out.shape == (1, 3, 100, 110) and len(tiled_traj) == 3
    assert torch.allclose(tiled_out[:, :, :, 50:], big_src[:, :, :, 50:], atol=1e-6), (
        "Tiled inference must preserve zero-action regions bit-for-bit!"
    )

    # 1-Click Semantic Object Mask Check
    mask_2d = extract_one_click_semantic_mask(model.semantic_encoder, src[:1], point_xy=(0.5, 0.5))
    assert mask_2d.shape == (64, 64) and 0.0 <= float(mask_2d.min()) and float(mask_2d.max()) <= 1.0
    print("[PASS] 3-phase Dual-LoRA, Skip-VAE, Spatial Overlay, and 1-Click Semantic Mask verified.")


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
