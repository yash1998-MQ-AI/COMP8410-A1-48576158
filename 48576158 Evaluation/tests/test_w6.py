"""Week 6 tests on synthetic videos (no GPU, no models)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pairs" / "w6_video_generation"))
sys.path.insert(0, str(ROOT))

import prepare_inputs as P  # noqa: E402
import score_w6 as S  # noqa: E402
import w6_common as C  # noqa: E402


def test_items_manifest():
    ids = C.item_ids()
    assert len(ids) == len(set(ids)) == 6
    assert all(C.prompt_of(i) for i in ids)


def test_frame_count_is_4n_plus_1():
    assert (C.NUM_FRAMES - 1) % 4 == 0


def test_crop_16x9_from_4x3():
    out = P.crop_16x9(Image.new("RGB", (1536, 1157), (10, 20, 30)))
    assert out.size == (1280, 720)


def test_sample_idx():
    assert S.sample_idx(73) == [0, 9, 18, 27, 36, 45, 54, 63, 72]
    assert S.sample_idx(3, 9) == [0, 1, 2]


def test_motion_static_vs_moving():
    x = np.linspace(0, 6 * np.pi, 320)
    base = np.tile((127 + 100 * np.sin(x))[None, :, None], (180, 1, 3)).astype(np.uint8)   # smooth texture
    static = [base.copy() for _ in range(10)]
    assert S.motion_stats(static)["motion_mad"] == 0.0
    smooth = [np.roll(base, 3 * i, axis=1) for i in range(10)]
    m = S.motion_stats(smooth)
    assert m["motion_mad"] > 1 and m["jitter_ratio"] < 0.2   # constant-speed pan: second difference ~ 0 relative to first
    shaky = [np.roll(base, 12 * (i % 2), axis=1) for i in range(10)]
    assert S.motion_stats(shaky)["jitter_ratio"] > 0.5      # back-and-forth: second difference is as large as the first


def test_luma_drift():
    a = np.full((90, 160, 3), 100, np.uint8)
    b = np.full((90, 160, 3), 130, np.uint8)
    assert S.motion_stats([a, a, b])["luma_drift"] == pytest.approx(30, abs=1)


def test_first_frame_ssim_identity():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    assert S.first_frame_ssim(img, img) == pytest.approx(1.0, abs=1e-6)
    assert S.first_frame_ssim(img, 255 - img) < 0.5


def test_short_side_crop():
    assert S.short_side_crop(np.zeros((144, 256, 3), np.uint8)).shape == (224, 224, 3)


def test_cos_rowwise():
    a = np.array([[1.0, 0], [0, 1.0]])
    assert S.cos(a, np.array([1.0, 0])).tolist() == pytest.approx([1.0, 0.0])


def test_read_video_roundtrip(tmp_path):
    import imageio.v2 as iio
    p = tmp_path / "t.mp4"
    frames = [np.full((64, 96, 3), 20 * i, np.uint8) for i in range(8)]
    iio.mimwrite(str(p), frames, fps=24)
    got, fps = S.read_video(p)
    assert len(got) == 8 and abs(fps - 24) < 1
