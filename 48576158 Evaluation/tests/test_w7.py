"""Week 7 unit tests: the pure functions, with no model and no GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAIR = ROOT / "pairs" / "w7_video_editing"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PAIR))

import w7_common as C  # noqa: E402


def test_vae_frames_is_4n_plus_1_and_capped():
    for n in range(5, 200):
        v = C.vae_frames(n)
        assert (v - 1) % 4 == 0
        assert 5 <= v <= C.MAX_FRAMES
        assert v <= n
    assert C.vae_frames(73) == 73
    assert C.vae_frames(81) == 81
    assert C.vae_frames(80) == 77
    assert C.vae_frames(1000) == C.MAX_FRAMES


def test_box_pixels_rounds_into_frame():
    assert C.box_pixels((0.0, 0.0, 1.0, 1.0)) == (0, 0, C.WIDTH, C.HEIGHT)
    x0, y0, x1, y1 = C.box_pixels((0.25, 0.5, 0.75, 1.0))
    assert (x0, y0, x1, y1) == (208, 240, 624, 480)


def test_matched_settings_between_models():
    a, b = C.SETTINGS["wan21_vace"], C.SETTINGS["lucy_edit"]
    for k in ("height", "width", "steps", "guidance"):
        assert a[k] == b[k], f"{k} differs: this pair must be a matched comparison"


def test_mask_frames_white_inside_black_outside():
    from gen_w7 import mask_frames

    ms = mask_frames((0.25, 0.5, 0.75, 1.0), 3)
    assert len(ms) == 3
    a = np.asarray(ms[0])
    assert a.shape == (C.HEIGHT, C.WIDTH, 3)
    assert a[300, 400].max() == 255          # inside the box
    assert a[10, 10].max() == 0              # outside the box


def test_cover_resize_keeps_target_shape():
    from pairs.w7_video_editing.prepare_inputs import cover_resize

    for shape in ((720, 1280, 3), (1280, 720, 3), (480, 480, 3)):
        out = cover_resize(np.zeros(shape, dtype=np.uint8))
        assert out.shape == (C.HEIGHT, C.WIDTH, 3)


def test_box_mask_matches_box_pixels():
    from score_w7 import box_mask

    m = box_mask((0.0, 0.5, 1.0, 1.0), (C.HEIGHT, C.WIDTH, 3))
    assert m.shape == (C.HEIGHT, C.WIDTH)
    assert not m[0].any()
    assert m[-1].all()
    assert m.sum() == pytest.approx(C.HEIGHT * C.WIDTH / 2, rel=0.01)


def test_psnr_identical_is_inf_and_noisy_is_finite():
    from score_w7 import psnr

    from score_w7 import PSNR_MAX

    a = np.full((20, 20), 128, dtype=np.uint8)
    assert psnr(a, a) == PSNR_MAX
    b = a.astype(np.int16) + 10
    assert 20 < psnr(a, np.clip(b, 0, 255).astype(np.uint8)) < 40


def test_region_fidelity_detects_a_change_confined_to_the_box():
    from score_w7 import region_fidelity

    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (C.HEIGHT, C.WIDTH, 3), dtype=np.uint8)
    src = [base.copy() for _ in range(5)]
    out = []
    for _ in range(5):
        f = base.copy()
        f[240:480, 208:624] = 0            # change only inside the box
        out.append(f)
    m = region_fidelity(src, out, (0.25, 0.5, 0.75, 1.0))
    assert m["bg_psnr"] > 60                                   # background untouched
    assert m["box_mad"] > 10                                   # box clearly changed
    assert m["edit_ratio"] > 10                                 # change is localised
    assert m["bg_mad"] == pytest.approx(0.0, abs=1e-6)         # nothing leaked outside
    assert 0.0 <= m["bg_ssim"] <= 1.0


def test_temporal_stats_zero_on_a_frozen_clip():
    from score_w7 import temporal_stats

    f = np.full((C.HEIGHT, C.WIDTH, 3), 100, dtype=np.uint8)
    m = temporal_stats([f.copy() for _ in range(6)])
    assert m["motion_mad"] == pytest.approx(0.0, abs=1e-6)
    assert m["flicker"] == pytest.approx(0.0, abs=1e-6)


def test_temporal_stats_flicker_flags_alternating_frames():
    from score_w7 import temporal_stats

    a = np.full((C.HEIGHT, C.WIDTH, 3), 50, dtype=np.uint8)
    b = np.full((C.HEIGHT, C.WIDTH, 3), 200, dtype=np.uint8)
    alt = temporal_stats([a, b, a, b, a, b])
    ramp = temporal_stats([np.full_like(a, v) for v in (50, 80, 110, 140, 170, 200)])
    assert alt["flicker"] > ramp["flicker"] * 5     # same motion, far more high-frequency noise


def test_sample_idx_bounds():
    from score_w7 import sample_idx

    for n in (1, 2, 5, 73, 81):
        idx = sample_idx(n)
        assert idx[0] == 0 and idx[-1] == n - 1
        assert len(idx) <= min(n, C.N_SCORE_FRAMES)
        assert idx == sorted(set(idx))


def test_cos_rows_is_one_for_identical_rows():
    from score_w7 import cos_rows

    a = np.random.default_rng(1).normal(size=(4, 8))
    assert np.allclose(cos_rows(a, a), 1.0, atol=1e-6)


def test_short_side_crop_is_square():
    from score_w7 import short_side_crop

    out = short_side_crop(np.zeros((C.HEIGHT, C.WIDTH, 3), dtype=np.uint8))
    assert out.shape == (C.CROP, C.CROP, 3)


def test_blank_masked_greys_only_the_box():
    from PIL import Image

    from gen_w7 import blank_masked

    im = Image.new("RGB", (C.WIDTH, C.HEIGHT), (10, 200, 30))
    box = (0.25, 0.5, 0.75, 1.0)
    out = blank_masked([im, im], box)
    assert len(out) == 2
    a = np.asarray(out[0])
    x0, y0, x1, y1 = C.box_pixels(box)
    assert (a[y0 + 5, x0 + 5] == 127).all()          # inside the box is mid-grey
    assert (a[5, 5] == (10, 200, 30)).all()          # outside is untouched
    assert (a[y0 - 5, x0 + 5] == (10, 200, 30)).all()
