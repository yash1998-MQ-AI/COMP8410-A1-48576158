"""Week 5 tests on fake meshes: renderer, crops, IoU, the generation loop and the report. No GPU, no models."""

import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")
from PIL import Image  # noqa: E402

PAIR_DIR = Path(__file__).resolve().parents[1] / "pairs" / "w5_image_to_3d"
sys.path.insert(0, str(PAIR_DIR))
sys.path.insert(0, str(PAIR_DIR.parents[1]))

import render_splat as RS  # noqa: E402
from a1eval import Record, append, load  # noqa: E402


def textured_box(path):
    m = trimesh.creation.box((0.6, 1.0, 0.4))
    tex = np.zeros((64, 64, 3), np.uint8)
    tex[:, :32] = (255, 0, 0)
    tex[:, 32:] = (0, 0, 255)
    uv = np.c_[m.vertices[:, 0] / 0.6 + 0.5, m.vertices[:, 1] + 0.5]
    m.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.fromarray(tex))
    m.export(path)
    return m


def test_texture_colours_survive_glb_roundtrip(tmp_path):
    textured_box(tmp_path / "b.glb")
    mesh = RS.load_mesh(tmp_path / "b.glb")
    pts, col, nrm, textured = RS.sample_points(mesh, 20000)
    assert textured
    assert col[:, 0].max() > 200 and col[:, 2].max() > 200 and col[:, 1].max() < 30  # red and blue only


def test_untextured_mesh_is_grey_and_flagged(tmp_path):
    trimesh.creation.icosphere().export(tmp_path / "s.glb")
    mesh = RS.load_mesh(tmp_path / "s.glb")
    _, col, _, textured = RS.sample_points(mesh, 5000)
    assert not textured and np.all(col == 180)


def test_render_mask_matches_projected_size():
    mesh = trimesh.creation.box((1.0, 2.0, 1.0))   # tall box
    pts, col, nrm, _ = RS.sample_points(mesh, 200000)
    pts, nrm = RS.orient(pts, nrm, "y")
    img, mask = RS.render(pts, col, nrm, 0, 0, 256)
    ys, xs = np.where(mask)
    assert (ys.max() - ys.min()) > 1.7 * (xs.max() - xs.min())   # taller than wide
    assert mask.sum() > 0.1 * mask.size and (img[~mask] == 255).all()


def test_z_up_model_is_stood_upright():
    mesh = trimesh.creation.box((1.0, 1.0, 3.0))   # tall along z
    pts, col, nrm, _ = RS.sample_points(mesh, 100000)
    p_y, _ = RS.orient(pts, nrm, "y")
    p_z, _ = RS.orient(pts, nrm, "z")
    assert np.ptp(p_y[:, 2]) > 2.5 and np.ptp(p_z[:, 1]) > 2.5


def test_nearest_surface_wins_zbuffer():
    # a red plate in front of a blue plate: from the front only red may show
    front = trimesh.creation.box((1, 1, 0.02)); front.apply_translation((0, 0, 0.5))
    back = trimesh.creation.box((1, 1, 0.02)); back.apply_translation((0, 0, -0.5))
    front.visual = trimesh.visual.ColorVisuals(front, vertex_colors=np.tile([255, 0, 0, 255], (8, 1)))
    back.visual = trimesh.visual.ColorVisuals(back, vertex_colors=np.tile([0, 0, 255, 255], (8, 1)))
    mesh = trimesh.util.concatenate([front, back])
    pts, col, nrm, _ = RS.sample_points(mesh, 200000)
    pts, nrm = RS.orient(pts, nrm, "y")
    img, mask = RS.render(pts, col, nrm, 0, 0, 128)
    centre = img[60:68, 60:68].reshape(-1, 3).mean(0)
    assert centre[0] > 150 and centre[2] < 60


def test_crop_square_is_scale_invariant():
    mask = np.zeros((300, 400), bool); mask[100:200, 150:200] = True
    big = np.zeros((300, 400), bool); big[50:250, 100:200] = True
    rgb = np.full((300, 400, 3), 255, np.uint8)
    _, m1 = RS.crop_square(rgb, mask, 64); _, m2 = RS.crop_square(rgb, big, 64)
    assert abs(m1.mean() - m2.mean()) < 0.02   # same shape, different size -> same crop


def test_iou_basics():
    a = np.zeros((10, 10), bool); a[:5] = True
    assert RS.iou(a, a) == 1.0 and RS.iou(a, ~a) == 0.0
    b = np.zeros((10, 10), bool); b[:, :5] = True
    assert RS.iou(a, b) == pytest.approx(25 / 75)


def test_photo_loader_composites_on_white(tmp_path):
    rgba = np.zeros((100, 100, 4), np.uint8); rgba[30:70, 30:70] = (10, 20, 30, 255)
    Image.fromarray(rgba, "RGBA").save(tmp_path / "p.png")
    img, mask = RS.load_photo(tmp_path / "p.png", 64)
    assert mask.any() and img[0, 0].tolist() == [255, 255, 255]


def test_generation_loop_is_resumable_and_records_failures(tmp_path):
    import gen_common
    import w5_common

    (tmp_path / "in").mkdir()
    for it in ("c01_topiary", "c02_lamp"):
        Image.new("RGBA", (8, 8)).save(tmp_path / "in" / f"{it}.png")
    calls = []

    def load_fn():
        return object()

    def gen_fn(h, image_path, seed, out):
        calls.append((Path(image_path).stem, seed))
        if seed == 1 and "lamp" in image_path:
            raise RuntimeError("OOM")
        Path(out).write_bytes(b"glb")
        return {"shape": 0.1, "texture_export": 0.2}

    argv = ["--items", "c01_topiary", "c02_lamp", "--seeds", "0", "1", "--inputs", str(tmp_path / "in"),
            "--out-root", str(tmp_path / "out"), "--results", str(tmp_path / "r.jsonl")]
    assert gen_common.run("trellis2", load_fn, gen_fn, argv) == 0
    rows = load(tmp_path / "r.jsonl")
    ok = [r for r in rows if r.metric == "gen_ok"]
    assert len(ok) == 4 and sum(r.value for r in ok) == 3
    assert any(r.metric == "latency_shape_s" for r in rows) and any(r.metric == "latency_s" for r in rows)
    n_calls = len(calls)
    gen_common.run("trellis2", load_fn, gen_fn, argv)       # second run: everything has a sidecar, only warm-up runs
    assert len(calls) == n_calls + 1


def test_report_tables_on_synthetic_results(tmp_path):
    import report_w5 as R

    rng = np.random.default_rng(0)
    path = tmp_path / "w5.jsonl"
    for model, base in (("trellis2", 0.75), ("hunyuan3d", 0.65)):
        for i in range(12):
            for seed in (0, 1, 2):
                for m in ("dino_best", "dino_mean", "clip_i_best", "clip_i_mean", "sil_iou_best"):
                    append(path, Record("w5", model, f"c{i:02d}", m, base + rng.normal(0, 0.03), seed))
                append(path, Record("w5", model, f"c{i:02d}", "dino_best_mismatch", 0.3, seed))
                append(path, Record("w5", model, f"c{i:02d}", "latency_s", 40 if model == "trellis2" else 120, seed))
                append(path, Record("w5", model, f"c{i:02d}", "gen_ok", 1.0, seed))
    df = R.frame(path)
    q = R.quality_table(df)
    row = q[q.metric == "dino_best"].iloc[0]
    assert row.diff_A_minus_B > 0.05 and row.verdict == "A better" and row.A_wins == 12
    assert not R.control_table(df).empty and not R.run_table(df).empty


def test_mesh_stats_ignore_uv_seams_and_count_pieces():
    box = trimesh.creation.box((1, 1, 1))
    st = RS.mesh_stats(box)
    assert st["watertight"] == 1.0 and st["n_components"] == 1 and st["n_faces"] == 12
    # split every vertex per face (what UV seams do): still the same closed surface
    split = trimesh.Trimesh(vertices=box.vertices[box.faces].reshape(-1, 3), faces=np.arange(36).reshape(-1, 3), process=False)
    st2 = RS.mesh_stats(split)
    assert st2["watertight"] == 1.0 and st2["n_components"] == 1 and st2["n_vertices"] == 36
    two = trimesh.util.concatenate([box, box.copy().apply_translation((3, 0, 0))])
    assert RS.mesh_stats(two)["n_components"] == 2
    open_mesh = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-1], process=False)
    assert RS.mesh_stats(open_mesh)["watertight"] == 0.0
