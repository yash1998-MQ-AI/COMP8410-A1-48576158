import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from a1eval import load  # noqa: E402
from a1eval.metrics import edge_ssim  # noqa: E402


def _scene(seed=0, size=(96, 128)):
    rng = np.random.default_rng(seed)
    img = np.zeros((*size, 3), dtype=np.uint8) + 80
    for _ in range(12):  # blocks give edges
        y, x = rng.integers(0, size[0] - 20), rng.integers(0, size[1] - 20)
        img[y : y + 20, x : x + 20] = rng.integers(0, 255, 3)
    return img


def test_edge_ssim_identity_and_lighting_shift():
    a = _scene()
    assert edge_ssim(a, a) == pytest.approx(1.0)
    brighter = np.clip(a.astype(int) * 1.3 + 20, 0, 255).astype(np.uint8)  # lighting change, same layout
    assert edge_ssim(a, brighter) > 0.8
    other = _scene(seed=5)  # different layout
    assert edge_ssim(a, other) < edge_ssim(a, brighter)


def test_edge_ssim_handles_size_mismatch():
    a = _scene()
    b = np.asarray(Image.fromarray(a).resize((64, 48)))
    assert 0.0 < edge_ssim(a, b) <= 1.0


def test_w2_pipeline_smoke(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    rows = []
    for i in range(3):
        p = src_dir / f"it{i}.jpg"
        Image.fromarray(_scene(i, (120, 160))).save(p)
        rows.append({"item_id": f"it{i}", "source": str(p), "origin": "x", "edit_type": "brighten",
                     "prompt": "p", "target_text": "t", "note": ""})
    items = tmp_path / "items.csv"
    with items.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    res = tmp_path / "w2.jsonl"
    out = tmp_path / "outputs"
    run = [sys.executable, str(ROOT / "pairs/w2_image_editing/run_w2.py"), "--model", "identity",
           "--items", str(items), "--out-root", str(out), "--results", str(res), "--long-side", "64"]
    subprocess.run(run, check=True)
    assert len([r for r in load(res) if r.metric == "latency_s"]) == 3
    subprocess.run(run, check=True)  # resume: nothing re-run
    assert len([r for r in load(res) if r.metric == "latency_s"]) == 3
    subprocess.run([sys.executable, str(ROOT / "pairs/w2_image_editing/score_w2.py"), "--items", str(items),
                    "--out-root", str(out), "--results", str(res), "--metrics", "edge_ssim"], check=True)
    scores = [r for r in load(res) if r.metric == "edge_ssim"]
    assert len(scores) == 3 and all(0.0 < r.value <= 1.0 for r in scores)
    assert (out / "w2" / "identity" / "it0.png").exists()
