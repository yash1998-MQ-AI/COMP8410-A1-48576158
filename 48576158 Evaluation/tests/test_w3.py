import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from a1eval import load

ROOT = Path(__file__).resolve().parents[1]


def test_w3_pipeline_smoke(tmp_path):
    ref = tmp_path / "ref.jpg"
    Image.fromarray((np.random.default_rng(0).random((64, 64, 3)) * 255).astype("uint8")).save(ref)
    items = tmp_path / "items.csv"
    with items.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "identity", "scenario", "ref", "prompt", "negative"])
        w.writeheader()
        for i in range(2):
            w.writerow({"item_id": f"x{i}", "identity": "x", "scenario": "s", "ref": str(ref), "prompt": "p", "negative": "n"})
    res, out = tmp_path / "w3.jsonl", tmp_path / "outputs"
    cmd = [sys.executable, str(ROOT / "pairs/w3_identity_preservation/run_w3.py"), "--model", "identity", "--items", str(items),
           "--out-root", str(out), "--results", str(res), "--seeds", "0", "1"]
    subprocess.run(cmd, check=True)
    assert len([r for r in load(res) if r.metric == "latency_s"]) == 4
    subprocess.run(cmd, check=True)  # resume: nothing re-run
    assert len([r for r in load(res) if r.metric == "latency_s"]) == 4
    assert (out / "w3" / "identity" / "x0__s0.png").exists()
