import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a1eval import Record, append, done_items, load, log_cost, timed, total_cost
from a1eval.aggregate import summarise, to_frame, to_markdown, write_pair_outputs
from a1eval.metrics import (
    bootstrap_ci,
    frame_flicker,
    masked_psnr,
    paired_bootstrap_diff,
    psnr,
    ssim,
    wilson_ci,
)


def test_psnr_known_value():
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = np.full((8, 8, 3), 10, dtype=np.uint8)
    assert psnr(a, b) == pytest.approx(10 * math.log10(255**2 / 100))
    assert psnr(a, a) == math.inf


def test_ssim_identical_is_one():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert ssim(a, a) == pytest.approx(1.0)
    assert ssim(a, 255 - a) < 0.5


def test_masked_psnr_ignores_changed_region():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = a.copy()
    b[:5] = 200  # top half changed
    keep = np.zeros((10, 10), dtype=bool)
    keep[5:] = True  # only bottom half counts
    assert masked_psnr(a, b, keep) == math.inf


def test_frame_flicker():
    f0 = np.zeros((4, 4, 3), dtype=np.uint8)
    f1 = np.full((4, 4, 3), 10, dtype=np.uint8)
    assert frame_flicker([f0, f1, f0]) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        frame_flicker([f0])


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    m, lo, hi = bootstrap_ci(vals)
    assert lo <= m <= hi
    assert bootstrap_ci(vals) == (m, lo, hi)
    assert math.isnan(bootstrap_ci([])[0])
    assert bootstrap_ci([float("nan"), 2.0]) == (2.0, 2.0, 2.0)


def test_paired_bootstrap_diff_sign():
    m, lo, hi = paired_bootstrap_diff([2, 3, 4, 5], [1, 1, 1, 1])
    assert m == pytest.approx(2.5) and lo > 0
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1], [1, 2])


def test_wilson_ci():
    p, lo, hi = wilson_ci(8, 10)
    assert p == 0.8 and lo < 0.8 < hi and 0 <= lo and hi <= 1
    assert math.isnan(wilson_ci(0, 0)[0])


def test_record_roundtrip_and_resume(tmp_path):
    f = tmp_path / "r.jsonl"
    append(f, Record("w2", "m1", "i1", "clip_t", 0.3, seed=1, meta={"prompt": "dusk"}))
    append(f, Record("w2", "m2", "i1", "clip_t", 0.2))
    rows = load(f)
    assert len(rows) == 2 and rows[0].meta["prompt"] == "dusk"
    assert done_items(f, "m1") == {"i1"} and done_items(f, "m3") == set()


def test_timed_logs_latency(tmp_path):
    f = tmp_path / "r.jsonl"
    with timed(f, "w2", "m1", "i1", seed=0) as ctx:
        ctx["note"] = "x"
    rows = [r for r in load(f) if r.metric == "latency_s"]
    assert len(rows) == 1 and rows[0].value >= 0 and rows[0].meta["note"] == "x"


def test_cost_ledger(tmp_path):
    f = tmp_path / "cost.csv"
    log_cost(f, "2026-10-30", "colab", "A100-80GB", 2.0, 0.75, "w2 run")
    log_cost(f, "2026-10-30", "colab", "L4", 1.0, 0.17)
    assert total_cost(f) == pytest.approx(1.67)
    assert total_cost(tmp_path / "missing.csv") == 0.0


def test_aggregate_outputs(tmp_path):
    f = tmp_path / "r.jsonl"
    for i in range(5):
        append(f, Record("w2", "a", f"i{i}", "clip_t", 0.30 + 0.01 * i))
        append(f, Record("w2", "b", f"i{i}", "clip_t", 0.25 + 0.01 * i))
    df = to_frame(f)
    s = summarise(df)
    assert set(s["model"]) == {"a", "b"} and (s["n"] == 5).all()
    assert "clip_t" in to_markdown(s)
    out = tmp_path / "out"
    write_pair_outputs(f, out)
    assert (out / "summary.csv").exists() and (out / "w2_clip_t.png").exists()


def test_survey_items_and_analysis(tmp_path):
    import csv
    import json
    import subprocess

    root = Path(__file__).resolve().parents[1]
    outs = tmp_path / "outputs"
    for pair in ("w2", "w3"):
        for model in ("ma", "mb"):
            d = outs / pair / model
            d.mkdir(parents=True)
            for item in ("i1", "i2", "i3"):
                (d / f"{item}.png").write_bytes(b"x")
    sv = tmp_path / "survey"
    subprocess.run([sys.executable, str(root / "survey/make_ab_items.py"), "--outputs", str(outs), "--out", str(sv), "--per-pair", "2"], check=True)
    key = json.loads((sv / "answer_key.json").read_text())
    assert len(key) == 4
    # every respondent always picks the model "ma"
    with (sv / "responses.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["respondent_id", "version"] + [f"q{q}" for q in key])
        for rid, ver in (("r1", "V1"), ("r2", "V2")):
            w.writerow([rid, ver] + [next(s for s in ("left", "right") if key[q][ver][s] == "ma") for q in key])
    subprocess.run([sys.executable, str(root / "survey/analyse_survey.py"), "--responses", str(sv / "responses.csv"), "--key", str(sv / "answer_key.json"), "--out", str(sv / "summary.csv")], check=True)
    rows = list(csv.DictReader((sv / "summary.csv").open()))
    ma = [r for r in rows if r["model"] == "ma"]
    assert ma and all(float(r["preference"]) == 1.0 for r in ma)
