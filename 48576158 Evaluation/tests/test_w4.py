"""Week 4 helpers on small fake data (no GPU, no nerfstudio needed)."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

PAIR = Path(__file__).resolve().parents[1] / "pairs" / "w4_nerf_3dgs"
sys.path.insert(0, str(PAIR))

import install_nerfstudio as inst  # noqa: E402
import report_w4 as rep  # noqa: E402
import w4_common as common  # noqa: E402


def test_run_dir_layout(tmp_path):
    assert common.run_dir("splatfacto", 2, tmp_path) == tmp_path / "splatfacto_s2" / "splatfacto" / "run"


@pytest.mark.parametrize("req,name", [
    ("viser ==0.2.7", "viser"), ("torch >=1.13.1", "torch"), ("torchmetrics[image] >=1.0.1", "torchmetrics"),
    ('rawpy >=0.18.1 ; platform_machine != "arm64"', "rawpy"), ("fpsample", "fpsample"), ("Pillow >=10.3.0", "pillow"),
])
def test_requirement_name(req, name):
    assert inst.requirement_name(req) == name


def _write_results(path, gap):
    rows = []
    rng = np.random.default_rng(0)
    for model, base in (("nerfacto", 24.0), ("splatfacto", 24.0 + gap)):
        for seed in (0, 1):
            for v in range(10):
                for metric, val in (("psnr", base + rng.normal(0, 0.3)), ("ssim", 0.8), ("lpips", 0.3)):
                    rows.append({"pair": "w4", "model": model, "item_id": f"IMG_{v:02d}__s{seed}", "metric": metric,
                                 "value": float(val), "seed": seed, "meta": {}, "ts": time.time()})
            rows.append({"pair": "w4", "model": model, "item_id": f"train__s{seed}", "metric": "train_time_s",
                         "value": 600.0, "seed": seed, "meta": {}, "ts": time.time()})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_paired_detects_a_real_gap_and_ignores_none(tmp_path):
    f = tmp_path / "w4.jsonl"
    _write_results(f, gap=1.5)
    v = rep.view_table(rep.frame(f))
    assert set(v["view"]) == {f"IMG_{i:02d}" for i in range(10)}      # seed suffix removed, run items excluded
    p = rep.paired(v).set_index("metric")
    assert p.loc["psnr", "diff"] == pytest.approx(1.5, abs=0.4)
    assert p.loc["psnr", "significant"] == "yes"
    assert p.loc["ssim", "significant"] == "no"                         # identical values: CI is [0, 0]


def test_cost_table_prices_training(tmp_path):
    f = tmp_path / "w4.jsonl"
    _write_results(f, gap=0.0)
    run = rep.run_table(rep.frame(f))
    priced = rep.cost_table(run, units_per_hr=6.0)
    # 600 s = 1/6 h; 6 units/h * A$0.1513 per unit
    assert priced["aud_per_run"].iloc[0] == pytest.approx(600 / 3600 * 6.0 * 0.1513)
    assert rep.cost_table(run, None)["aud_per_run"].isna().all()


def test_report_end_to_end_writes_tables_and_figures(tmp_path):
    f = tmp_path / "w4.jsonl"
    _write_results(f, gap=1.0)
    df = rep.frame(f)
    v = rep.view_table(df)
    out = tmp_path / "assets"
    import matplotlib
    matplotlib.use("Agg")
    rep.figures(v, rep.model_summary(v), rep.run_table(df), out, show=False)
    assert (out / "w4_quality.png").exists() and (out / "w4_per_view_psnr.png").exists()
