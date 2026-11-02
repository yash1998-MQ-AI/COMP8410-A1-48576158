"""Week 6 tables, figures and a qualitative strip sheet from results/w6.jsonl.

    python pairs/w6_video_generation/report_w6.py --show

One row per item (seeds averaged); model difference = paired bootstrap over items. report_assets/w6/ gets tables and
plots only; the frame strips (they show the rooms) go to outputs/w6_figures/ (git-ignored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from a1eval import load  # noqa: E402
from a1eval import metrics as M  # noqa: E402
from w6_common import LABEL, MODELS, OUT_ROOT, RESULTS, ROOT, SEEDS, item_ids, video_path  # noqa: E402

A, B = MODELS
COLOUR = {A: "#1b6ca8", B: "#d1495b"}
QUALITY = ["clip_i_src_mean", "dino_src_mean", "dino_src_min", "dino_consec", "first_frame_ssim", "clip_t"]
MOTION = ["motion_mad", "jitter_ratio", "luma_drift", "dino_first_last"]
RUN = ["latency_s", "latency_sample_s", "peak_vram_gb", "peak_vram_reserved_gb", "file_mb"]
CONTROL = {"clip_i_src_mean": "clip_i_src_mismatch", "dino_src_mean": "dino_src_mismatch"}


def frame(results: Path) -> pd.DataFrame:
    rows = [{"model": r.model, "item": r.item_id, "seed": r.seed, "metric": r.metric, "value": r.value}
            for r in load(results) if r.item_id != "all"]
    return pd.DataFrame(rows)


def per_item(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    t = df[df.metric == metric].groupby(["item", "model"]).value.mean().unstack("model")
    return t.dropna()


def compare(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = []
    for m in metrics:
        t = per_item(df, m)
        if t.empty or A not in t or B not in t:
            continue
        a, b = t[A].to_numpy(), t[B].to_numpy()
        ma, la, ha = M.bootstrap_ci(a)
        mb, lb, hb = M.bootstrap_ci(b)
        d, ld, hd = M.paired_bootstrap_diff(a, b)
        out.append({"metric": m, "n_items": len(t), A: ma, f"{A}_lo": la, f"{A}_hi": ha, B: mb, f"{B}_lo": lb, f"{B}_hi": hb,
                    "diff_A_minus_B": d, "diff_lo": ld, "diff_hi": hd, "A_higher": int((a > b).sum()), "B_higher": int((b > a).sum()),
                    "verdict": "A higher" if ld > 0 else ("B higher" if hd < 0 else "no clear difference")})
    return pd.DataFrame(out)


def run_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in RUN:
        for model in MODELS:
            v = df[(df.metric == m) & (df.model == model)].value.to_numpy()
            if v.size:
                rows.append({"metric": m, "model": model, "n": v.size, "mean": v.mean(), "median": float(np.median(v)), "min": v.min(), "max": v.max()})
    ok = df[df.metric == "gen_ok"].groupby("model").value.agg(["sum", "count"])
    for model, r in ok.iterrows():
        rows.append({"metric": "gen_ok", "model": model, "n": int(r["count"]), "mean": r["sum"] / r["count"]})
    return pd.DataFrame(rows)


def plots(df: pd.DataFrame, out: Path) -> None:
    for name, metrics in (("quality", QUALITY[:5]), ("motion", MOTION)):
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.4))
        for ax, m in zip(np.atleast_1d(axes), metrics):
            t = per_item(df, m)
            if t.empty:
                continue
            for i, model in enumerate(MODELS):
                ax.bar(i, t[model].mean(), color=COLOUR[model], alpha=0.8)
                ax.scatter(np.full(len(t), i) + np.linspace(-0.15, 0.15, len(t)), t[model], color="k", s=10, zorder=3)
            ax.set_xticks([0, 1], [LABEL[A].split()[0], LABEL[B].split()[0]])
            ax.set_title(m, fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"w6_{name}.png", dpi=150)
        plt.close(fig)


def strips(out_dir: Path, n_frames: int = 5) -> None:
    from score_w6 import read_video, sample_idx

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in item_ids():
        for model in MODELS:
            p = video_path(model, item, SEEDS[0])
            if not p.exists():
                continue
            fr, _ = read_video(p)
            rows.append(np.concatenate([np.asarray(Image.fromarray(fr[i]).resize((256, 144))) for i in sample_idx(len(fr), n_frames)], axis=1))
    if rows:
        Image.fromarray(np.concatenate(rows, axis=0)).save(out_dir / "w6_strips.jpg", quality=88)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=ROOT / "report_assets" / "w6")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)
    df = frame(args.results)
    if df.empty:
        print("no rows in", args.results)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    q = compare(df, QUALITY + MOTION)
    q.to_csv(args.out / "w6_quality.csv", index=False)
    ctrl = compare(df, [v for v in CONTROL.values()]) if any(v in set(df.metric) for v in CONTROL.values()) else pd.DataFrame()
    ctrl.to_csv(args.out / "w6_control.csv", index=False)
    run = run_table(df)
    run.to_csv(args.out / "w6_run.csv", index=False)
    items = df[df.metric.isin(["dino_src_mean", "dino_consec", "motion_mad", "latency_s"])].pivot_table(
        index=["item", "model"], columns="metric", values="value")
    items.to_csv(args.out / "w6_per_item.csv")
    plots(df, args.out)
    try:
        strips(ROOT / "outputs" / "w6_figures")
    except Exception as exc:
        print("strips skipped:", exc)
    (args.out / "summary.md").write_text("## Quality and motion\n" + q.round(3).to_markdown(index=False) +
                                          "\n\n## Run cost\n" + run.round(2).to_markdown(index=False) + "\n", encoding="utf-8")
    if args.show:
        print(q.round(3).to_string(index=False))
        print(run.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
