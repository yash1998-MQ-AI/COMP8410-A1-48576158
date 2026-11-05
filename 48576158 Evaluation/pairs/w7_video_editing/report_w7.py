"""Week 7 tables and figures from results/w7.jsonl.

    python pairs/w7_video_editing/report_w7.py

One row per item (seeds averaged); model difference = paired bootstrap over items.
Writes report_assets/w7/{w7_edit.csv,w7_fidelity.csv,w7_temporal.csv,w7_run.csv,
w7_control.csv,w7_per_item.csv,w7_edit.png,summary.md}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from a1eval import load  # noqa: E402
from a1eval import metrics as M  # noqa: E402
from w7_common import LABEL, MODELS, RESULTS, ROOT  # noqa: E402

A, B = MODELS
COLOUR = {A: "#1b6ca8", B: "#d1495b"}
EDIT = ["clip_t_gain", "clip_t", "clip_t_src", "box_mad", "bg_mad", "edit_ratio"]
FIDELITY = ["bg_psnr", "bg_ssim", "dino_src_mean"]
TEMPORAL = ["flicker", "flicker_src", "flicker_excess", "dino_consec", "motion_mad", "motion_mad_src"]
RUN = ["latency_s", "peak_vram_gb", "file_mb", "n_frames"]
HIGHER_IS_BETTER = {"clip_t_gain": True, "clip_t": True, "bg_psnr": True, "bg_ssim": True,
                    "dino_src_mean": True, "dino_consec": True, "flicker": False,
                    "flicker_excess": False, "latency_s": False, "peak_vram_gb": False}


def frame(results: Path) -> pd.DataFrame:
    rows = [{"model": r.model, "item": r.item_id, "seed": r.seed, "metric": r.metric, "value": r.value}
            for r in load(results) if r.item_id != "all"]
    return pd.DataFrame(rows)


def per_item(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
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
        out.append({"metric": m, "n_items": len(t),
                    A: ma, f"{A}_lo": la, f"{A}_hi": ha,
                    B: mb, f"{B}_lo": lb, f"{B}_hi": hb,
                    "diff_A_minus_B": d, "diff_lo": ld, "diff_hi": hd,
                    "A_higher": int((a > b).sum()), "B_higher": int((b > a).sum()),
                    "verdict": "A higher" if ld > 0 else ("B higher" if hd < 0 else "no clear difference")})
    return pd.DataFrame(out)


def run_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in RUN:
        for model in MODELS:
            v = df[(df.metric == m) & (df.model == model)].value.to_numpy()
            if v.size:
                rows.append({"metric": m, "model": model, "n": int(v.size), "mean": float(v.mean()),
                             "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())})
    return pd.DataFrame(rows)


def control_table(df: pd.DataFrame) -> pd.DataFrame:
    """clip_t against the right instruction vs a different item's instruction."""
    rows = []
    for model in MODELS:
        d = df[df.model == model]
        matched = d[d.metric == "clip_t"].value.to_numpy()
        mismatched = d[d.metric == "clip_t_mismatch"].value.to_numpy()
        if matched.size and mismatched.size:
            rows.append({"model": model, "n": int(matched.size),
                         "clip_t_matched_instruction": float(matched.mean()),
                         "clip_t_other_instruction": float(mismatched.mean()),
                         "gap": float(matched.mean() - mismatched.mean())})
    return pd.DataFrame(rows)


def per_item_table(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["clip_t_gain", "bg_psnr", "box_mad", "flicker_excess", "latency_s"]
    out = []
    for m in keep:
        t = per_item(df, m)
        for item, r in t.iterrows():
            row = {"item": item, "metric": m}
            row.update({k: float(v) for k, v in r.items()})
            out.append(row)
    return pd.DataFrame(out)


def plot(df: pd.DataFrame, path: Path) -> None:
    panels = [("clip_t_gain", "Edit effect (CLIP gain)"), ("bg_psnr", "Background PSNR (dB)"),
              ("flicker_excess", "Added flicker"), ("latency_s", "Latency (s)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.4))
    for ax, (m, title) in zip(np.atleast_1d(axes), panels):
        t = per_item(df, m)
        if t.empty or A not in t or B not in t:
            ax.set_axis_off()
            ax.set_title(f"{title}\n(no data)")
            continue
        x = np.arange(len(t))
        ax.bar(x - 0.2, t[A], 0.4, label=LABEL[A], color=COLOUR[A])
        ax.bar(x + 0.2, t[B], 0.4, label=LABEL[B], color=COLOUR[B])
        ax.set_xticks(x)
        ax.set_xticklabels(t.index, rotation=35, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="#444", lw=0.6)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=ROOT / "report_assets" / "w7")
    args = ap.parse_args(argv)

    df = frame(args.results)
    if df.empty:
        print(f"no rows in {args.results}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    tables = {"w7_edit": compare(df, EDIT), "w7_fidelity": compare(df, FIDELITY),
              "w7_temporal": compare(df, TEMPORAL), "w7_run": run_table(df),
              "w7_control": control_table(df), "w7_per_item": per_item_table(df)}
    for name, t in tables.items():
        if not t.empty:
            t.to_csv(args.out / f"{name}.csv", index=False)
    plot(df, args.out / "w7_edit.png")

    lines = [f"# Week 7 results --- {LABEL[A]} (A) vs {LABEL[B]} (B)", ""]
    for title, key in (("Did the edit land", "w7_edit"), ("Fidelity outside the edit region", "w7_fidelity"),
                       ("Temporal stability", "w7_temporal"), ("Run cost", "w7_run"),
                       ("Instruction-matching control", "w7_control")):
        t = tables[key]
        if not t.empty:
            lines += [f"## {title}", "", t.round(3).to_markdown(index=False), ""]
    lines += ["Higher is better for clip_t_gain, bg_psnr, bg_ssim, dino_src_mean and dino_consec;",
              "lower is better for flicker, flicker_excess, latency_s and peak_vram_gb.",
              "box_mad and motion_mad are descriptive, not better/worse.", ""]
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    for key in ("w7_edit", "w7_fidelity", "w7_temporal", "w7_control"):
        if not tables[key].empty:
            print(f"\n== {key}")
            print(tables[key].round(3).to_string(index=False))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
