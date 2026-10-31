"""Week 5: tables, figures and a qualitative grid from results/w5.jsonl.

    python pairs/w5_image_to_3d/report_w5.py --show

A photo's score is the mean over the generation seeds; the model difference is a paired bootstrap over the
photos (item = unit of analysis, seeds are repeats of the same item). Run-level numbers (latency, memory) are
shown as mean with min-max range. report_assets/w5/ holds tables and plots only; the qualitative grid goes to
outputs/w5_figures/ (git-ignored) because it contains the photographed objects.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from a1eval import load  # noqa: E402
from a1eval import metrics as M  # noqa: E402
from w5_common import CROP_SIZE, LABEL, MODELS, RESULTS, ROOT, VIEWS_ROOT  # noqa: E402

COLOUR = {"trellis2": "#1b6ca8", "hunyuan3d": "#d1495b"}
A, B = MODELS
QUALITY = ["dino_best", "dino_mean", "clip_i_best", "clip_i_mean", "sil_iou_best"]
CONTROL = {"dino_best": "dino_best_mismatch", "clip_i_best": "clip_i_best_mismatch"}
RUN = ["latency_s", "latency_shape_s", "latency_texture_export_s", "peak_vram_gb", "peak_vram_reserved_gb"]
GEOM = ["n_faces", "watertight", "n_components", "textured", "glb_size_mb"]
TITLE = {"dino_best": "DINOv2 similarity, best view", "dino_mean": "DINOv2 similarity, mean of 8 views",
         "clip_i_best": "CLIP-I, best view", "clip_i_mean": "CLIP-I, mean of 8 views",
         "sil_iou_best": "Silhouette IoU, best view"}


def frame(path: Path) -> pd.DataFrame:
    rows = [{"model": r.model, "item": r.item_id, "metric": r.metric, "value": r.value, "seed": r.seed}
            for r in load(path)]
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return pd.DataFrame(rows)


def per_item(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = df[df.metric == metric].groupby(["item", "model"])["value"].mean().unstack("model")
    return g.dropna()


def quality_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m in QUALITY:
        t = per_item(df, m)
        if t.empty or A not in t or B not in t:
            continue
        a, b = t[A].to_numpy(), t[B].to_numpy()
        ma, la, ha = M.bootstrap_ci(a)
        mb, lb, hb = M.bootstrap_ci(b)
        d, ld, hd = M.paired_bootstrap_diff(a, b)
        out.append({"metric": m, "n_items": len(t), f"{A}": ma, f"{A}_lo": la, f"{A}_hi": ha,
                    f"{B}": mb, f"{B}_lo": lb, f"{B}_hi": hb, "diff_A_minus_B": d, "diff_lo": ld, "diff_hi": hd,
                    "A_wins": int((a > b).sum()), "B_wins": int((b > a).sum()),
                    "verdict": "A better" if ld > 0 else ("B better" if hd < 0 else "no clear difference")})
    return pd.DataFrame(out)


def control_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m, c in CONTROL.items():
        for model in MODELS:
            s, k = df[(df.metric == m) & (df.model == model)]["value"], df[(df.metric == c) & (df.model == model)]["value"]
            if len(s) and len(k):
                out.append({"metric": m, "model": model, "matching_photo": s.mean(), "different_item_photo": k.mean(),
                            "gap": s.mean() - k.mean()})
    return pd.DataFrame(out)


def run_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m in RUN + ["load_time_s"]:
        for model in MODELS:
            v = df[(df.metric == m) & (df.model == model)]["value"]
            if len(v):
                out.append({"metric": m, "model": model, "n": len(v), "mean": v.mean(), "min": v.min(), "max": v.max()})
    return pd.DataFrame(out)


def geometry_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m in GEOM:
        for model in MODELS:
            v = df[(df.metric == m) & (df.model == model)]["value"]
            if len(v):
                out.append({"metric": m, "model": model, "n": len(v), "mean": v.mean(), "min": v.min(), "max": v.max()})
    for model in MODELS:
        v = df[(df.metric == "gen_ok") & (df.model == model)]["value"]
        if len(v):
            out.append({"metric": "generation_success_rate", "model": model, "n": len(v), "mean": v.mean(),
                        "min": v.min(), "max": v.max()})
    return pd.DataFrame(out)


def seed_stability(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m in ("dino_best", "sil_iou_best"):
        for model in MODELS:
            g = df[(df.metric == m) & (df.model == model)].groupby("item")["value"].std(ddof=0)
            if len(g):
                out.append({"metric": m, "model": model, "mean_within_item_sd": g.mean()})
    return pd.DataFrame(out)


def md(df: pd.DataFrame, digits: int = 3) -> str:
    return df.to_markdown(index=False, floatfmt=f".{digits}f")


def plot_quality(q: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    q = q[q.metric.isin(["dino_best", "clip_i_best", "sil_iou_best"])]
    fig, axes = plt.subplots(1, len(q), figsize=(3.2 * len(q), 3.2))
    for ax, r in zip(np.atleast_1d(axes), q.itertuples()):
        for i, model in enumerate(MODELS):
            m, lo, hi = getattr(r, model), getattr(r, f"{model}_lo"), getattr(r, f"{model}_hi")
            ax.bar(i, m, yerr=[[m - lo], [hi - m]], capsize=4, color=COLOUR[model])
        ax.set_xticks(range(2), [LABEL[m] for m in MODELS], rotation=15, ha="right")
        ax.set_title(TITLE[r.metric], fontsize=9)
        ax.set_ylim(0, 1)
    fig.suptitle(f"Agreement with the input photo (mean over items, 95% bootstrap CI, n={int(q.n_items.min())})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_items(df: pd.DataFrame, out: Path, metric: str = "dino_best") -> None:
    import matplotlib.pyplot as plt

    t = per_item(df, metric)
    if t.empty:
        return
    t = t.sort_values(A)
    fig, ax = plt.subplots(figsize=(5.6, 0.34 * len(t) + 1.2))
    y = np.arange(len(t))
    ax.hlines(y, t[B], t[A], color="#999999", lw=1)
    ax.scatter(t[A], y, color=COLOUR[A], label=LABEL[A], zorder=3)
    ax.scatter(t[B], y, color=COLOUR[B], label=LABEL[B], zorder=3)
    ax.set_yticks(y, t.index)
    ax.set_xlabel(TITLE[metric])
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_run(r: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    keep = [("latency_s", "Latency per mesh (s)"), ("peak_vram_gb", "Peak GPU memory (GB, allocated)")]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    for ax, (m, title) in zip(axes, keep):
        for i, model in enumerate(MODELS):
            s = r[(r.metric == m) & (r.model == model)]
            if s.empty:
                continue
            v = s.iloc[0]
            ax.bar(i, v["mean"], yerr=[[v["mean"] - v["min"]], [v["max"] - v["mean"]]], capsize=4, color=COLOUR[model])
        ax.set_xticks(range(2), [LABEL[m] for m in MODELS], rotation=15, ha="right")
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def qualitative_grid(df: pd.DataFrame, out: Path, seed: int = 0) -> bool:
    """Photo | TRELLIS.2 best view | Hunyuan3D best view, one row per item (seed 0). Reads the saved view sheets."""
    from a1eval.schema import load as load_rows

    best = {}
    for r in load_rows(RESULTS if not hasattr(qualitative_grid, "results") else qualitative_grid.results):
        if r.metric == "dino_best" and r.seed == seed:
            best[(r.model, r.item_id)] = int(r.meta.get("best_view_dino", 0))
    items = sorted({i for (_, i) in best})
    if not items:
        return False
    rows = []
    for it in items:
        tiles = []
        for k, model in enumerate(MODELS):
            f = qualitative_grid.views_root / model / f"{it}_s{seed}.jpg"
            if not f.exists():
                tiles = []
                break
            sheet = Image.open(f)
            if k == 0:
                tiles.append(sheet.crop((0, 0, CROP_SIZE, CROP_SIZE)))
            v = best.get((model, it), 0) + 1   # tile 0 is the photo
            cols = sheet.width // CROP_SIZE
            x, y = (v % cols) * CROP_SIZE, (v // cols) * CROP_SIZE
            tiles.append(sheet.crop((x, y, x + CROP_SIZE, y + CROP_SIZE)))
        if tiles:
            rows.append((it, tiles))
    if not rows:
        return False
    from PIL import ImageDraw

    pad = 18
    grid = Image.new("RGB", (3 * CROP_SIZE, len(rows) * (CROP_SIZE + pad)), (255, 255, 255))
    d = ImageDraw.Draw(grid)
    for r, (it, tiles) in enumerate(rows):
        for c, t in enumerate(tiles):
            grid.paste(t, (c * CROP_SIZE, r * (CROP_SIZE + pad) + pad))
        d.text((4, r * (CROP_SIZE + pad) + 3), f"{it}   photo | {LABEL[A]} | {LABEL[B]}", fill=(0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out, quality=90)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=ROOT / "report_assets" / "w5")
    ap.add_argument("--views-root", type=Path, default=VIEWS_ROOT)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    matplotlib.use("Agg")

    df = frame(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    q, c, r, g, s = quality_table(df), control_table(df), run_table(df), geometry_table(df), seed_stability(df)
    for name, t in (("quality", q), ("control", c), ("run", r), ("geometry", g), ("seed_stability", s)):
        t.to_csv(args.out / f"w5_{name}.csv", index=False)
    text = ["# Week 5 results", "", "## Agreement with the input photo (per item, mean over seeds)", md(q), "",
            "## Metric sanity check: matching photo vs a different item's photo", md(c), "",
            "## Run cost per mesh", md(r), "", "## Geometry and reliability", md(g), "",
            "## Seed stability (mean within-item SD)", md(s), ""]
    (args.out / "w5_tables.md").write_text("\n".join(text), encoding="utf-8")
    if not q.empty:
        plot_quality(q, args.out / "w5_quality.png")
        plot_items(df, args.out / "w5_items_dino.png")
    if not r.empty:
        plot_run(r, args.out / "w5_run.png")
    qualitative_grid.views_root = args.views_root
    qualitative_grid.results = args.results
    ok = qualitative_grid(df, args.views_root / "w5_qualitative.jpg")
    print(f"wrote {args.out} (qualitative grid: {'yes' if ok else 'no'})")
    if args.show:
        print("\n".join(text))


if __name__ == "__main__":
    main()
