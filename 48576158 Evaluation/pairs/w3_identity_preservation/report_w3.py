"""Week 3: tables, figures and a qualitative grid from results/w3.jsonl.

    python pairs/w3_identity_preservation/report_w3.py --show          # inline, in Colab
    python pairs/w3_identity_preservation/report_w3.py --grid          # also the image grids

Tables and figures go to report_assets/w3/ (safe to commit). The qualitative
grids contain faces, so they are written under data/processed/w3_run/figures/,
which is git-ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from a1eval import load  # noqa: E402
from a1eval import metrics as M  # noqa: E402

COLOUR = {"pulid-v1.1": "#1b6ca8", "instantid": "#d1495b"}
HIGHER_IS_BETTER = {"id_facenet": True, "id_facenet_heldout": True, "id_arcface": True,
                    "clip_t": True, "face_found": True, "latency_s": False, "peak_vram_gb": False}
LABEL = {"id_facenet": "Identity (FaceNet cosine)", "id_facenet_heldout": "Identity vs held-out photos",
         "id_arcface": "Identity (ArcFace cosine)", "clip_t": "Prompt adherence (CLIP)",
         "face_found": "Face found", "latency_s": "Latency (s)", "peak_vram_gb": "Peak VRAM (GB)"}


def frame(results: Path) -> pd.DataFrame:
    rows = []
    for r in load(results):
        rows.append({"model": r.model, "item_id": r.item_id, "metric": r.metric, "value": r.value,
                     "seed": r.seed, "identity": r.meta.get("identity"), "scenario": r.meta.get("scenario"),
                     "item": r.meta.get("item")})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no rows in {results}")
    return df


def ci_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = []
    for keys, g in df.groupby(by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        vals = g["value"].tolist()
        mean, lo, hi = M.bootstrap_ci(vals, n_boot=10000, seed=0)
        out.append(dict(zip(by, keys)) | {"n": len(vals), "n_valid": int(np.sum(~np.isnan(vals))),
                                          "mean": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(out)


def paired_table(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Mean difference a - b over items both models produced, with a bootstrap CI."""
    out = []
    for metric, g in df.groupby("metric"):
        wide = g.pivot_table(index="item_id", columns="model", values="value", dropna=False)
        if a not in wide or b not in wide:
            continue
        pair = wide[[a, b]].dropna()
        if pair.empty:
            continue
        mean, lo, hi = M.paired_bootstrap_diff(pair[a].tolist(), pair[b].tolist(), n_boot=10000, seed=0)
        out.append({"metric": metric, "n_items": len(pair), f"{a}_mean": pair[a].mean(),
                    f"{b}_mean": pair[b].mean(), "diff": mean, "lo": lo, "hi": hi,
                    "significant": "yes" if (lo > 0 or hi < 0) else "no",   # a CI of exactly [0, 0] is not a difference
                    "higher_is_better": HIGHER_IS_BETTER.get(metric, True)})
    return pd.DataFrame(out)


def _bars(ax, sub: pd.DataFrame, group: str, models: list[str], title: str, ylabel: str) -> None:
    levels = sorted(sub[group].dropna().unique())
    x = np.arange(len(levels))
    w = min(0.8 / max(len(models), 1), 0.3)
    for i, m in enumerate(models):
        s = sub[sub["model"] == m].set_index(group).reindex(levels)
        off = (i - (len(models) - 1) / 2) * w
        mean, lo, hi = (s[c].to_numpy(dtype=float) for c in ("mean", "lo", "hi"))
        ax.bar(x + off, mean, w, label=m, color=COLOUR.get(m, f"C{i}"),
               yerr=[mean - lo, hi - mean], capsize=3, error_kw={"lw": 1})
    ax.set_xticks(x, [LABEL.get(v, v) if group == "metric" else v for v in levels], rotation=0, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.25)


def figures(df: pd.DataFrame, out: Path, calib: dict | None, show: bool):
    import matplotlib.pyplot as plt

    models = [m for m in ["pulid-v1.1", "instantid"] if m in set(df["model"])]
    out.mkdir(parents=True, exist_ok=True)

    # 1. headline: identity against prompt adherence, plus face-found rate
    head = ci_table(df[df["metric"].isin(["id_facenet", "id_facenet_heldout", "clip_t", "face_found"])],
                    ["metric", "model"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, metrics, title in [(axes[0], ["id_facenet", "id_facenet_heldout"], "Identity preserved"),
                               (axes[1], ["clip_t"], "Prompt adherence"),
                               (axes[2], ["face_found"], "Face rendered at all")]:
        _bars(ax, head[head["metric"].isin(metrics)], "metric", models, title, "mean (95% CI)")
    if calib:
        for key, style, lbl in [("within_person", "--", "same person, different photo"),
                                ("between_person", ":", "different people")]:
            v = (calib.get(key) or {}).get("mean")
            if v is not None:
                axes[0].axhline(v, ls=style, lw=1, color="#444")
                axes[0].annotate(lbl, (0.02, v), xycoords=("axes fraction", "data"), fontsize=7,
                                 va="bottom", color="#444")
    axes[0].legend(fontsize=8)
    fig.suptitle("Week 3: PuLID v1.1 vs InstantID", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "w3_headline.png", dpi=160)
    if show:
        plt.show()

    # 2. fairness: identity score per reference identity, and per scenario
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    idf = df[df["metric"] == "id_facenet"]
    _bars(axes[0], ci_table(idf, ["identity", "model"]), "identity", models,
          "Identity score by reference face", "FaceNet cosine (95% CI)")
    _bars(axes[1], ci_table(idf, ["scenario", "model"]), "scenario", models,
          "Identity score by scenario", "FaceNet cosine (95% CI)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "w3_fairness.png", dpi=160)
    if show:
        plt.show()

    # 3. cost of running it
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric in zip(axes, ["latency_s", "peak_vram_gb"]):
        sub = ci_table(df[df["metric"] == metric], ["model"])
        mean, lo, hi = (sub[c].to_numpy(dtype=float) for c in ("mean", "lo", "hi"))
        ax.bar(sub["model"], mean, 0.5, color=[COLOUR.get(m, "C0") for m in sub["model"]],
               yerr=[mean - lo, hi - mean], capsize=4, error_kw={"lw": 1})
        ax.set_title(LABEL[metric], fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "w3_cost.png", dpi=160)
    if show:
        plt.show()


def grids(items: list[dict], out_root: Path, fig_dir: Path, show: bool, seed: int = 0):
    """Reference photo next to each model's image, for the report's qualitative section."""
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    models = [m for m in ["pulid-v1.1", "instantid"] if (out_root / "w3" / m).is_dir()]
    plans = {"identities": [it for it in items if it["scenario"] == "headshot"],
             "scenarios": [it for it in items if it["identity"] == "me"]}
    for name, rows in plans.items():
        if not rows:
            continue
        fig, axes = plt.subplots(len(rows), 1 + len(models),
                                 figsize=(3.1 * (1 + len(models)), 3.1 * len(rows)))
        axes = np.atleast_2d(axes)
        for r, it in enumerate(rows):
            ref = Path(it["ref"])
            panels = [("reference", ref if ref.is_absolute() else ROOT / ref)]
            panels += [(m, out_root / "w3" / m / f"{it['item_id']}__s{seed}.png") for m in models]
            for c, (title, path) in enumerate(panels):
                ax = axes[r, c]
                ax.axis("off")
                if path.exists():
                    ax.imshow(M.load_image(path))
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
                if r == 0:
                    ax.set_title(title, fontsize=10)
                if c == 0:
                    ax.text(-0.04, 0.5, it["identity"] if name == "identities" else it["scenario"],
                            transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=10)
        fig.suptitle(f"Week 3 qualitative: by {name} (seed {seed})", fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / f"w3_qualitative_{name}.jpg", dpi=130, pil_kwargs={"quality": 88})
        if show:
            plt.show()
        print(f"wrote {fig_dir / f'w3_qualitative_{name}.jpg'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "w3.jsonl"))
    ap.add_argument("--items", default=str(Path(__file__).with_name("items.csv")))
    ap.add_argument("--out-root", default=str(ROOT / "outputs"))
    ap.add_argument("--out", default=str(ROOT / "report_assets" / "w3"))
    ap.add_argument("--fig-dir", default=str(ROOT / "data" / "processed" / "w3_run" / "figures"))
    ap.add_argument("--show", action="store_true", help="display inline (Colab)")
    ap.add_argument("--grid", action="store_true", help="also build the qualitative image grids")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    df = frame(Path(args.results))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    calib_path = Path(args.results).with_name("w3_calibration.json")
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else None

    summary = ci_table(df, ["metric", "model"]).sort_values(["metric", "model"])
    paired = paired_table(df, "pulid-v1.1", "instantid")
    by_identity = ci_table(df[df["metric"] == "id_facenet"], ["identity", "model"]).sort_values(["identity", "model"])
    by_scenario = ci_table(df[df["metric"] == "id_facenet"], ["scenario", "model"]).sort_values(["scenario", "model"])

    for name, t in [("w3_summary", summary), ("w3_paired", paired),
                    ("w3_by_identity", by_identity), ("w3_by_scenario", by_scenario)]:
        t.to_csv(out / f"{name}.csv", index=False)
        (out / f"{name}.md").write_text(t.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")

    pd.set_option("display.width", 160, "display.max_columns", 20)
    print("\n== per model ==\n", summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n== PuLID v1.1 minus InstantID, paired by item ==\n",
          paired.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n== identity score by reference face ==\n",
          by_identity.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if calib:
        print("\n== FaceNet scale ==\n  same person, different photo:",
              round(calib["within_person"]["mean"], 3) if calib["within_person"]["mean"] else "n/a",
              "\n  different people:",
              round(calib["between_person"]["mean"], 3) if calib["between_person"]["mean"] else "n/a")

    figures(df, out, calib, args.show)
    if args.grid:
        items = list(csv.DictReader(open(args.items, encoding="utf-8")))
        grids(items, Path(args.out_root), Path(args.fig_dir), args.show)
    print(f"\nwrote tables and figures to {out}")


if __name__ == "__main__":
    main()
