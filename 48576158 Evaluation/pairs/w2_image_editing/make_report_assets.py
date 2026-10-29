"""Week 2: tables and figures for the report, built only from results/w2.jsonl.

    python pairs/w2_image_editing/make_report_assets.py
    python pairs/w2_image_editing/make_report_assets.py --qualitative    # also the photo grid (private)

Public outputs (no photos), written to report_assets/w2/:
  w2_summary.csv / .md        per model and metric: n, mean, 95% bootstrap CI
  w2_paired.csv / .md         Qwen minus FLUX on the same 15 photos: mean difference, 95% CI, wins
  w2_by_edit.csv / .md        mean metrics per edit type
  w2_quality_paired.png       per-photo lines for the three quality metrics
  w2_edge_ssim_by_edit.png    layout kept, by edit type
  w2_speed_memory.png         time per edit and peak GPU memory

Private output (shows the rooms; git-ignored), written to data/processed/w2_run/figures/:
  w2_qualitative.png          original | FLUX | Qwen for one photo per edit type
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from a1eval import load  # noqa: E402
from a1eval.metrics import bootstrap_ci  # noqa: E402

A, B = "flux1-kontext-dev", "qwen-image-edit-2509"
NAME = {A: "FLUX.1 Kontext [dev]", B: "Qwen-Image-Edit-2509"}
COLOUR = {A: "#1b6ca8", B: "#d1495b"}  # blue / red: passes the CVD and contrast checks
INK, MUTED, GRID = "#1a1a19", "#52514e", "#e4e3df"
QUALITY = [("edge_ssim", "Layout kept\n(edge SSIM)"), ("clip_i", "Content kept\n(CLIP-I)"), ("clip_gain", "Edit achieved\n(CLIP gain)")]
EDIT_ORDER = ["brighten", "overcast", "golden_hour", "twilight"]


def frame(results: Path) -> pd.DataFrame:
    rows = load(results)
    df = pd.DataFrame([{"model": r.model, "item_id": r.item_id, "metric": r.metric, "value": r.value, **(r.meta or {})} for r in rows])
    edit = df[df.metric == "latency_s"].drop_duplicates("item_id").set_index("item_id")["edit_type"]
    df["edit_type"] = df["item_id"].map(edit)
    return df


def wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df[df.metric == metric].pivot(index="item_id", columns="model", values="value").dropna()


def paired(df: pd.DataFrame, metric: str, seed: int = 0, n_boot: int = 10000) -> dict:
    w = wide(df, metric)
    d = (w[B] - w[A]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d))).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"metric": metric, "n": len(d), "mean_diff_qwen_minus_flux": d.mean(), "ci_lo": lo, "ci_hi": hi, "qwen_higher_on": int((d > 0).sum())}


def md_table(t: pd.DataFrame, digits: int = 3) -> str:
    cols = list(t.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in t.iterrows():
        out.append("| " + " | ".join(f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def tables(df: pd.DataFrame, out: Path) -> dict:
    s = []
    for (model, metric), g in df.groupby(["model", "metric"]):
        m, lo, hi = bootstrap_ci(g["value"].tolist())
        s.append({"model": NAME[model], "metric": metric, "n": len(g), "mean": m, "ci_lo": lo, "ci_hi": hi})
    summ = pd.DataFrame(s)
    p = pd.DataFrame([paired(df, m) for m, _ in QUALITY] + [paired(df, "latency_s"), paired(df, "peak_vram_gb")])
    lat = df[df.metric == "latency_s"].groupby("model").value.mean()
    p_speed = pd.DataFrame({"model": [NAME[A], NAME[B]], "images_per_hour": [3600 / lat[A], 3600 / lat[B]]})
    by = (df[df.metric.isin([m for m, _ in QUALITY])].pivot_table(index=["edit_type", "model"], columns="metric", values="value").reset_index())
    by["model"] = by["model"].map(NAME)
    by["edit_type"] = pd.Categorical(by["edit_type"], EDIT_ORDER, ordered=True)
    by = by.sort_values(["edit_type", "model"])
    n_by = df[(df.metric == "latency_s") & (df.model == A)].groupby("edit_type").size()
    by.insert(2, "photos", by["edit_type"].map(n_by).astype(int))
    for name, t in (("w2_summary", summ), ("w2_paired", p), ("w2_by_edit", by), ("w2_speed", p_speed)):
        t.to_csv(out / f"{name}.csv", index=False)
        (out / f"{name}.md").write_text(md_table(t) + "\n", encoding="utf-8")
    return {"summary": summ, "paired": p, "by_edit": by, "speed": p_speed}


def fig_quality(df: pd.DataFrame, pr: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.2))
    for ax, (m, title) in zip(axes, QUALITY):
        w = wide(df, m)
        for _, r in w.iterrows():
            ax.plot([0, 1], [r[A], r[B]], color="#b9b8b2", linewidth=0.9, zorder=1)
        for x, mod in enumerate((A, B)):
            ax.scatter(np.full(len(w), x), w[mod], s=18, color=COLOUR[mod], edgecolor="white", linewidth=0.6, zorder=3)
            ax.hlines(w[mod].mean(), x - 0.18, x + 0.18, color=INK, linewidth=2.0, zorder=4)
        row = pr[pr.metric == m].iloc[0]
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xticks([0, 1], ["FLUX", "Qwen"], fontsize=9, color=INK)
        ax.set_xlim(-0.4, 1.4)
        sign = "+" if row.mean_diff_qwen_minus_flux >= 0 else "−"
        ax.set_xlabel(f"Qwen {sign}{abs(row.mean_diff_qwen_minus_flux):.3f} (95% CI {row.ci_lo:+.3f} to {row.ci_hi:+.3f})\nQwen higher on {row.qwen_higher_on} of {row.n} photos",
                      fontsize=7.5, color=MUTED, labelpad=6)
        style(ax)
    fig.suptitle("Same 15 photos, same prompts: each line joins one photo's two scores (bar = mean)", fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "w2_quality_paired.png", dpi=220)
    plt.close(fig)


def fig_by_edit(df: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    w_a, w_b = wide(df, "edge_ssim")[A], wide(df, "edge_ssim")[B]
    et = df[df.metric == "latency_s"].drop_duplicates("item_id").set_index("item_id").edit_type
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(EDIT_ORDER))
    for off, (mod, ser) in zip((-0.19, 0.19), ((A, w_a), (B, w_b))):
        means = [ser[et[ser.index] == e].mean() for e in EDIT_ORDER]
        ax.bar(x + off, means, width=0.34, color=COLOUR[mod], label=NAME[mod])
        for xi, e in zip(x, EDIT_ORDER):
            v = ser[et[ser.index] == e]
            ax.scatter(np.full(len(v), xi + off), v, s=11, color="white", edgecolor=INK, linewidth=0.6, zorder=3)
    ax.set_xticks(x, [f"{e.replace('_', ' ')}\n(n={int((et.loc[w_a.index] == e).sum())})" for e in EDIT_ORDER], fontsize=8.5, color=INK)
    ax.set_ylabel("Layout kept (edge SSIM, higher = better)", fontsize=8.5, color=MUTED)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2, bbox_to_anchor=(1, 1.12))
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "w2_edge_ssim_by_edit.png", dpi=220)
    plt.close(fig)


def fig_speed(df: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, (m, title, unit) in zip(axes, (("latency_s", "Time per edit", "seconds"), ("peak_vram_gb", "Peak GPU memory", "GB"))):
        vals = [df[(df.metric == m) & (df.model == mod)].value for mod in (A, B)]
        means = [v.mean() for v in vals]
        ax.bar([0, 1], means, color=[COLOUR[A], COLOUR[B]], width=0.55)
        for i, v in enumerate(means):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks([0, 1], ["FLUX", "Qwen"], fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_ylabel(unit, fontsize=8.5, color=MUTED)
        ax.set_ylim(0, max(means) * 1.15)
        style(ax)
    fig.tight_layout()
    fig.savefig(out / "w2_speed_memory.png", dpi=220)
    plt.close(fig)


def fig_qualitative(df: pd.DataFrame, root: Path, items: pd.DataFrame, out: Path, picks: dict[str, str]) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, axs = plt.subplots(len(picks), 3, figsize=(9.6, 2.45 * len(picks)), squeeze=False)
    for r, (edit, item) in enumerate(picks.items()):
        paths = [root / items.loc[item, "source"], root / "outputs/w2" / A / f"{item}.png", root / "outputs/w2" / B / f"{item}.png"]
        titles = [f"Original: {item}\n(edit asked: {edit.replace('_', ' ')})", "FLUX.1 Kontext [dev]", "Qwen-Image-Edit-2509"]
        for c in range(3):
            axs[r][c].imshow(Image.open(paths[c]).convert("RGB"))
            axs[r][c].axis("off")
            axs[r][c].set_title(titles[c], fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "w2_qualitative.png", dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT), help="repo folder holding results/, outputs/ and data/")
    ap.add_argument("--results", default=None)
    ap.add_argument("--qualitative", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    out = root / "report_assets" / "w2"
    out.mkdir(parents=True, exist_ok=True)
    df = frame(Path(args.results) if args.results else root / "results" / "w2.jsonl")
    t = tables(df, out)
    fig_quality(df, t["paired"], out)
    fig_by_edit(df, out)
    fig_speed(df, out)
    print(t["paired"].round(3).to_string(index=False))
    print(f"wrote tables and figures to {out}")
    if args.qualitative:
        items = pd.read_csv(Path(__file__).with_name("items.csv")).set_index("item_id")
        picks = {"golden_hour": "bedroom_desk", "twilight": "bedroom_door", "overcast": "street_flare", "brighten": "bedroom_frame"}
        fig_qualitative(df, root, items, root / "data/processed/w2_run/figures", picks)
        print("wrote private qualitative grid to data/processed/w2_run/figures/")


if __name__ == "__main__":
    main()
