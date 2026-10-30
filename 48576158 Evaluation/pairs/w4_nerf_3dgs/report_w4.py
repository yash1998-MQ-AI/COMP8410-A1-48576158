"""Week 4: tables, figures, a qualitative grid and a side-by-side clip from results/w4.jsonl.

    python pairs/w4_nerf_3dgs/report_w4.py --show --units-per-hr 4.8

report_assets/w4/ gets tables and plots with no photos in them (safe to commit). The qualitative grid and the
clip show the private room, so they go to outputs/w4_figures/, which is git-ignored.

Held-out views are evaluated per photo; a photo's score is the mean over training seeds, and the model
difference is a paired bootstrap over photos. The run-level numbers (time, memory, size, FPS) come from only
a few seeds, so they are shown as mean with the min-max range, not as a confidence interval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from a1eval import load  # noqa: E402
from a1eval import metrics as M  # noqa: E402
from w4_common import LABEL, MODELS, RESULTS, ROOT  # noqa: E402

COLOUR = {"nerfacto": "#1b6ca8", "splatfacto": "#d1495b"}
VIEW_METRICS = ["psnr", "ssim", "lpips"]
RUN_METRICS = ["train_time_s", "peak_vram_gb", "model_size_mb", "render_fps", "n_gaussians"]
TITLE = {"psnr": "PSNR (dB, higher is better)", "ssim": "SSIM (higher is better)", "lpips": "LPIPS (lower is better)",
         "train_time_s": "Training time (s)", "peak_vram_gb": "Peak GPU memory (GB)",
         "model_size_mb": "Checkpoint size (MB)", "render_fps": "Render speed (frames/s)",
         "n_gaussians": "Gaussians"}
UNIT_PRICE_AUD = 0.1513  # A$ per Colab compute unit (A$15.13 per 100)


def frame(path: Path) -> pd.DataFrame:
    rows = [{"model": r.model, "item": r.item_id, "metric": r.metric, "value": r.value, "seed": r.seed}
            for r in load(path)]
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return pd.DataFrame(rows)


def view_table(df: pd.DataFrame) -> pd.DataFrame:
    """Held-out views: one row per (photo, model, metric), averaged over seeds."""
    v = df[df["metric"].isin(VIEW_METRICS) & ~df["item"].str.startswith(("train__", "eval__"))].copy()
    v["view"] = v["item"].str.rsplit("__s", n=1).str[0]
    return v.groupby(["view", "model", "metric"], as_index=False)["value"].mean()


def model_summary(v: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (metric, model), g in v.groupby(["metric", "model"]):
        mean, lo, hi = M.bootstrap_ci(g["value"].tolist(), n_boot=10000, seed=0)
        out.append({"metric": metric, "model": model, "n_views": len(g), "mean": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(out)


def paired(v: pd.DataFrame, a: str = "splatfacto", b: str = "nerfacto") -> pd.DataFrame:
    out = []
    for metric, g in v.groupby("metric"):
        wide = g.pivot(index="view", columns="model", values="value").dropna()
        if a not in wide or b not in wide or wide.empty:
            continue
        mean, lo, hi = M.paired_bootstrap_diff(wide[a].tolist(), wide[b].tolist(), n_boot=10000, seed=0)
        out.append({"metric": metric, "n_views": len(wide), f"{a}_mean": wide[a].mean(), f"{b}_mean": wide[b].mean(),
                    "diff": mean, "lo": lo, "hi": hi, "significant": "yes" if (lo > 0 or hi < 0) else "no"})
    return pd.DataFrame(out)


def run_table(df: pd.DataFrame) -> pd.DataFrame:
    r = df[df["metric"].isin(RUN_METRICS)]
    out = []
    for (metric, model), g in r.groupby(["metric", "model"]):
        out.append({"metric": metric, "model": model, "n_seeds": len(g), "mean": g["value"].mean(),
                    "min": g["value"].min(), "max": g["value"].max()})
    return pd.DataFrame(out)


def seed_spread(df: pd.DataFrame) -> pd.DataFrame:
    """How much does the mean held-out PSNR move between training seeds? Needs >1 seed."""
    v = df[(df["metric"] == "psnr") & ~df["item"].str.startswith(("train__", "eval__"))]
    s = v.groupby(["model", "seed"], as_index=False)["value"].mean()
    return s.groupby("model", as_index=False).agg(mean_psnr=("value", "mean"), sd_across_seeds=("value", "std"),
                                                  n_seeds=("value", "size"))


def cost_table(run: pd.DataFrame, units_per_hr: float | None) -> pd.DataFrame:
    t = run[run["metric"] == "train_time_s"].copy()
    if t.empty:
        return t
    t["train_minutes_per_run"] = t["mean"] / 60
    if units_per_hr is None:
        t["aud_per_run"] = np.nan
    else:
        t["aud_per_run"] = t["mean"] / 3600 * units_per_hr * UNIT_PRICE_AUD
    return t[["model", "n_seeds", "train_minutes_per_run", "aud_per_run"]]


# ------------------------------------------------------------------- figures
def figures(v: pd.DataFrame, summ: pd.DataFrame, run: pd.DataFrame, out: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    models = [m for m in MODELS if m in set(summ["model"])]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, metric in zip(axes, VIEW_METRICS):
        s = summ[summ["metric"] == metric].set_index("model").reindex(models)
        mean, lo, hi = (s[c].to_numpy(dtype=float) for c in ("mean", "lo", "hi"))
        ax.bar([LABEL[m] for m in models], mean, 0.55, color=[COLOUR[m] for m in models],
               yerr=[np.maximum(mean - lo, 0), np.maximum(hi - mean, 0)], capsize=4, error_kw={"lw": 1})
        ax.set_title(TITLE[metric], fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        lo_lim = min(lo) - 0.5 * (max(hi) - min(lo) + 1e-9)
        if metric != "lpips":
            ax.set_ylim(max(lo_lim, 0), None)
    fig.suptitle("Week 4: held-out views (mean over photos, 95% CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "w4_quality.png", dpi=160)
    if show:
        plt.show()

    fig, ax = plt.subplots(figsize=(9, 3.6))
    wide = v[v["metric"] == "psnr"].pivot(index="view", columns="model", values="value")
    xs = np.arange(len(wide))
    for m in models:
        ax.plot(xs, wide[m], "o-", color=COLOUR[m], label=LABEL[m], lw=1.2, ms=4)
    ax.set_xticks(xs, [str(i + 1) for i in xs], fontsize=8)
    ax.set_xlabel("held-out view")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Per-view PSNR: does one model win everywhere or only on some views?", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "w4_per_view_psnr.png", dpi=160)
    if show:
        plt.show()

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, metric in zip(axes, ["train_time_s", "render_fps", "model_size_mb", "peak_vram_gb"]):
        s = run[run["metric"] == metric].set_index("model").reindex(models)
        if s["mean"].isna().all():
            ax.axis("off")
            continue
        mean = s["mean"].to_numpy(dtype=float)
        ax.bar([LABEL[m] for m in models], mean, 0.55, color=[COLOUR[m] for m in models],
               yerr=[np.maximum(mean - s["min"].to_numpy(dtype=float), 0),
                     np.maximum(s["max"].to_numpy(dtype=float) - mean, 0)],
               capsize=4, error_kw={"lw": 1})
        ax.set_title(TITLE[metric], fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        if metric == "render_fps":
            ax.set_yscale("log")
    fig.suptitle("Week 4: cost of producing and serving the model (mean, bars show min to max over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "w4_cost.png", dpi=160)
    if show:
        plt.show()


def qualitative(v: pd.DataFrame, out_root: Path, fig_dir: Path, show: bool, seed: int, k: int = 5) -> None:
    """Real photo, both renders and both error maps for the views that separate the models most."""
    import matplotlib.pyplot as plt
    from PIL import Image

    fig_dir.mkdir(parents=True, exist_ok=True)
    wide = v[v["metric"] == "psnr"].pivot(index="view", columns="model", values="value").dropna()
    if wide.empty or any(m not in wide for m in MODELS):
        return
    diff = (wide["splatfacto"] - wide["nerfacto"]).sort_values()
    picks = list(diff.index[: k // 2]) + list(diff.index[-(k - k // 2):])  # nerfacto-favouring then 3DGS-favouring
    picks = list(dict.fromkeys(picks))

    def read(path: Path):
        return np.asarray(Image.open(path).convert("RGB")) if path.exists() else None

    fig, axes = plt.subplots(len(picks), 5, figsize=(15, 2.7 * len(picks)))
    axes = np.atleast_2d(axes)
    heads = ["real photo", LABEL["nerfacto"], LABEL["splatfacto"], "error: Nerfacto", "error: 3DGS"]
    frames = []
    for r, view in enumerate(picks):
        gt = read(out_root / "w4" / "gt" / f"{view}.png")
        pn = read(out_root / "w4" / "nerfacto" / f"s{seed}" / f"{view}.png")
        ps = read(out_root / "w4" / "splatfacto" / f"s{seed}" / f"{view}.png")
        if gt is None or pn is None or ps is None:
            continue
        en = np.abs(gt.astype(int) - pn.astype(int)).mean(-1)
        es = np.abs(gt.astype(int) - ps.astype(int)).mean(-1)
        for c, img in enumerate([gt, pn, ps]):
            axes[r, c].imshow(img)
        axes[r, 3].imshow(en, cmap="magma", vmin=0, vmax=80)
        axes[r, 4].imshow(es, cmap="magma", vmin=0, vmax=80)
        axes[r, 0].set_ylabel(f"d PSNR {diff[view]:+.1f} dB", fontsize=9)
        for c in range(5):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(heads[c], fontsize=10)
        frames.append(np.concatenate([gt, pn, ps], axis=1))
    fig.suptitle("Held-out views where the models differ most (seed %d; error maps share one scale)" % seed, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "w4_qualitative.jpg", dpi=120, pil_kwargs={"quality": 88})
    if show:
        plt.show()
    print("wrote", fig_dir / "w4_qualitative.jpg")
    clip(out_root, fig_dir, v, seed)


def clip(out_root: Path, fig_dir: Path, v: pd.DataFrame, seed: int, seconds_per_view: float = 1.5) -> None:
    """Every held-out view as real photo | Nerfacto | 3DGS, for the presentation."""
    from PIL import Image, ImageDraw

    views = sorted(v["view"].unique())
    frames = []
    for view in views:
        parts = []
        for tag, path in [("real photo", out_root / "w4" / "gt" / f"{view}.png"),
                          (LABEL["nerfacto"], out_root / "w4" / "nerfacto" / f"s{seed}" / f"{view}.png"),
                          (LABEL["splatfacto"], out_root / "w4" / "splatfacto" / f"s{seed}" / f"{view}.png")]:
            if not path.exists():
                break
            im = Image.open(path).convert("RGB")
            ImageDraw.Draw(im).text((10, 8), tag, fill=(255, 255, 0))
            parts.append(np.asarray(im))
        if len(parts) == 3:
            frames.append(np.concatenate(parts, axis=1))
    if not frames:
        return
    h, w = frames[0].shape[:2]
    frames = [f[: h - h % 2, : w - w % 2] for f in frames]
    target = fig_dir / "w4_side_by_side.mp4"
    fps = 1.0 / seconds_per_view
    try:
        import imageio.v2 as imageio

        imageio.mimsave(target, frames, fps=fps, codec="libx264", quality=7)
    except Exception as e:  # no ffmpeg backend: fall back to OpenCV, then to a GIF
        try:
            import cv2

            wr = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
            for f in frames:
                wr.write(f[:, :, ::-1])
            wr.release()
        except Exception:
            target = fig_dir / "w4_side_by_side.gif"
            Image.fromarray(frames[0]).save(target, save_all=True, append_images=[Image.fromarray(f) for f in frames[1:]],
                                            duration=int(seconds_per_view * 1000), loop=0)
        print("clip written with fallback writer:", type(e).__name__)
    print("wrote", target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default=str(ROOT / "report_assets" / "w4"))
    ap.add_argument("--out-root", default=str(ROOT / "outputs"))
    ap.add_argument("--fig-dir", default=str(ROOT / "outputs" / "w4_figures"))
    ap.add_argument("--units-per-hr", type=float, default=None, help="Colab compute units per hour for this GPU")
    ap.add_argument("--seed", type=int, default=0, help="which training seed to show in the pictures")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-photos", action="store_true", help="skip the grid and clip (they contain the room)")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    df = frame(Path(args.results))
    v = view_table(df)
    summ, pair, run = model_summary(v), paired(v), run_table(df)
    spread, cost = seed_spread(df), cost_table(run, args.units_per_hr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {"w4_summary": summ, "w4_paired": pair, "w4_run_level": run, "w4_seed_spread": spread, "w4_cost": cost}
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)
        (out / f"{name}.md").write_text(t.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")

    pd.set_option("display.width", 170, "display.max_columns", 20)
    for title, t in [("held-out views, per model", summ), ("3DGS minus Nerfacto, paired over photos", pair),
                     ("per run (time, memory, size, speed)", run), ("PSNR spread between training seeds", spread),
                     ("training cost", cost)]:
        print(f"\n== {title} ==\n", t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.units_per_hr is None:
        print("\n(pass --units-per-hr to price the training; read it from Colab's Resources panel)")

    figures(v, summ, run, out, args.show)
    if not args.no_photos:
        qualitative(v, Path(args.out_root), Path(args.fig_dir), args.show, args.seed)
    print(f"\nwrote tables and figures to {out}")


if __name__ == "__main__":
    main()
