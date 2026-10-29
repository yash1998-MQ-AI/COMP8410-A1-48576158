"""Aggregate result rows into tables and figures for the report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import bootstrap_ci
from .schema import load


def to_frame(results_path: str | Path) -> pd.DataFrame:
    rows = load(results_path)
    return pd.DataFrame(
        [
            {"pair": r.pair, "model": r.model, "item_id": r.item_id, "metric": r.metric, "value": r.value, "seed": r.seed}
            for r in rows
        ]
    )


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per (pair, model, metric): n, mean and 95% bootstrap CI."""
    out = []
    for (pair, model, metric), g in df.groupby(["pair", "model", "metric"]):
        mean, lo, hi = bootstrap_ci(g["value"].tolist())
        out.append(
            {"pair": pair, "model": model, "metric": metric, "n": int(g["value"].notna().sum()), "mean": mean, "ci_lo": lo, "ci_hi": hi}
        )
    return pd.DataFrame(out)


def to_markdown(summary: pd.DataFrame, digits: int = 3) -> str:
    lines = ["| pair | model | metric | n | mean | 95% CI |", "|---|---|---|---|---|---|"]
    for r in summary.itertuples():
        lines.append(f"| {r.pair} | {r.model} | {r.metric} | {r.n} | {r.mean:.{digits}f} | [{r.ci_lo:.{digits}f}, {r.ci_hi:.{digits}f}] |")
    return "\n".join(lines)


def plot_metric(summary: pd.DataFrame, pair: str, metric: str, out_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summary[(summary.pair == pair) & (summary.metric == metric)].sort_values("model")
    if s.empty:
        raise ValueError(f"no rows for pair={pair} metric={metric}")
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    err = [s["mean"] - s["ci_lo"], s["ci_hi"] - s["mean"]]
    ax.bar(s["model"], s["mean"], yerr=err, capsize=4)
    ax.set_ylabel(metric)
    ax.set_title(f"{pair}: {metric} (mean, 95% CI, n={int(s['n'].min())})")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_pair_outputs(results_path: str | Path, out_dir: str | Path) -> pd.DataFrame:
    """Write summary.csv, summary.md and one PNG per metric for a results file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = to_frame(results_path)
    if df.empty:
        raise ValueError(f"no results in {results_path}")
    summary = summarise(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    for (pair, metric), _ in summary.groupby(["pair", "metric"]):
        plot_metric(summary, pair, metric, out_dir / f"{pair}_{metric}.png")
    return summary
