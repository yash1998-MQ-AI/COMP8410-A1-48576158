"""Week 8 tables and figures from results/w8.jsonl. python report_w8.py --show

Success rate per agent with a Wilson interval (episodes pooled over tasks and seeds); the qwen_dom vs uitars_vis difference
is a paired bootstrap over tasks (each task's success rate averaged over its seeds). Injection: attack-followed rate and the
drop in success against the clean runs of the same tasks. Only tables and plots are written (no screenshots).
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
from w8_common import AGENTS, LABEL, RESULTS, ROOT  # noqa: E402

COLOUR = {"rule": "#8d99ae", "qwen_dom": "#1b6ca8", "uitars_vis": "#d1495b"}


def frame(path: Path) -> pd.DataFrame:
    rows = []
    for r in load(path):
        if r.item_id == "all":
            continue
        cond, task = r.item_id.split("/", 1)
        rows.append({"agent": r.model, "cond": cond, "task": task, "seed": r.seed, "metric": r.metric, "value": r.value})
    return pd.DataFrame(rows)


def success_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cond in ("clean", "inject"):
        for a in AGENTS:
            v = df[(df.agent == a) & (df.cond == cond) & (df.metric == "success")].value
            if v.empty:
                continue
            p, lo, hi = M.wilson_ci(int(v.sum()), len(v))
            r = {"cond": cond, "agent": a, "episodes": len(v), "successes": int(v.sum()), "success_rate": p, "ci_lo": lo, "ci_hi": hi}
            for m in ("n_steps", "invalid_rate", "latency_per_step_s", "latency_s", "gen_tokens"):
                x = df[(df.agent == a) & (df.cond == cond) & (df.metric == m)].value
                r[m] = float(x.mean()) if len(x) else np.nan
            if cond == "inject":
                x = df[(df.agent == a) & (df.cond == "inject") & (df.metric == "injection_followed")].value
                r["attack_followed_rate"] = float(x.mean()) if len(x) else np.nan
            out.append(r)
    return pd.DataFrame(out)


def per_task(df: pd.DataFrame) -> pd.DataFrame:
    s = df[(df.metric == "success") & (df.cond == "clean")]
    return s.groupby(["task", "agent"]).value.mean().unstack("agent").reindex(columns=[a for a in AGENTS if a in set(s.agent)])


def paired(df: pd.DataFrame, a="uitars_vis", b="qwen_dom") -> dict:
    t = per_task(df)
    if a not in t or b not in t:
        return {}
    t = t[[a, b]].dropna()
    d, lo, hi = M.paired_bootstrap_diff(t[a].to_numpy(), t[b].to_numpy())
    return {"a": a, "b": b, "n_tasks": len(t), "diff_a_minus_b": d, "lo": lo, "hi": hi,
            "verdict": f"{a} better" if lo > 0 else (f"{b} better" if hi < 0 else "no clear difference")}


def injection_drop(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    inj_tasks = set(df[df.cond == "inject"].task)
    for a in AGENTS:
        c = df[(df.agent == a) & (df.cond == "clean") & (df.metric == "success") & (df.task.isin(inj_tasks))].value
        i = df[(df.agent == a) & (df.cond == "inject") & (df.metric == "success")].value
        if len(c) and len(i):
            rows.append({"agent": a, "clean_success_same_tasks": c.mean(), "inject_success": i.mean(), "drop": c.mean() - i.mean()})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=ROOT / "report_assets" / "w8")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)
    df = frame(args.results)
    if df.empty:
        print("no rows in", args.results)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    st = success_table(df)
    st.to_csv(args.out / "w8_success.csv", index=False)
    pt = per_task(df)
    pt.to_csv(args.out / "w8_per_task.csv")
    pr = pd.DataFrame([paired(df)])
    pr.to_csv(args.out / "w8_paired.csv", index=False)
    ij = injection_drop(df)
    ij.to_csv(args.out / "w8_injection.csv", index=False)

    clean = st[st.cond == "clean"]
    if not clean.empty:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for i, r in enumerate(clean.itertuples()):
            ax.bar(i, r.success_rate, color=COLOUR[r.agent], alpha=0.85)
            ax.errorbar(i, r.success_rate, yerr=[[r.success_rate - r.ci_lo], [r.ci_hi - r.success_rate]], color="k", capsize=3)
        ax.set_xticks(range(len(clean)), [LABEL[a].split(",")[0] for a in clean.agent], rotation=15, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("success rate (Wilson 95%)")
        fig.tight_layout()
        fig.savefig(args.out / "w8_success.png", dpi=150)
        plt.close(fig)
    (args.out / "summary.md").write_text("## Success\n" + st.round(3).to_markdown(index=False) + "\n\n## Per task (clean)\n" + pt.round(2).to_markdown() +
                                          "\n\n## Paired (tasks)\n" + pr.round(3).to_markdown(index=False) + "\n\n## Injection\n" + ij.round(3).to_markdown(index=False) + "\n", encoding="utf-8")
    if args.show:
        print(st.round(3).to_string(index=False))
        print(pt.round(2).to_string())
        print(pr.round(3).to_string(index=False))
        print(ij.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
