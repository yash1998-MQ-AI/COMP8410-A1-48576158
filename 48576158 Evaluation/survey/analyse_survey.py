"""Turn survey responses into per-pair preference rates with Wilson intervals.

responses.csv columns: respondent_id, version (V1|V2), q1, q2, ... where each
qN is "left" or "right" (Google Forms: map the option text to these values
before export). Ties/skips: leave the cell empty; they are reported, not counted.

    python survey/analyse_survey.py --responses survey/responses.csv --key survey/answer_key.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a1eval.metrics import wilson_ci  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", default="survey/summary.csv")
    args = ap.parse_args()

    key = json.load(open(args.key, encoding="utf-8"))
    wins = defaultdict(lambda: defaultdict(int))  # pair -> model -> wins
    totals = defaultdict(int)  # pair -> counted votes
    skipped = defaultdict(int)
    raters = set()

    with open(args.responses, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raters.add(row["respondent_id"])
            ver = row["version"]
            for q, info in key.items():
                ans = (row.get(f"q{q}") or "").strip().lower()
                if ans not in ("left", "right"):
                    skipped[info["pair"]] += 1
                    continue
                winner = info[ver][ans]
                wins[info["pair"]][winner] += 1
                totals[info["pair"]] += 1

    rows = []
    for pair in sorted(totals):
        models = sorted({m for q in key.values() if q["pair"] == pair for m in (q["V1"]["left"], q["V1"]["right"])})
        for m in models:
            k = wins[pair][m]
            p, lo, hi = wilson_ci(k, totals[pair])
            rows.append({"pair": pair, "model": m, "votes_for": k, "votes_total": totals[pair],
                         "preference": round(p, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "skipped": skipped[pair]})
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["pair"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(raters)} raters")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
