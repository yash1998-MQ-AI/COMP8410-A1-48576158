"""Command line: python -m a1eval summarize results/w2.jsonl --out report_assets/w2"""

from __future__ import annotations

import argparse

from .aggregate import write_pair_outputs
from .runlog import total_cost


def main() -> None:
    p = argparse.ArgumentParser(prog="a1eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summarize", help="results JSONL -> summary.csv/.md + figures")
    s.add_argument("results")
    s.add_argument("--out", required=True)

    c = sub.add_parser("cost", help="print total GPU spend from the cost ledger")
    c.add_argument("ledger", nargs="?", default="cost_log.csv")

    args = p.parse_args()
    if args.cmd == "summarize":
        summary = write_pair_outputs(args.results, args.out)
        print(summary.to_string(index=False))
    elif args.cmd == "cost":
        print(f"total spend: A${total_cost(args.ledger):.2f} (cap A$60)")


if __name__ == "__main__":
    main()
