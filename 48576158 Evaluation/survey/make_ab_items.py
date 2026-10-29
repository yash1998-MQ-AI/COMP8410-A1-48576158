"""Build the blind A/B user-study items.

Reads generated outputs laid out as   outputs/<pair>/<model>/<item_id>.<ext>
and writes, for two form versions (V1, V2), which model appears on the left.
V2 is V1 mirrored, so sending half the raters each version balances position
bias. The answer key (answer_key.json) must NOT be shared with raters.

    python survey/make_ab_items.py --outputs outputs --per-pair 2 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--per-pair", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="survey")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.outputs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = []
    for pair_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        models = sorted(m.name for m in pair_dir.iterdir() if m.is_dir())
        if len(models) != 2:
            print(f"skip {pair_dir.name}: expected 2 model folders, found {models}")
            continue
        a_files = {f.stem: f for f in (pair_dir / models[0]).iterdir() if f.is_file()}
        b_files = {f.stem: f for f in (pair_dir / models[1]).iterdir() if f.is_file()}
        common = sorted(set(a_files) & set(b_files))
        chosen = rng.sample(common, min(args.per_pair, len(common)))
        for item in chosen:
            flip = rng.random() < 0.5
            left, right = (models[1], models[0]) if flip else (models[0], models[1])
            items.append({"pair": pair_dir.name, "item_id": item, "v1_left": left, "v1_right": right})

    rng.shuffle(items)  # mix pairs so raters do not see them grouped
    with (out / "survey_items.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["q", "pair", "item_id", "version", "left_file", "right_file"])
        for q, it in enumerate(items, 1):
            for version, (l, r) in {"V1": (it["v1_left"], it["v1_right"]), "V2": (it["v1_right"], it["v1_left"])}.items():
                ext_l = next((root / it["pair"] / l).glob(it["item_id"] + ".*")).suffix
                ext_r = next((root / it["pair"] / r).glob(it["item_id"] + ".*")).suffix
                w.writerow([q, it["pair"], it["item_id"], version, f"{it['pair']}/{l}/{it['item_id']}{ext_l}", f"{it['pair']}/{r}/{it['item_id']}{ext_r}"])

    key = {str(q): {"pair": it["pair"], "item_id": it["item_id"], "V1": {"left": it["v1_left"], "right": it["v1_right"]},
                    "V2": {"left": it["v1_right"], "right": it["v1_left"]}} for q, it in enumerate(items, 1)}
    (out / "answer_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"wrote {len(items)} questions x 2 versions to {out}")


if __name__ == "__main__":
    main()
