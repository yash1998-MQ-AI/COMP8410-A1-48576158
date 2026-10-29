"""Week 2: score every generated edit and append metric rows to results/w2.jsonl.

Metrics (all per item):
  edge_ssim  structure preserved: SSIM of Sobel edge maps, source vs edit (1 = same layout)
  clip_i     content retained: CLIP image-image cosine, source vs edit
  clip_gain  edit achieved: CLIP text-image similarity of the edit to the target description
             minus the same value for the unedited source (> 0 = moved toward the target)

    python pairs/w2_image_editing/score_w2.py
    python pairs/w2_image_editing/score_w2.py --metrics edge_ssim      # CPU only
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from a1eval import Record, append, load  # noqa: E402
from a1eval import metrics as M  # noqa: E402

ALL = ["edge_ssim", "clip_i", "clip_gain"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(Path(__file__).with_name("items.csv")))
    ap.add_argument("--out-root", default=str(ROOT / "outputs"))
    ap.add_argument("--results", default=str(ROOT / "results" / "w2.jsonl"))
    ap.add_argument("--metrics", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--models", nargs="*", help="default: every folder under outputs/w2")
    args = ap.parse_args()

    items = list(csv.DictReader(open(args.items, encoding="utf-8")))
    root = Path(args.out_root) / "w2"
    models = args.models or sorted(p.name for p in root.iterdir() if p.is_dir())
    have = {(r.model, r.item_id, r.metric) for r in load(args.results)}
    n = 0
    for model in models:
        for it in items:
            edited = root / model / f"{it['item_id']}.png"
            if not edited.exists():
                continue
            todo = [m for m in args.metrics if (model, it["item_id"], m) not in have]
            if not todo:
                continue
            src_path = Path(it["source"])
            src = M.load_image(src_path if src_path.is_absolute() else ROOT / src_path)
            ed = M.load_image(edited)
            for m in todo:
                if m == "edge_ssim":
                    v = M.edge_ssim(src, ed)
                elif m == "clip_i":
                    v = M.clip_image_image(src, ed)
                else:
                    v = M.clip_text_image(ed, it["target_text"]) - M.clip_text_image(src, it["target_text"])
                append(args.results, Record("w2", model, it["item_id"], m, float(v), None, {"edit_type": it["edit_type"]}))
                n += 1
    print(f"appended {n} metric rows to {args.results}")


if __name__ == "__main__":
    main()
