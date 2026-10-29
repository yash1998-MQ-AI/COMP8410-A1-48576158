"""Week 3: the test list. 4 identities x 4 listing-material prompts = 16 items (fixed before any run).

Reference photos live in data/processed/w3_refs/<identity>.jpg (git-ignored, private).
  me                    a photo of the student (consent given by the student)
  syn_a, syn_b, syn_c   synthetic faces made by make_synthetic_refs.py

    python pairs/w3_identity_preservation/prepare_items.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ITEMS = Path(__file__).with_name("items.csv")
IDENTITIES = ["me", "syn_a", "syn_b", "syn_c"]
PROMPTS = {
    "headshot": "professional headshot photo of a real estate agent in a navy blazer, soft studio lighting, plain light grey background, sharp focus",
    "exterior": "photo of a real estate agent standing in front of a modern two-storey Australian suburban house, sunny day, smiling, upper body",
    "interior": "photo of a real estate agent in a bright modern living room holding house keys, natural window light, smiling",
    "open_home": "photo of a real estate agent beside an open-for-inspection sign on a footpath, overcast day, friendly expression",
}
NEGATIVE = "lowres, blurry, deformed, extra fingers, bad anatomy, watermark, text, cartoon, painting"


def main() -> None:
    rows = []
    for ident in IDENTITIES:
        for key, prompt in PROMPTS.items():
            rows.append({"item_id": f"{ident}_{key}", "identity": ident, "scenario": key,
                         "ref": f"data/processed/w3_refs/{ident}.jpg", "prompt": prompt, "negative": NEGATIVE})
    with ITEMS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    missing = [r["ref"] for r in rows if not (ROOT / r["ref"]).exists()]
    print(f"wrote {len(rows)} items to {ITEMS.name}")
    if missing:
        print("missing reference images:", sorted(set(missing)))


if __name__ == "__main__":
    main()
