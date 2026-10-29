"""Build the Week 2 test set: 15 listing-style photos, one edit prompt each.

Reads private captures from data/raw (never modified), writes metadata-free,
resized copies to data/processed/w2_listing/ and the item table items.csv.
Edit SELECTION below to change which photos are used; the rule is to cover
different rooms/viewpoints and keep a few hard cases (sun flare, mirror).

    python pairs/w2_image_editing/prepare_items.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "w2_listing"
ITEMS = Path(__file__).with_name("items.csv")
LONG_SIDE = 1536

# Scope: lighting, time of day and weather instruction edits only.
# No staging, clutter removal, object insertion or background fill.
KEEP = "Keep the layout, furniture, objects and everything else exactly the same."
EDITS = {
    "golden_hour": (
        f"Change the lighting to warm golden-hour sunset light. {KEEP}",
        "a real estate photo in warm golden hour sunset light",
    ),
    "overcast": (
        f"Change the weather to a soft overcast day with even, diffuse daylight. {KEEP}",
        "a real estate photo on a soft overcast day with diffuse daylight",
    ),
    "twilight": (
        f"Change the time of day to dusk with a deep blue twilight sky and warm lamps switched on. {KEEP}",
        "a real estate photo at dusk with blue twilight and warm lights",
    ),
    "brighten": (
        f"Make the scene brighter with clean natural daylight and balanced exposure. {KEEP}",
        "a bright, evenly exposed real estate photo in clean daylight",
    ),
}

# (set folder, file name or index into the sorted folder, item id, edit type, note)
SELECTION = [
    ("Set B", "IMG_20261029_164444258.jpg", "bedroom_desk", "golden_hour", ""),
    ("Set B", "IMG_20261029_164448153.jpg", "bedroom_bed", "overcast", ""),
    ("Set B", "IMG_20261029_164456853.jpg", "bedroom_door", "twilight", ""),
    ("Set B", "IMG_20261029_164515294.jpg", "bedroom_frame", "brighten", ""),
    ("Set B", "IMG_20261029_164623054.jpg", "street_house", "golden_hour", "exterior"),
    ("Set B", "IMG_20261029_164630630.jpg", "street_flare", "overcast", "hard case: sun flare"),
    ("Set B", "IMG_20261029_164636767.jpg", "street_hedge", "twilight", "exterior"),
    ("Set B", "IMG_20261029_164710091.jpg", "kitchen_a", "brighten", ""),
    ("Set B", "IMG_20261029_164711626.jpg", "kitchen_b", "twilight", ""),
    ("Set A", 9, "main_bed_wide", "twilight", ""),
    ("Set A", 24, "main_balcony_door", "overcast", "bright window"),
    ("Set A", 44, "main_bed_window", "golden_hour", ""),
    ("Set A", 66, "main_fan_blind", "brighten", ""),
    ("Set A", 90, "main_balcony_view", "twilight", "bright window"),
    ("Set A", 58, "main_mirror_robe", "golden_hour", "hard case: mirror"),
]


def resolve(folder: str, ref) -> Path:
    d = RAW / folder
    if isinstance(ref, int):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        return files[ref]
    return d / ref


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder, ref, item_id, edit, note in SELECTION:
        src = resolve(folder, ref)
        im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        im.thumbnail((LONG_SIDE, LONG_SIDE))
        clean = Image.new("RGB", im.size)
        clean.paste(im)  # no EXIF/GPS carried over
        dest = OUT / f"{item_id}.jpg"
        clean.save(dest, quality=94)
        prompt, target = EDITS[edit]
        rows.append(
            {
                "item_id": item_id,
                "source": dest.relative_to(ROOT).as_posix(),
                "origin": f"{folder}/{src.name}",
                "edit_type": edit,
                "prompt": prompt,
                "target_text": target,
                "note": note,
            }
        )
    with ITEMS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} images to {OUT} and {ITEMS.name}")


if __name__ == "__main__":
    main()
