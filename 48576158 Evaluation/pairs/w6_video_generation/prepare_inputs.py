"""Centre-crop the chosen listing photos to 16:9 and save them as PNGs that both models receive unchanged.

    python prepare_inputs.py --src /content/drive/MyDrive/comp8410_a1 --out /content/w6_work/inputs

--src is searched recursively for <item_id>.png (or .jpg), excluding any path with an 'outputs' folder (W2 edited results). Use --src <BASE>/w2_listing.
Writes a contact sheet (inputs_sheet.jpg) next to the inputs for a quick look.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from w6_common import INPUT_DIR, item_ids  # noqa: E402

TARGET = 16 / 9


def crop_16x9(img: Image.Image, out_w: int = 1280) -> Image.Image:
    w, h = img.size
    if w / h > TARGET:  # too wide: trim the sides
        nw = int(round(h * TARGET))
        box = ((w - nw) // 2, 0, (w - nw) // 2 + nw, h)
    else:               # too tall: trim top and bottom (listing photos are 4:3)
        nh = int(round(w / TARGET))
        box = (0, (h - nh) // 2, w, (h - nh) // 2 + nh)
    out_h = int(round(out_w / TARGET))
    return img.crop(box).resize((out_w, out_h), Image.LANCZOS)


def find(src: Path, item: str) -> Path | None:
    """Original listing photo only. Drive also holds W2 EDITED outputs with the same file names (outputs/w2/...):
    those are excluded, otherwise the models would be fed relit dusk/golden-hour images."""
    for ext in ("jpg", "jpeg", "png"):
        hits = [h for h in sorted(src.rglob(f"{item}.{ext}")) if "outputs" not in {p.lower() for p in h.parts}]
        if hits:
            return hits[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=INPUT_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tiles, missing = [], []
    for item in item_ids():
        p = find(args.src, item)
        if p is None:
            missing.append(item)
            continue
        im = crop_16x9(Image.open(p).convert("RGB"))
        im.save(args.out / f"{item}.png")
        tiles.append(im.resize((320, 180)))
        print(f"{item}: {p} -> {im.size}")
    if missing:
        print("MISSING:", missing)
        return 2
    sheet = Image.new("RGB", (320 * 3, 180 * ((len(tiles) + 2) // 3)))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 3) * 320, (i // 3) * 180))
    sheet.save(args.out / "inputs_sheet.jpg")
    print(f"{len(tiles)} inputs ready in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
