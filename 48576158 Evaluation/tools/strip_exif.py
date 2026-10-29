"""Copy images with all metadata (EXIF, GPS, thumbnails) removed.

    python tools/strip_exif.py data/raw data/public [--max-side 1600]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps

EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--max-side", type=int, default=1600)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    n = 0
    for f in sorted(src.rglob("*")):
        if f.suffix.lower() not in EXT:
            continue
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")  # apply rotation, then drop metadata
        img.thumbnail((args.max_side, args.max_side))
        clean = Image.new("RGB", img.size)
        clean.paste(img)  # new image object carries no metadata
        out = dst / f.relative_to(src).with_suffix(".jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        clean.save(out, quality=92)
        n += 1
    print(f"wrote {n} images to {dst}")


if __name__ == "__main__":
    main()
