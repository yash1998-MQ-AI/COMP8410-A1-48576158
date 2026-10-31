"""Week 5: build the model inputs from the private raw photos in data/raw/Set C.

    python pairs/w5_image_to_3d/prepare_inputs.py [--out data/processed/w5_setC] [--max-side 2048]

For each row of set_c_manifest.csv: apply the EXIF orientation, drop ALL metadata (the phone wrote GPS
coordinates into every file), convert to RGB and save as <item_id>.jpg with the longer side at most
--max-side. The raw files are never modified. Upload the output folder to Drive (comp8410_a1/w5_data/setC/).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw" / "Set C"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "w5_setC"))
    ap.add_argument("--max-side", type=int, default=2048)
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(HERE / "set_c_manifest.csv", encoding="utf-8")))
    missing = [r["original_file"] for r in rows if not (raw / r["original_file"]).exists()]
    if missing:
        raise SystemExit(f"missing raw files: {missing}")
    for r in rows:
        with Image.open(raw / r["original_file"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
            clean = Image.frombytes("RGB", im.size, im.tobytes())  # fresh object: no EXIF, ICC or XMP carried over
            clean.save(out / f"{r['item_id']}.jpg", quality=95)
        print(f"{r['item_id']:16s} {clean.size}", flush=True)
    print(f"wrote {len(rows)} images to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
