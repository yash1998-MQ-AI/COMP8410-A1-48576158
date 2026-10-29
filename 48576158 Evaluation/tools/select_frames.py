"""Screen a photo capture for 3D reconstruction and write a cleaned copy.

Flags (and drops from the cleaned copy):
  size    - resolution differs from the most common one (mixed cameras/modes break a
            single-camera reconstruction)
  blur    - sharpness well below the capture median
  colour  - exposure or white balance far from the capture median (auto-exposure drift)
  dup     - almost identical to the previous kept frame (no new viewpoint)

Originals are never modified. The cleaned copy has metadata (EXIF/GPS) removed,
is auto-rotated and resized. A manifest records every decision.

    python tools/select_frames.py --src "data/raw/Set A" --dst data/processed/w4_room
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXT = {".jpg", ".jpeg", ".png"}


def stats(path: Path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    size = im.size
    small = np.asarray(im.resize((size[0] // 4, size[1] // 4)), dtype=np.float32)
    g = small.mean(axis=2)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    thumb = np.asarray(im.convert("L").resize((64, 48)), dtype=np.float32)
    return size, float(lap.var()), small.reshape(-1, 3).mean(axis=0), thumb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--blur-rel", type=float, default=0.4, help="blur if sharpness < this x median")
    ap.add_argument("--colour-k", type=float, default=3.5, help="robust z-score limit for mean R/G/B")
    ap.add_argument("--dup-diff", type=float, default=1.5, help="mean abs diff (0-255) below which a frame is a duplicate")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in EXT)
    if not files:
        raise SystemExit(f"no images in {src}")

    info = {f: stats(f) for f in files}
    common = Counter(v[0] for v in info.values()).most_common(1)[0][0]
    sharp = np.array([v[1] for v in info.values() if v[0] == common])
    rgb = np.array([v[2] for v in info.values() if v[0] == common])
    med_sharp = float(np.median(sharp))
    med_rgb = np.median(rgb, axis=0)
    mad = np.median(np.abs(rgb - med_rgb), axis=0) * 1.4826 + 1e-6

    rows, last_thumb = [], None
    for f in files:
        size, sh, mean_rgb, thumb = info[f]
        reason = ""
        if size != common:
            reason = "size"
        elif sh < args.blur_rel * med_sharp:
            reason = "blur"
        elif np.any(np.abs(mean_rgb - med_rgb) / mad > args.colour_k):
            reason = "colour"
        elif last_thumb is not None and float(np.mean(np.abs(thumb - last_thumb))) < args.dup_diff:
            reason = "dup"
        if not reason:
            last_thumb = thumb
        rows.append({"file": f.name, "width": size[0], "height": size[1], "sharpness": round(sh, 1),
                     "mean_r": round(float(mean_rgb[0]), 1), "mean_g": round(float(mean_rgb[1]), 1),
                     "mean_b": round(float(mean_rgb[2]), 1), "dropped_for": reason})

    kept = [r for r in rows if not r["dropped_for"]]
    print(f"{len(files)} images; most common size {common}; median sharpness {med_sharp:.0f}")
    print("dropped:", dict(Counter(r["dropped_for"] for r in rows if r["dropped_for"])), "| kept:", len(kept))

    if args.dry_run:
        return
    dst.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(kept):
        r["clean_name"] = f"{i:04d}.jpg"
        im = ImageOps.exif_transpose(Image.open(src / r["file"])).convert("RGB")
        im.thumbnail((args.max_side, args.max_side))
        clean = Image.new("RGB", im.size)
        clean.paste(im)  # fresh image object: no EXIF/GPS carried over
        clean.save(dst / r["clean_name"], quality=94)
    with (dst / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "clean_name", "width", "height", "sharpness", "mean_r", "mean_g", "mean_b", "dropped_for"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(kept)} cleaned images and manifest.csv to {dst}")


if __name__ == "__main__":
    main()
