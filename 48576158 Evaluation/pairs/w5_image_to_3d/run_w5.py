"""Week 5 input preparation: background removal, shared by both models.

    python pairs/w5_image_to_3d/run_w5.py prep            # needs data/processed/w5_setC/<item>.jpg (prepare_inputs.py)

Both models are given the same RGBA PNG (object cut out, cropped to a square with a margin). The cut-out also
provides the photo silhouette used for the IoU score. A hand-made mask in data/masks/<item>.png (white = object)
overrides the automatic one; use it when the contact sheet shows rembg failing (glass, reflective surfaces).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from w5_common import INPUT_DIR, ROOT, VIEWS_ROOT, item_ids  # noqa: E402


def largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def cut_out(img: Image.Image, session) -> np.ndarray:
    from rembg import remove

    out = remove(img.convert("RGB"), session=session, only_mask=True)
    return np.asarray(out.convert("L")) > 127


def square_rgba(img: Image.Image, mask: np.ndarray, margin: float = 0.1, max_side: int = 1024) -> Image.Image:
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(round(max(y1 - y0, x1 - x0) * (1 + 2 * margin)))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    top, left = cy - side // 2, cx - side // 2
    rgba = np.zeros((side, side, 4), dtype=np.uint8)
    src = np.asarray(img.convert("RGB"))
    sy0, sx0 = max(top, 0), max(left, 0)
    sy1, sx1 = min(top + side, src.shape[0]), min(left + side, src.shape[1])
    rgba[sy0 - top: sy1 - top, sx0 - left: sx1 - left, :3] = src[sy0:sy1, sx0:sx1]
    rgba[sy0 - top: sy1 - top, sx0 - left: sx1 - left, 3] = mask[sy0:sy1, sx0:sx1].astype(np.uint8) * 255
    out = Image.fromarray(rgba, "RGBA")
    if side > max_side:
        out = out.resize((max_side, max_side), Image.LANCZOS)
    return out


def prep(args) -> None:
    from rembg import new_session

    try:
        session = new_session("birefnet-general")
    except Exception as exc:
        print("birefnet-general unavailable, falling back to u2net:", exc)
        session = new_session("u2net")
    args.out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for item in item_ids():
        src = args.src / f"{item}.jpg"
        if not src.exists():
            print(f"missing {src}; run prepare_inputs.py first")
            continue
        img = Image.open(src)
        override = ROOT / "data" / "masks" / f"{item}.png"
        if override.exists():
            mask = np.asarray(Image.open(override).convert("L").resize(img.size)) > 127
            how = "manual mask"
        else:
            mask = largest_component(cut_out(img, session))
            how = "rembg"
        frac = float(mask.mean())
        rgba = square_rgba(img, mask)
        rgba.save(args.out / f"{item}.png")
        Image.fromarray(np.asarray(rgba)[..., 3]).save(args.out / f"{item}_mask.png")
        print(f"{item}: {how}, object covers {frac:.0%} of the photo")
        comp = Image.new("RGB", rgba.size, (255, 255, 255))
        comp.paste(rgba, mask=rgba.split()[3])
        tiles.append(comp.resize((256, 256)))
    if tiles:
        cols = 5
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 256, rows * 256), (255, 255, 255))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * 256, (i // cols) * 256))
        VIEWS_ROOT.mkdir(parents=True, exist_ok=True)
        sheet.save(VIEWS_ROOT / "inputs_sheet.jpg", quality=90)
        print("wrote", VIEWS_ROOT / "inputs_sheet.jpg", "- check every cut-out before generating")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep"])
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "processed" / "w5_setC")
    ap.add_argument("--out", type=Path, default=INPUT_DIR)
    args = ap.parse_args()
    prep(args)


if __name__ == "__main__":
    main()
