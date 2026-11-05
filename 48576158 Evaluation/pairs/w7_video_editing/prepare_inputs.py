"""Week 7 inputs: any folder of clips -> 832x480 mp4s at a 4n+1 frame count, plus items.csv.

    python prepare_inputs.py --src /content/drive/MyDrive/comp8410_a1/w7_data/setD
    python prepare_inputs.py --src /content/a1/.../outputs/w6/hunyuanvideo15   # fallback: Pair 5 clips

Writes INPUT_DIR/<item_id>.mp4 (the identical file fed to both models) and items.csv with one
edit instruction and one edit box per clip, cycled from EDIT_BANK. Edit items.csv afterwards
if you want different instructions; nothing downstream regenerates it unless --force is given.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from w7_common import (CLIP_ORDER, EDIT_BANK, EDIT_PLAN, FPS, HEIGHT, INPUT_DIR, MANIFEST,  # noqa: E402
                       MAX_FRAMES, START_FRAC, WIDTH, vae_frames)

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}


def stem_key(p: Path) -> str:
    return p.stem.strip().lower()


def find_clips(src: Path) -> list[Path]:
    """Clips under src. If CLIP_ORDER names any of them, use exactly those, in that order."""
    if src.is_file():
        return [src]
    found = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in VIDEO_EXT and p.is_file()]
    by_key = {stem_key(p): p for p in found}
    chosen = [by_key[k] for k in CLIP_ORDER if k in by_key]
    if chosen:
        skipped = [p.name for p in found if stem_key(p) not in set(CLIP_ORDER)]
        if skipped:
            print(f"not in CLIP_ORDER, skipping: {', '.join(skipped)}")
        return chosen
    return found


def edit_for(path: Path, index: int):
    """The planned edit for this clip, or the next family from the fallback bank."""
    k = stem_key(path)
    if k in EDIT_PLAN:
        return EDIT_PLAN[k]
    print(f"  (no EDIT_PLAN entry for '{k}' - falling back to the bank; check the instruction makes sense)")
    return EDIT_BANK[index % len(EDIT_BANK)]


def slug(p: Path, used: set[str]) -> str:
    s = "".join(c if (c.isalnum() or c in "-_") else "_" for c in p.stem).strip("_").lower() or "clip"
    base, k = s, 2
    while s in used:
        s, k = f"{base}_{k}", k + 1
    used.add(s)
    return s


def read_frames(path: Path, start: int = 0, count: int | None = None,
                transform=None) -> tuple[list[np.ndarray], float]:
    """Frames [start, start+count) as uint8 RGB, plus fps.

    Streams and stops early: a 1080p clip held whole in memory is several GB, which kills a
    Colab runtime. `transform` is applied per frame as it is read, so the resized frames are
    the only thing retained.
    """
    keep = (lambda f: f) if transform is None else transform
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            for _ in range(start):                      # grab() decodes without converting
                if not cap.grab():
                    break
            frames = []
            while count is None or len(frames) < count:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(keep(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            cap.release()
            if frames:
                return frames, float(fps)
            cap.release()
    except ImportError:
        pass
    import imageio.v3 as iio

    meta = iio.immeta(str(path))
    frames = []
    for i, f in enumerate(iio.imiter(str(path))):
        if i < start:
            continue
        if count is not None and len(frames) >= count:
            break
        frames.append(keep(np.asarray(f)))
    return frames, float(meta.get("fps", 0.0))


def frame_count(path: Path) -> int:
    """Total frames without decoding them all."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if n > 0:
            return n
    except ImportError:
        pass
    import imageio.v3 as iio

    try:
        meta = iio.immeta(str(path))
        d, fps = float(meta.get("duration", 0)), float(meta.get("fps", 0))
        if d and fps:
            return int(d * fps)
    except Exception:
        pass
    return sum(1 for _ in iio.imiter(str(path)))


def cover_resize(frame: np.ndarray, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Scale to cover WIDTHxHEIGHT then centre-crop. Keeps aspect ratio, never letterboxes."""
    im = Image.fromarray(frame)
    w, h = im.size
    s = max(width / w, height / h)
    im = im.resize((max(width, int(round(w * s))), max(height, int(round(h * s)))), Image.LANCZOS)
    w, h = im.size
    l, t = (w - width) // 2, (h - height) // 2
    return np.asarray(im.crop((l, t, l + width, t + height)))


def write_mp4(frames: list[np.ndarray], path: Path, fps: float = FPS) -> None:
    import imageio.v3 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(path), np.stack(frames), fps=fps, codec="libx264",
                output_params=["-crf", "16", "-preset", "veryfast"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="folder (or single file) of input clips")
    ap.add_argument("--dest", type=Path, default=INPUT_DIR)
    ap.add_argument("--limit", type=int, default=6, help="max clips to prepare (budget control)")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--start-frac", type=float, default=START_FRAC,
                    help="skip this fraction of each clip before taking frames (default %(default)s)")
    ap.add_argument("--force", action="store_true", help="overwrite items.csv if it exists")
    args = ap.parse_args(argv)

    clips = find_clips(args.src)
    if not clips:
        print(f"no video files found under {args.src}", file=sys.stderr)
        return 2
    clips = clips[: args.limit]

    args.dest.mkdir(parents=True, exist_ok=True)
    rows, used = [], set()
    for k, src in enumerate(clips):
        total = frame_count(src)
        if total < 5:
            print(f"skip {src.name}: only {total} frames")
            continue
        start = int(total * max(0.0, min(0.8, args.start_frac)))
        n = vae_frames(total - start, args.max_frames)
        frames, fps = read_frames(src, start=start, count=n, transform=cover_resize)
        if len(frames) < 5:
            print(f"skip {src.name}: read only {len(frames)} frames from offset {start}")
            continue
        n = vae_frames(len(frames), args.max_frames)
        frames = frames[:n]
        item = slug(src, used)
        dst = args.dest / f"{item}.mp4"
        write_mp4(frames, dst, fps if fps > 1 else FPS)
        suffix, prompt, box = edit_for(src, k)
        rows.append({"item_id": item, "edit": suffix, "prompt": prompt,
                     "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                     "n_frames": n, "start_frame": start, "source": str(src)})
        print(f"[{item}] {src.name}: frames {start}..{start+n} ({n}) {WIDTH}x{HEIGHT}  edit={suffix}")
        print(f"         \"{prompt}\"  box={box}")

    if not rows:
        print("nothing prepared", file=sys.stderr)
        return 2
    if MANIFEST.exists() and not args.force:
        print(f"\n{MANIFEST.name} already exists - keeping it (pass --force to overwrite)")
    else:
        with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {MANIFEST} with {len(rows)} items")
    return 0


if __name__ == "__main__":
    sys.exit(main())
