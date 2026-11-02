"""Week 6 scoring (CPU is enough; a T4 is faster). python score_w6.py [--rescore]

Per generated clip (model, item, seed), 9 evenly spaced frames, short side resized to 224 and centre-cropped:
  clip_t           mean CLIP text-frame cosine for the prompt                 (prompt alignment, weak: CLIP is coarse)
  clip_i_src_mean  mean CLIP image-image cosine, frames vs the input photo     (content kept)
  clip_i_src_min   worst frame                                                  (drift)
  dino_src_mean/min  same with DINOv2-base CLS                                  (finer layout/identity kept)
  dino_consec      mean DINOv2 cosine between consecutive sampled frames        (temporal consistency)
  dino_first_last  DINOv2 cosine first vs last frame                            (how far the scene travelled)
  first_frame_ssim SSIM of frame 0 vs the input photo (256x144, grey)           (does it start from the photo)
  motion_mad       mean abs frame difference, 0-255, 160x90 grey                (amount of motion; 0 = frozen)
  jitter_ratio     mean abs second difference / motion_mad                      (shaky vs smooth motion; lower = smoother)
  luma_drift       |mean luminance last - first|                                (exposure / colour drift)
  <metric>_mismatch  clip_i / dino against a DIFFERENT item's photo             (control: shows the scale has headroom)
  n_frames, fps, width, height, file_mb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from a1eval import Record, append, load  # noqa: E402
from w6_common import (CLIP_NAME, CROP, DINO_NAME, INPUT_DIR, MODELS, N_SCORE_FRAMES, OUT_ROOT, PAIR,  # noqa: E402
                       RESULTS, SEEDS, item_ids, prompt_of, video_path)


# ---------------------------------------------------------------- pure functions (unit-tested)
def read_video(path: Path):
    """All frames as uint8 RGB arrays plus fps. Tries OpenCV, then imageio."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if frames:
            return frames, float(fps)
    except ImportError:
        pass
    import imageio.v3 as iio

    meta = iio.immeta(str(path))
    return [np.asarray(f) for f in iio.imiter(str(path))], float(meta.get("fps", 0.0))


def short_side_crop(img: np.ndarray, size: int = CROP) -> np.ndarray:
    im = Image.fromarray(img)
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.LANCZOS)
    w, h = im.size
    l, t = (w - size) // 2, (h - size) // 2
    return np.asarray(im.crop((l, t, l + size, t + size)))


def sample_idx(n: int, k: int = N_SCORE_FRAMES) -> list[int]:
    return sorted({int(round(i)) for i in np.linspace(0, n - 1, k)})


def _grey_small(f: np.ndarray, size=(160, 90)) -> np.ndarray:
    return np.asarray(Image.fromarray(f).convert("L").resize(size, Image.BILINEAR), dtype=np.float32)


def motion_stats(frames: list[np.ndarray]) -> dict:
    g = np.stack([_grey_small(f) for f in frames])
    d1 = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
    mad = float(d1.mean())
    if len(g) >= 3:
        d2 = np.abs(g[2:] - 2 * g[1:-1] + g[:-2]).mean(axis=(1, 2))
        jit = float(d2.mean())
    else:
        jit = float("nan")
    lum = [float(np.asarray(Image.fromarray(f).convert("L")).mean()) for f in (frames[0], frames[-1])]
    return {"motion_mad": mad, "jitter_ratio": jit / mad if mad > 1e-6 else float("nan"),
            "luma_drift": abs(lum[1] - lum[0])}


def first_frame_ssim(frame0: np.ndarray, photo: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    a = _grey_small(frame0, (256, 144))
    b = _grey_small(photo, (256, 144))
    return float(structural_similarity(a, b, data_range=255))


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine of a (n,d) against one vector b (d,)."""
    return (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-12)


# ---------------------------------------------------------------- embedders
class Embedder:
    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        from a1eval import metrics as M

        self.torch = torch
        self.clip, self.clip_proc, self.dev = M._clip(CLIP_NAME)
        self._img, self._txt = M._image_embeds, M._text_embeds
        self.dino = AutoModel.from_pretrained(DINO_NAME).to(self.dev).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained(DINO_NAME)

    def images(self, imgs: list[np.ndarray]):
        torch = self.torch
        pil = [Image.fromarray(i) for i in imgs]
        with torch.no_grad():
            c = self.clip_proc(images=pil, return_tensors="pt").to(self.dev)
            ce = self._img(self.clip, c["pixel_values"])
            d = self.dino_proc(images=pil, return_tensors="pt", do_resize=False, do_center_crop=False).to(self.dev)
            de = self.dino(**d).last_hidden_state[:, 0]
        return ce.float().cpu().numpy(), de.float().cpu().numpy()

    def text(self, s: str):
        with self.torch.no_grad():
            t = self.clip_proc(text=[s], return_tensors="pt", padding=True, truncation=True).to(self.dev)
            return self._txt(self.clip, t["input_ids"], t["attention_mask"]).float().cpu().numpy()[0]


def score_clip(emb: Embedder, frames, fps, photo, prompt, other_photo) -> dict:
    idx = sample_idx(len(frames))
    crops = [short_side_crop(frames[i]) for i in idx]
    ce, de = emb.images(crops + [short_side_crop(photo), short_side_crop(other_photo)])
    cf, df = ce[: len(idx)], de[: len(idx)]
    cp, dp = ce[len(idx)], de[len(idx)]
    co, do = ce[len(idx) + 1], de[len(idx) + 1]
    t = emb.text(prompt)
    cs, ds = cos(cf, cp), cos(df, dp)
    consec = [float(np.dot(df[i], df[i + 1]) / (np.linalg.norm(df[i]) * np.linalg.norm(df[i + 1]))) for i in range(len(df) - 1)]
    out = {
        "clip_t": float(cos(cf, t).mean()),
        "clip_i_src_mean": float(cs.mean()), "clip_i_src_min": float(cs.min()),
        "dino_src_mean": float(ds.mean()), "dino_src_min": float(ds.min()),
        "dino_consec": float(np.mean(consec)),
        "dino_first_last": float(cos(df[:1], df[-1])[0]),
        "clip_i_src_mismatch": float(cos(cf, co).mean()), "dino_src_mismatch": float(cos(df, do).mean()),
        "first_frame_ssim": first_frame_ssim(frames[0], photo),
        "n_frames": float(len(frames)), "fps": float(fps), "height": float(frames[0].shape[0]), "width": float(frames[0].shape[1]),
    }
    out.update(motion_stats(frames))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, default=INPUT_DIR)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args(argv)
    ids = item_ids()
    scored = {(r.model, r.item_id, r.seed) for r in load(args.results) if r.metric == "motion_mad"}
    emb = Embedder()
    for model in MODELS:
        for k, item in enumerate(ids):
            for seed in SEEDS:
                p = video_path(model, item, seed, args.out_root)
                if not p.exists() or (model, item, seed) in scored and not args.rescore:
                    continue
                frames, fps = read_video(p)
                photo = np.asarray(Image.open(args.inputs / f"{item}.png").convert("RGB"))
                other = np.asarray(Image.open(args.inputs / f"{ids[(k + 1) % len(ids)]}.png").convert("RGB"))
                m = score_clip(emb, frames, fps, photo, prompt_of(item), other)
                m["file_mb"] = p.stat().st_size / 1e6
                for name, v in m.items():
                    append(args.results, Record(PAIR, model, item, name, float(v), seed))
                print(f"[{model}] {item}: dino_src {m['dino_src_mean']:.3f} consec {m['dino_consec']:.3f} motion {m['motion_mad']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
