"""Week 7 scoring (CPU is enough; a T4 is faster). python score_w7.py [--rescore]

Per edited clip (model, item, seed). 9 evenly spaced frames, short side 224, centre-cropped,
for the CLIP/DINO metrics; full-resolution frames for the pixel metrics.

  clip_t            mean CLIP text-frame cosine for the edit instruction      (did the edit land)
  clip_t_src        the same measured on the INPUT clip                        (baseline)
  clip_t_gain       clip_t - clip_t_src                                        (edit effect, the headline)
  clip_t_mismatch   output against a DIFFERENT item's instruction              (control: scale has headroom)
  bg_psnr           PSNR input vs output OUTSIDE the edit box, dB             (fidelity of what must not change)
  bg_ssim           SSIM input vs output OUTSIDE the edit box                  (same, structural)
  box_mad           mean abs difference INSIDE the edit box, 0-255             (edit magnitude; 0 = no-op)
  bg_mad            mean abs difference OUTSIDE the edit box, 0-255             (leakage outside the box)
  edit_ratio        box_mad / bg_mad, capped at 1000                            (is the change localised)
  flicker           mean abs second temporal difference, 0-255                 (high-frequency temporal noise)
  flicker_src       the same on the input clip                                 (baseline)
  flicker_excess    flicker - flicker_src                                      (noise the model ADDED)
  motion_mad        mean abs frame difference                                  (catches a frozen output)
  dino_consec       DINOv2 cosine between consecutive sampled frames           (temporal coherence)
  dino_src_mean     DINOv2 cosine, output frames vs matching input frames      (content kept)
  n_frames, fps, file_mb
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
from w7_common import (CLIP_NAME, CROP, DINO_NAME, HEIGHT, MODELS, N_SCORE_FRAMES, OUT_ROOT, PAIR,  # noqa: E402
                       RESULTS, SEEDS, WIDTH, box_of, box_pixels, clip_path, item_ids, prompt_of,
                       video_path)
from pairs.w7_video_editing.prepare_inputs import read_frames  # noqa: E402


# ---------------------------------------------------------------- pure functions (unit-tested)
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


def _grey_small(f: np.ndarray, size=(208, 120)) -> np.ndarray:
    return np.asarray(Image.fromarray(f).convert("L").resize(size, Image.BILINEAR), dtype=np.float32)


def box_mask(box, shape) -> np.ndarray:
    """Boolean mask, True inside the edit box, for a frame of shape (h, w, 3) or (h, w)."""
    h, w = shape[0], shape[1]
    x0, y0, x1, y1 = box_pixels(box, w, h)
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


PSNR_MAX = 100.0   # dB ceiling for an identical region: keeps the metric finite and averageable
RATIO_MAX = 1000.0  # ceiling for edit_ratio when the background is pixel-identical


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB over the supplied values (already masked/flattened).

    An identical region has zero MSE and infinite PSNR, which cannot be averaged, so it is
    reported as PSNR_MAX. Anything at the ceiling means "pixel-identical", not "very good".
    """
    if a.size == 0:
        return float("nan")
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return PSNR_MAX
    return float(min(PSNR_MAX, 10.0 * np.log10(255.0 ** 2 / mse)))


def region_fidelity(src: list[np.ndarray], out: list[np.ndarray], box) -> dict:
    """Fidelity outside the edit box and change magnitude inside it.

    SSIM is computed as a full per-pixel map and then averaged over the outside mask only,
    so an interior box does not defeat it. edit_ratio is capped: a perfectly preserved
    background makes the denominator zero, which is the best case, not a missing value.
    """
    from skimage.metrics import structural_similarity

    n = min(len(src), len(out))
    m = box_mask(box, out[0].shape)
    inside, outside = m, ~m
    bg_p, bg_s, in_mad, out_mad = [], [], [], []
    for i in range(n):
        a, b = src[i], out[i]
        if a.shape != b.shape:
            a = np.asarray(Image.fromarray(a).resize((b.shape[1], b.shape[0]), Image.LANCZOS))
        ga = np.asarray(Image.fromarray(a).convert("L"), dtype=np.float64)
        gb = np.asarray(Image.fromarray(b).convert("L"), dtype=np.float64)
        if outside.any():
            bg_p.append(psnr(a[outside], b[outside]))
            _, smap = structural_similarity(ga, gb, data_range=255, full=True)
            bg_s.append(float(smap[outside].mean()))
            out_mad.append(float(np.abs(ga[outside] - gb[outside]).mean()))
        if inside.any():
            in_mad.append(float(np.abs(ga[inside] - gb[inside]).mean()))
    bm = float(np.mean(in_mad)) if in_mad else float("nan")
    om = float(np.mean(out_mad)) if out_mad else float("nan")
    if not np.isfinite(om) or not np.isfinite(bm):
        ratio = float("nan")
    elif om < 1e-3:                       # background pixel-identical: maximally localised edit
        ratio = RATIO_MAX
    else:
        ratio = min(RATIO_MAX, bm / om)
    return {"bg_psnr": float(np.mean(bg_p)) if bg_p else float("nan"),
            "bg_ssim": float(np.mean(bg_s)) if bg_s else float("nan"),
            "box_mad": bm,
            "bg_mad": om,
            "edit_ratio": ratio}


def temporal_stats(frames: list[np.ndarray]) -> dict:
    """motion_mad (first difference) and flicker (second difference), both 0-255 greyscale."""
    g = np.stack([_grey_small(f) for f in frames])
    mad = float(np.abs(np.diff(g, axis=0)).mean()) if len(g) >= 2 else float("nan")
    flick = float(np.abs(g[2:] - 2 * g[1:-1] + g[:-2]).mean()) if len(g) >= 3 else float("nan")
    return {"motion_mad": mad, "flicker": flick}


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-12)


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine between two (n,d) matrices."""
    num = (a * b).sum(axis=1)
    return num / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


# ---------------------------------------------------------------- embedder
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


def score_pair(emb: Embedder, src: list[np.ndarray], out: list[np.ndarray], fps: float,
               box, prompt: str, other_prompt: str) -> dict:
    n = min(len(src), len(out))
    idx = sample_idx(n)
    oc = [short_side_crop(out[i]) for i in idx]
    sc = [short_side_crop(src[i]) for i in idx]
    ce, de = emb.images(oc + sc)
    k = len(idx)
    co, do_ = ce[:k], de[:k]
    cs, ds = ce[k:], de[k:]
    t, t_other = emb.text(prompt), emb.text(other_prompt)
    consec = [float(np.dot(do_[i], do_[i + 1]) / (np.linalg.norm(do_[i]) * np.linalg.norm(do_[i + 1]) + 1e-12))
              for i in range(k - 1)]
    clip_t = float(cos(co, t).mean())
    clip_t_src = float(cos(cs, t).mean())
    m = {
        "clip_t": clip_t,
        "clip_t_src": clip_t_src,
        "clip_t_gain": clip_t - clip_t_src,
        "clip_t_mismatch": float(cos(co, t_other).mean()),
        "dino_src_mean": float(cos_rows(do_, ds).mean()),
        "dino_consec": float(np.mean(consec)) if consec else float("nan"),
        "n_frames": float(n),
        "fps": float(fps),
    }
    m.update(region_fidelity(src[:n], out[:n], box))
    ts_out, ts_src = temporal_stats(out[:n]), temporal_stats(src[:n])
    m["motion_mad"] = ts_out["motion_mad"]
    m["motion_mad_src"] = ts_src["motion_mad"]
    m["flicker"] = ts_out["flicker"]
    m["flicker_src"] = ts_src["flicker"]
    m["flicker_excess"] = ts_out["flicker"] - ts_src["flicker"]
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args(argv)

    ids = item_ids()
    scored = {(r.model, r.item_id, r.seed) for r in load(args.results) if r.metric == "clip_t_gain"}
    emb = Embedder()
    for model in MODELS:
        for k, item in enumerate(ids):
            for seed in SEEDS:
                p = video_path(model, item, seed, args.out_root)
                if not p.exists():
                    continue
                if (model, item, seed) in scored and not args.rescore:
                    continue
                out, fps = read_frames(p)
                src, _ = read_frames(clip_path(item))
                other = prompt_of(ids[(k + 1) % len(ids)])
                m = score_pair(emb, src, out, fps, box_of(item), prompt_of(item), other)
                m["file_mb"] = p.stat().st_size / 1e6
                for name, v in m.items():
                    append(args.results, Record(PAIR, model, item, name, float(v), seed))
                print(f"[{model}] {item}: gain {m['clip_t_gain']:+.3f}  bg_psnr {m['bg_psnr']:.1f}dB  "
                      f"box_mad {m['box_mad']:.1f}  flicker+ {m['flicker_excess']:+.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
