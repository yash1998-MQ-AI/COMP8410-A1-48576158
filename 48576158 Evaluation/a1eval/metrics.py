"""Metric functions shared by all pairs.

Light metrics (PSNR, SSIM, frame difference, bootstrap statistics) need only
numpy/scikit-image. Model-based metrics (CLIP, LPIPS, ArcFace identity) import
their dependencies lazily so the package works without a GPU.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PIL import Image


# ---------------------------------------------------------------- image io
def load_image(path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


# ------------------------------------------------------- reconstruction
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(255.0**2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def masked_psnr(a: np.ndarray, b: np.ndarray, keep_mask: np.ndarray) -> float:
    """PSNR over pixels where keep_mask is True (e.g. region that should NOT change)."""
    keep = keep_mask.astype(bool)
    if not keep.any():
        raise ValueError("keep_mask selects no pixels")
    return psnr(a[keep], b[keep])


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    p = np.pad(gray.astype(np.float32), 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return np.hypot(gx, gy)


def edge_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structure preservation: SSIM between Sobel edge maps of two images.

    Unlike pixel metrics it tolerates global lighting/colour changes, so it suits
    relighting and time-of-day edits where the layout should stay fixed. If the
    sizes differ, b is resized to a.
    """
    from skimage.metrics import structural_similarity

    if a.shape[:2] != b.shape[:2]:
        b = np.asarray(Image.fromarray(b).resize((a.shape[1], a.shape[0]), Image.LANCZOS))
    ma, mb = _sobel_magnitude(a.mean(axis=2)), _sobel_magnitude(b.mean(axis=2))
    return float(structural_similarity(ma, mb, data_range=max(float(ma.max()), float(mb.max()), 1e-6)))


# ---------------------------------------------------------------- video
def frame_flicker(frames: Sequence[np.ndarray]) -> float:
    """Mean absolute difference between consecutive frames (0-255 scale).

    Lower is smoother, but static videos score 0, so report it next to a
    prompt-alignment metric rather than alone.
    """
    if len(frames) < 2:
        raise ValueError("need at least two frames")
    diffs = [
        float(np.mean(np.abs(frames[i + 1].astype(np.float32) - frames[i].astype(np.float32))))
        for i in range(len(frames) - 1)
    ]
    return float(np.mean(diffs))


# ----------------------------------------------------- model-based (lazy)
_CLIP = {}


def _clip(name: str = "openai/clip-vit-base-patch32"):
    if name not in _CLIP:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(name).to(dev).eval()
        _CLIP[name] = (model, CLIPProcessor.from_pretrained(name), dev)
    return _CLIP[name]


def _image_embeds(model, pixel_values):
    """Projected CLIP image embeddings; built from the sub-modules so it does not
    depend on what a given transformers version returns from get_image_features."""
    return model.visual_projection(model.vision_model(pixel_values=pixel_values).pooler_output)


def _text_embeds(model, input_ids, attention_mask):
    return model.text_projection(model.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output)


def clip_text_image(image: np.ndarray, text: str, name: str = "openai/clip-vit-base-patch32") -> float:
    """Cosine similarity between an image and a text prompt (CLIP-T)."""
    import torch

    model, proc, dev = _clip(name)
    inputs = proc(text=[text], images=Image.fromarray(image), return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        img = _image_embeds(model, inputs["pixel_values"])
        txt = _text_embeds(model, inputs["input_ids"], inputs["attention_mask"])
    img = img / img.norm(dim=-1, keepdim=True)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    return float((img * txt).sum())


def clip_image_image(a: np.ndarray, b: np.ndarray, name: str = "openai/clip-vit-base-patch32") -> float:
    """Cosine similarity between two images in CLIP space (CLIP-I)."""
    import torch

    model, proc, dev = _clip(name)
    inputs = proc(images=[Image.fromarray(a), Image.fromarray(b)], return_tensors="pt").to(dev)
    with torch.no_grad():
        emb = _image_embeds(model, inputs["pixel_values"])
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return float((emb[0] * emb[1]).sum())


_LPIPS = {}


def lpips_distance(a: np.ndarray, b: np.ndarray, net: str = "alex") -> float:
    import lpips
    import torch

    if net not in _LPIPS:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _LPIPS[net] = (lpips.LPIPS(net=net).to(dev).eval(), dev)
    model, dev = _LPIPS[net]

    def prep(x):
        t = torch.from_numpy(x).permute(2, 0, 1).float() / 127.5 - 1.0
        return t.unsqueeze(0).to(dev)

    with torch.no_grad():
        return float(model(prep(a), prep(b)))


def identity_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of ArcFace embeddings (insightface). Research use only.

    Returns NaN when no face is detected in either image so the failure is
    visible in the results instead of silently dropped.
    """
    from insightface.app import FaceAnalysis

    global _FACE
    try:
        _FACE
    except NameError:
        _FACE = FaceAnalysis(name="buffalo_l")
        _FACE.prepare(ctx_id=0, det_size=(640, 640))
    fa, fb = _FACE.get(a[:, :, ::-1]), _FACE.get(b[:, :, ::-1])
    if not fa or not fb:
        return float("nan")
    ea, eb = fa[0].normed_embedding, fb[0].normed_embedding
    return float(np.dot(ea, eb))


def facenet_embedding(img: np.ndarray) -> np.ndarray | None:
    """L2-normalised 512-d FaceNet embedding (InceptionResnetV1, VGGFace2) of the largest face.

    Returns None when MTCNN finds no face, so a failure to render a face is
    visible in the results instead of being scored as a bad match.

    FaceNet is used as the primary identity metric because its architecture and
    training set are independent of the ArcFace recognisers that PuLID and
    InstantID use internally; scoring with the same family of recogniser a
    model conditions on would reward it for matching its own encoder.
    """
    import torch
    from PIL import Image as _PILImage

    global _FACENET
    try:
        _FACENET
    except NameError:
        from facenet_pytorch import MTCNN, InceptionResnetV1

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _FACENET = (
            MTCNN(image_size=160, margin=0, select_largest=True, device=dev),
            InceptionResnetV1(pretrained="vggface2").eval().to(dev),
            dev,
        )
    mtcnn, net, dev = _FACENET
    face = mtcnn(_PILImage.fromarray(img))
    if face is None:
        return None
    with torch.no_grad():
        e = net(face.unsqueeze(0).to(dev))[0]
    return (e / e.norm()).cpu().numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two embeddings."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ------------------------------------------------------------ statistics
def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of the mean. Returns (mean, lo, hi).

    NaNs are dropped. With very small n the interval is wide on purpose;
    report n next to it.
    """
    x = np.asarray([v for v in values if not (isinstance(v, float) and math.isnan(v))], dtype=np.float64)
    if x.size == 0:
        return (math.nan, math.nan, math.nan)
    if x.size == 1:
        return (float(x[0]), float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return (float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 5000, seed: int = 0):
    """Mean of (a - b) over matched items with a bootstrap CI. Returns (mean, lo, hi)."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    return bootstrap_ci(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), n_boot=n_boot, seed=seed)


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion k/n. Returns (p, lo, hi)."""
    if n == 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (p, centre - half, centre + half)
