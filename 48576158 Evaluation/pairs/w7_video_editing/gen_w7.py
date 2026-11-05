"""Week 7 generation: Wan2.1-VACE-1.3B vs Lucy-Edit-Dev on identical clips and instructions.

    python gen_w7.py wan21_vace --smoke      # 1 item, 6 steps, cheap sanity check
    python gen_w7.py lucy_edit               # all items, full settings
    python gen_w7.py both                    # sequential, one model loaded at a time

Resumable: a sidecar <item>_s<seed>.json beside each clip records the run; an item whose
sidecar says ok=true is skipped. A failed sidecar is retried (do not skip failures).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from a1eval import Record, append  # noqa: E402
from w7_common import (FPS, GUIDANCE, HEIGHT, HF_ID, MAX_FRAMES, MODELS, NEGATIVE, OUT_ROOT, PAIR,  # noqa: E402
                       RESULTS, SEEDS, STEPS, WIDTH, box_of, box_pixels, clip_path, item_ids,
                       prompt_of, vae_frames, video_path)


# ---------------------------------------------------------------- pure helpers (unit-tested)
def mask_frames(box, n: int, width: int = WIDTH, height: int = HEIGHT):
    """n identical masks: white (255) inside the edit box = regenerate, black = keep."""
    from PIL import Image

    x0, y0, x1, y1 = box_pixels(box, width, height)
    a = np.zeros((height, width), dtype=np.uint8)
    a[y0:y1, x0:x1] = 255
    return [Image.fromarray(a).convert("RGB") for _ in range(n)]


def load_clip(path: Path, cap: int = MAX_FRAMES):
    """The prepared clip as PIL RGB images at WIDTHxHEIGHT, trimmed to a 4n+1 count <= cap.

    No padding: repeating the last frame would add static frames and depress the motion metrics.
    """
    from PIL import Image

    sys.path.insert(0, str(HERE))
    from prepare_inputs import read_frames

    frames, _ = read_frames(path)
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    n = vae_frames(len(frames), cap)
    out = [Image.fromarray(f).convert("RGB") for f in frames[:n]]
    return [im if im.size == (WIDTH, HEIGHT) else im.resize((WIDTH, HEIGHT), Image.LANCZOS) for im in out]


def write_mp4(frames, path: Path, fps: float = FPS) -> None:
    import imageio.v3 as iio

    arr = np.stack([np.asarray(f) if not isinstance(f, np.ndarray) else f for f in frames])
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(path), arr, fps=fps, codec="libx264",
                output_params=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast"])


# ---------------------------------------------------------------- model loading
def load_vace():
    import torch
    from diffusers import AutoencoderKLWan, WanVACEPipeline

    mid = HF_ID["wan21_vace"]
    vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(mid, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe


def load_lucy():
    import torch
    from diffusers import AutoencoderKLWan, LucyEditPipeline

    mid = HF_ID["lucy_edit"]
    vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
    pipe = LucyEditPipeline.from_pretrained(mid, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe


LOADERS = {"wan21_vace": load_vace, "lucy_edit": load_lucy}


def generate(pipe, model: str, video, box, prompt: str, seed: int, steps: int):
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    kw = dict(prompt=prompt, negative_prompt=NEGATIVE, height=HEIGHT, width=WIDTH,
              num_frames=len(video), num_inference_steps=steps, guidance_scale=GUIDANCE, generator=g)
    if model == "wan21_vace":
        return pipe(video=video, mask=mask_frames(box, len(video)), **kw).frames[0]
    return pipe(video=video, **kw).frames[0]


# ---------------------------------------------------------------- driver
def run_model(model: str, ids, seeds, out_root: Path, results: Path, steps: int) -> tuple[int, int]:
    import torch

    t0 = time.perf_counter()
    pipe = LOADERS[model]()
    load_s = time.perf_counter() - t0
    print(f"[{model}] loaded in {load_s:.0f}s", flush=True)
    append(results, Record(PAIR, model, "all", "load_time_s", load_s, None))

    ok = fail = 0
    for item in ids:
        for seed in seeds:
            dst = video_path(model, item, seed, out_root)
            side = dst.with_suffix(".json")
            if side.exists():
                try:
                    if json.loads(side.read_text()).get("ok"):
                        continue
                except Exception:
                    pass
            src = clip_path(item)
            if not src.exists():
                print(f"[{model}] {item}: missing input {src} - run prepare_inputs.py", flush=True)
                fail += 1
                continue
            try:
                torch.cuda.reset_peak_memory_stats()
                video = load_clip(src)
                t = time.perf_counter()
                frames = generate(pipe, model, video, box_of(item), prompt_of(item), seed, steps)
                dt = time.perf_counter() - t
                write_mp4(frames, dst)
                vram = torch.cuda.max_memory_allocated() / 1e9
                append(results, Record(PAIR, model, item, "latency_s", dt, seed))
                append(results, Record(PAIR, model, item, "peak_vram_gb", vram, seed))
                side.write_text(json.dumps({"ok": True, "latency_s": dt, "peak_vram_gb": vram,
                                            "n_frames": len(video), "steps": steps}, indent=1))
                ok += 1
                print(f"[{model}] {item} s{seed}: {dt:.0f}s  {vram:.1f}GB  {len(video)}f -> {dst.name}", flush=True)
            except Exception as exc:
                traceback.print_exc()
                side.parent.mkdir(parents=True, exist_ok=True)
                side.write_text(json.dumps({"ok": False, "error": repr(exc)[:400]}, indent=1))
                fail += 1
    del pipe
    torch.cuda.empty_cache()
    return ok, fail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODELS) + ["both"])
    ap.add_argument("--items", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--smoke", action="store_true", help="1 item, 6 steps, separate output folder")
    args = ap.parse_args(argv)

    ids = args.items or item_ids()
    steps = args.steps
    if args.smoke:
        ids, steps = ids[:1], 6
        args.out_root = args.out_root.parent / "w7_smoke"
        args.results = args.results.with_name("w7_smoke.jsonl")

    models = list(MODELS) if args.model == "both" else [args.model]
    total_ok = total_fail = 0
    for m in models:
        ok, fail = run_model(m, ids, args.seeds, args.out_root, args.results, steps)
        total_ok, total_fail = total_ok + ok, total_fail + fail
    print(f"\nfinished: {total_ok} ok, {total_fail} failed")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
