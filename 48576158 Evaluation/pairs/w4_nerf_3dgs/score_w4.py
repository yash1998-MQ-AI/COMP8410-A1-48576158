"""Week 4: render every held-out view from each trained model and score it against the real photo.

    python pairs/w4_nerf_3dgs/score_w4.py --models nerfacto splatfacto --seeds 0 1 2

Per held-out view (item id "<photo>__s<seed>"): psnr, ssim, lpips (same implementation for both models, from
a1eval.metrics, on the same 8-bit pixels). Per run (item id "eval__s<seed>"): render_fps and, for 3DGS,
n_gaussians. Renders go to outputs/w4/<model>/s<seed>/ and the real photos to outputs/w4/gt/; these show a
private room, so outputs/ stays out of Git.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from a1eval import Record, append, load  # noqa: E402
from a1eval import metrics as M  # noqa: E402
from w4_common import MODELS, NS_OUT, PAIR, RESULTS, ROOT, run_dir  # noqa: E402

CHUNK = 1 << 15  # rays per chunk for Nerfacto's ray marching; splatfacto ignores it


def to_uint8(x) -> np.ndarray:
    """Network output or ground truth (float 0-1, possibly on GPU) -> uint8 HxWx3."""
    arr = x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=np.float32)
    if arr.max() > 1.5:  # already 0-255
        arr = arr / 255.0
    return (np.clip(arr[..., :3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def load_pipeline(folder: Path):
    import torch

    # Newer torch refuses to unpickle some objects by default; these are our own checkpoints.
    torch.load = functools.partial(torch.load, weights_only=False)
    from nerfstudio.utils.eval_utils import eval_setup

    # Nerfacto's ParallelDataManager starts background ray-loading processes in setup_train and has no eval
    # loader in "inference" mode. We only need the held-out images and cameras, which we read from the dataset
    # directly (below), so switch those loaders off. "test" mode (not "inference") keeps the downscaled images.
    from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager

    ParallelDataManager.setup_train = lambda self: None
    ParallelDataManager.setup_eval = lambda self: None
    _, pipeline, _, step = eval_setup(folder / "config.yml", eval_num_rays_per_chunk=CHUNK, test_mode="test")
    pipeline.eval()
    return pipeline, step


def eval_views(pipeline):
    """Yield (index, camera, ground-truth image) for every held-out view, for any nerfstudio data manager."""
    dataset = pipeline.datamanager.eval_dataset
    device = pipeline.device
    for idx in range(len(dataset)):
        camera = dataset.cameras[idx: idx + 1].to(device)
        camera.metadata = {"cam_idx": idx}
        yield idx, camera, dataset[idx]["image"]


def score_run(model: str, seed: int, args: argparse.Namespace, have: set) -> int:
    import torch

    folder = run_dir(model, seed, Path(args.ns_out))
    if not (folder / "config.yml").exists():
        print(f"missing run {model} seed {seed}: train it first", flush=True)
        return 0
    pipeline, step = load_pipeline(folder)
    out_dir = Path(args.out_root) / "w4" / model / f"s{seed}"
    gt_dir = Path(args.out_root) / "w4" / "gt"
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    names = [Path(p).stem for p in pipeline.datamanager.eval_dataset.image_filenames]
    n = 0
    times: list[float] = []
    for idx, camera, gt_image in eval_views(pipeline):
        name = names[idx]
        item = f"{name}__s{seed}"
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = pipeline.model.get_outputs_for_camera(camera)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        pred = to_uint8(outputs["rgb"])
        gt = to_uint8(gt_image)
        Image.fromarray(pred).save(out_dir / f"{name}.png")
        Image.fromarray(gt).save(gt_dir / f"{name}.png")
        meta = {"view": name, "iters_step": int(step)}
        todo = [m for m in ("psnr", "ssim", "lpips") if (model, item, m) not in have]
        for m in todo:
            v = {"psnr": M.psnr, "ssim": M.ssim, "lpips": M.lpips_distance}[m](gt, pred)
            append(args.results, Record(PAIR, model, item, m, float(v), seed, meta))
            n += 1

    run_item = f"eval__s{seed}"
    if times and (model, run_item, "render_fps") not in have:
        steady = times[2:] if len(times) > 4 else times   # first renders include warm-up
        meta = {"n_views": len(times), "width": int(pred.shape[1]), "height": int(pred.shape[0]), "chunk": CHUNK}
        append(args.results, Record(PAIR, model, run_item, "render_fps", 1.0 / float(np.mean(steady)), seed, meta))
        n += 1
    if model == "splatfacto" and (model, run_item, "n_gaussians") not in have:
        append(args.results, Record(PAIR, model, run_item, "n_gaussians", float(pipeline.model.num_points), seed, {}))
        n += 1
    del pipeline
    torch.cuda.empty_cache()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--ns-out", default=str(NS_OUT))
    ap.add_argument("--out-root", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    have = {(r.model, r.item_id, r.metric) for r in load(args.results)}
    total = 0
    for model in args.models:
        for seed in args.seeds:
            total += score_run(model, seed, args, have)
    print(f"appended {total} metric rows to {args.results}", flush=True)


if __name__ == "__main__":
    main()
