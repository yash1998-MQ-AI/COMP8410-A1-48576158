"""Week 4: pose estimation and training for Nerfacto and 3DGS (splatfacto).

    python pairs/w4_nerf_3dgs/run_w4.py process --images /content/drive/MyDrive/comp8410_a1/w4_data/setA
    python pairs/w4_nerf_3dgs/run_w4.py train --models nerfacto splatfacto --seeds 0 --iters 200   # smoke
    python pairs/w4_nerf_3dgs/run_w4.py train --models nerfacto splatfacto --seeds 0 1 2            # full

`process` runs COLMAP once. Both models train on those same poses, the same images and the same held-out
frames, so a difference in the results is a difference between the methods and not between the data.
`train` records wall-clock training time, peak GPU memory and checkpoint size per run in results/w4.jsonl.
Held-out scoring is a separate step: score_w4.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from a1eval import Record, append, load  # noqa: E402
from w4_common import (  # noqa: E402
    DEFAULT_ITERS, DOWNSCALE, EVAL_INTERVAL, INPUT_DIR, MIN_REGISTERED, MODELS, NERFSTUDIO_VERSION, NS_OUT, PAIR,
    RESULTS, SCENE_DIR, SCENE_JSON, SEEDS, SUMMARY, run_dir,
)

IMG_EXT = {".jpg", ".jpeg", ".png"}


def gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return out[0] if out else "none"
    except Exception:
        return "unknown"


def vram_used_mib() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return int(out[0]) if out else 0
    except Exception:
        return 0


def versions() -> dict:
    """Library and tool versions for the evidence file."""
    info = {"python": sys.version.split()[0], "nerfstudio_pinned": NERFSTUDIO_VERSION}
    for pkg in ("torch", "gsplat", "nerfacc", "nerfstudio", "numpy", "lpips"):
        try:
            import importlib.metadata as md
            info[pkg] = md.version(pkg)
        except Exception:
            info[pkg] = None
    try:
        out = subprocess.run(["dpkg", "-s", "colmap"], capture_output=True, text=True).stdout
        info["colmap"] = next((ln.split(":", 1)[1].strip() for ln in out.splitlines() if ln.startswith("Version")), None)
    except Exception:
        info["colmap"] = None
    return info


def write_summary(update: dict) -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    data.update(update)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ process
def cmd_process(args: argparse.Namespace) -> int:
    src = Path(args.images)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"no images found in {src}")
    # One shared camera model: keep only photos with the dominant resolution (Set A holds 5 stray 1440x1080 files
    # among 100 at 4080x3072; mixed sizes under one set of intrinsics made COLMAP fail in the smoke test).
    from collections import Counter
    from PIL import Image, ImageOps
    sizes = {}
    for p in files:
        with Image.open(p) as im:
            sizes[p] = ImageOps.exif_transpose(im).size
    main_size, _ = Counter(sizes.values()).most_common(1)[0]
    dropped = [p.name for p in files if sizes[p] != main_size]
    files = [p for p in files if sizes[p] == main_size]
    print(f"kept {len(files)} photos at {main_size[0]}x{main_size[1]}; dropped {len(dropped)} with another size: {dropped}", flush=True)
    if args.limit and len(files) > args.limit:  # spread evenly so the subset still has overlap and baseline
        step = len(files) / args.limit
        files = [files[int(i * step)] for i in range(args.limit)]
    stage = INPUT_DIR
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for p in (stage, SCENE_DIR):
        if " " in str(p):
            raise SystemExit(f"working folder {p} contains a space; ns-process-data cannot handle that (set W4_WORK)")
    if SCENE_DIR.exists():  # a stale database.db / images from an earlier run gives COLMAP two cameras and no frames
        shutil.rmtree(SCENE_DIR)
    img_dir, img_dir2 = SCENE_DIR / "images", SCENE_DIR / f"images_{DOWNSCALE}"
    for d in (img_dir, img_dir2):
        d.mkdir(parents=True, exist_ok=True)
    for p in files:  # resize to --max-side (CPU pose estimation; also fixes the training resolution)
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
            im.save(img_dir / (p.stem + ".jpg"), quality=95)
            w, h = im.size
            im.resize((w // DOWNSCALE, h // DOWNSCALE), Image.LANCZOS).save(img_dir2 / (p.stem + ".jpg"), quality=95)

    # Poses: pycolmap (current COLMAP) instead of the apt COLMAP 3.7 that ns-process-data calls. On the same photos
    # apt COLMAP 3.7 found no initial image pair, while pycolmap registered every frame.
    try:
        import pycolmap
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "pycolmap"])
        import pycolmap
    t0 = time.perf_counter()
    colmap_dir = SCENE_DIR / "colmap"
    sparse = colmap_dir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    db = colmap_dir / "database.db"
    pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, device=pycolmap.Device.cpu)
    if args.matching == "sequential":
        pycolmap.match_sequential(db, device=pycolmap.Device.cpu)
    else:
        pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
    models = pycolmap.incremental_mapping(db, img_dir, sparse)
    if not models:
        print("pycolmap built no sparse model", flush=True)
        return 1
    best = max(models, key=lambda k: models[k].num_reg_images())
    print(f"pycolmap {pycolmap.__version__}: {len(models)} model(s), best has {models[best].num_reg_images()} "
          f"registered images", flush=True)
    from nerfstudio.process_data.colmap_utils import colmap_to_json
    colmap_to_json(recon_dir=sparse / str(best), output_dir=SCENE_DIR)
    elapsed = time.perf_counter() - t0
    tj = SCENE_DIR / "transforms.json"
    if not tj.exists():
        print("transforms.json was not written", flush=True)
        return 1

    n_reg = len(json.loads(tj.read_text(encoding="utf-8"))["frames"])
    frac = n_reg / len(files)
    scene = {"n_input": len(files), "n_registered": n_reg, "fraction_registered": round(frac, 4),
             "matching": args.matching, "max_side": args.max_side, "pose_tool": "pycolmap " + pycolmap.__version__, "dropped_other_size": dropped, "process_seconds": round(elapsed, 1), "min_required": MIN_REGISTERED,
             "gate_passed": frac >= MIN_REGISTERED}
    SCENE_JSON.parent.mkdir(parents=True, exist_ok=True)
    SCENE_JSON.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    write_summary({"scene": scene, "gpu": gpu_name(), "versions": versions()})
    print(json.dumps(scene, indent=2), flush=True)
    if frac < MIN_REGISTERED:
        print(f"GATE FAILED: COLMAP placed {n_reg}/{len(files)} frames ({frac:.0%}); need {MIN_REGISTERED:.0%}.", flush=True)
        return 2
    print(f"GATE PASSED: {n_reg}/{len(files)} frames registered", flush=True)
    return 0


# -------------------------------------------------------------------- train
def train_cmd(model: str, seed: int, iters: int, ns_out: Path) -> list[str]:
    return ["ns-train", model,
            "--max-num-iterations", str(iters),
            "--steps-per-save", str(iters),
            "--steps-per-eval-batch", "0", "--steps-per-eval-image", "0", "--steps-per-eval-all-images", "0",
            "--machine.seed", str(seed),
            "--output-dir", str(ns_out), "--experiment-name", f"{model}_s{seed}", "--timestamp", "run",
            "--vis", "tensorboard",
            "nerfstudio-data", "--data", str(SCENE_DIR),
            "--eval-mode", "interval", "--eval-interval", str(EVAL_INTERVAL),
            "--downscale-factor", str(DOWNSCALE)]


def checkpoint_mb(folder: Path) -> float:
    ckpts = sorted((folder / "nerfstudio_models").glob("*.ckpt"))
    return ckpts[-1].stat().st_size / 1024**2 if ckpts else float("nan")


def cmd_train(args: argparse.Namespace) -> int:
    if not (SCENE_DIR / "transforms.json").exists():
        raise SystemExit("no scene yet: run the `process` step first")
    scene = json.loads(SCENE_JSON.read_text(encoding="utf-8")) if SCENE_JSON.exists() else {}
    if scene and not scene.get("gate_passed", True) and not args.ignore_gate:
        raise SystemExit("pose gate failed in the process step; fix that before training")
    ns_out = Path(args.ns_out)
    have = {(r.model, r.item_id, r.metric) for r in load(args.results)}
    failures = 0
    for model in args.models:
        for seed in args.seeds:
            item = f"train__s{seed}"
            if (model, item, "train_time_s") in have:
                print(f"skip {model} seed {seed}: already trained", flush=True)
                continue
            shutil.rmtree(ns_out / f"{model}_s{seed}", ignore_errors=True)
            base = vram_used_mib()
            peak = [base]
            stop = threading.Event()

            def poll() -> None:
                while not stop.is_set():
                    peak[0] = max(peak[0], vram_used_mib())
                    stop.wait(2.0)

            th = threading.Thread(target=poll, daemon=True)
            th.start()
            cmd = train_cmd(model, seed, args.iters, ns_out)
            print(" ".join(cmd), flush=True)
            t0 = time.perf_counter()
            rc = subprocess.call(cmd)
            elapsed = time.perf_counter() - t0
            stop.set()
            th.join()
            folder = run_dir(model, seed, ns_out)
            if rc != 0 or not (folder / "config.yml").exists():
                print(f"FAILED {model} seed {seed} (exit {rc})", flush=True)
                failures += 1
                continue
            meta = {"iters": args.iters, "downscale": DOWNSCALE, "eval_interval": EVAL_INTERVAL,
                    "gpu": gpu_name(), "scene_frames": scene.get("n_registered")}
            rows = {"train_time_s": elapsed, "peak_vram_gb": max(peak[0] - base, 0) / 1024,
                    "model_size_mb": checkpoint_mb(folder)}
            for metric, value in rows.items():
                append(args.results, Record(PAIR, model, item, metric, float(value), seed, meta))
            print(f"done {model} seed {seed}: {elapsed:.0f}s, {rows['peak_vram_gb']:.1f} GB, "
                  f"{rows['model_size_mb']:.0f} MB", flush=True)
    write_summary({"train": {"iters": args.iters, "seeds": list(args.seeds), "models": list(args.models),
                             "gpu": gpu_name(), "versions": versions()}})
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("process")
    p.add_argument("--images", required=True, help="folder with the source photos")
    p.add_argument("--limit", type=int, default=0, help="use only the first N photos (smoke test)")
    p.add_argument("--max-side", type=int, default=1440, help="resize photos so the longer side is this many px")
    p.add_argument("--matching", default="exhaustive", choices=["exhaustive", "sequential", "vocab_tree"])
    t = sub.add_parser("train")
    t.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    t.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    t.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    t.add_argument("--results", default=str(RESULTS))
    t.add_argument("--ns-out", default=str(NS_OUT))
    t.add_argument("--ignore-gate", action="store_true")
    args = ap.parse_args()
    return cmd_process(args) if args.stage == "process" else cmd_train(args)


if __name__ == "__main__":
    sys.exit(main())
