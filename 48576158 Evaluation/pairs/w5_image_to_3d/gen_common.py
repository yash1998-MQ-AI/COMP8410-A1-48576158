"""Shared generation loop for both Week 5 models. Each model runs in its own Python environment
(they need different torch/CUDA builds), so this file imports only the standard library, torch and a1eval.

A model file supplies `load()` and `generate(handle, image_path, seed, out_glb) -> dict of stage timings`.
The loop is resumable (a finished item/seed leaves a .json sidecar), records failures instead of crashing,
and does one unrecorded warm-up call so CUDA start-up cost does not count as latency.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from a1eval import Record, append, timed  # noqa: E402
from w5_common import INPUT_DIR, OUT_ROOT, PAIR, RESULTS, SEEDS, WARMUP_SEED, item_ids, mesh_path  # noqa: E402


def gpu_info() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return {"gpu": p.name, "gpu_gb": round(p.total_memory / 1024**3, 1), "torch": torch.__version__}
    except Exception:
        pass
    return {"gpu": "none"}


def run(model: str, load, generate, argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="*", default=None, help="item ids (default: every item in the manifest)")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--inputs", type=Path, default=INPUT_DIR)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args(argv)

    items = args.items or item_ids()
    missing = [i for i in items if not (args.inputs / f"{i}.png").exists()]
    if missing:
        print(f"missing input images for: {missing}. Run run_w5.py prep first.")
        return 2

    t0 = time.perf_counter()
    handle = load()
    load_s = time.perf_counter() - t0
    info = gpu_info()
    print(f"[{model}] loaded in {load_s:.0f}s on {info}")
    done_marker = args.results.parent / f".{model}_load_logged"
    append(args.results, Record(PAIR, model, "all", "load_time_s", load_s, None, info))

    if not args.no_warmup:
        warm = args.out_root / "_warmup" / f"{model}.glb"
        warm.parent.mkdir(parents=True, exist_ok=True)
        try:
            t = time.perf_counter()
            generate(handle, str(args.inputs / f"{items[0]}.png"), WARMUP_SEED, str(warm))
            print(f"[{model}] warm-up call {time.perf_counter() - t:.0f}s (not recorded)")
        except Exception as exc:  # a failing warm-up is worth seeing but must not stop the run
            print(f"[{model}] warm-up failed: {exc!r}")
            traceback.print_exc()

    n_ok = n_fail = 0
    for item in items:
        for seed in args.seeds:
            out = mesh_path(model, item, seed, args.out_root)
            side = out.with_suffix(".json")
            if side.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                with timed(args.results, PAIR, model, item, seed=seed) as ctx:
                    stages = generate(handle, str(args.inputs / f"{item}.png"), seed, str(out)) or {}
                    ctx.update({"stages": stages})
                for k, v in stages.items():
                    append(args.results, Record(PAIR, model, item, f"latency_{k}_s", float(v), seed))
                try:
                    import torch

                    if torch.cuda.is_available():
                        gb = torch.cuda.max_memory_reserved() / 1024**3
                        append(args.results, Record(PAIR, model, item, "peak_vram_reserved_gb", gb, seed))
                except Exception:
                    pass
                ok = out.exists() and out.stat().st_size > 0
                append(args.results, Record(PAIR, model, item, "gen_ok", float(ok), seed))
                side.write_text(json.dumps({"ok": ok, "stages": stages}))
                n_ok += int(ok)
                n_fail += int(not ok)
                print(f"[{model}] {item} seed {seed}: {'ok' if ok else 'NO FILE'} {stages}")
            except Exception as exc:
                append(args.results, Record(PAIR, model, item, "gen_ok", 0.0, seed, {"error": repr(exc)[:300]}))
                side.write_text(json.dumps({"ok": False, "error": repr(exc)[:300]}))
                n_fail += 1
                print(f"[{model}] {item} seed {seed}: FAILED {exc!r}")
                traceback.print_exc()
    print(f"[{model}] finished: {n_ok} ok, {n_fail} failed")
    return 0
