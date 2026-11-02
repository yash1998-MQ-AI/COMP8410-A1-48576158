"""Week 6 generation: python gen_w6.py wan22|hunyuanvideo15 [--items ...] [--seeds 0] [--smoke]

One environment runs both models (no compiled extensions). Resumable: a finished item leaves a .json sidecar.
Each model first does a cheap warm-up (25 frames, 2 steps, not recorded) so CUDA start-up is not billed to item 1.
Failures are recorded as gen_ok=0 instead of stopping the run.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from a1eval import Record, append, timed  # noqa: E402
from w6_common import (FPS, INPUT_DIR, NUM_FRAMES, OUT_ROOT, PAIR, RESULTS, SEEDS, SETTINGS, WARMUP_SEED,  # noqa: E402
                       item_ids, prompt_of, video_path)

WAN_ROOT = os.environ.get("WAN_ROOT", "/content/Wan2.2")
WAN_CKPT = os.environ.get("WAN_CKPT", "/content/ckpt/Wan2.2-TI2V-5B")
HF_REPO_WAN = "Wan-AI/Wan2.2-TI2V-5B"


def _cuda_gb():
    import torch
    return torch.cuda.max_memory_allocated() / 1024**3, torch.cuda.max_memory_reserved() / 1024**3


# ---------------------------------------------------------------- Wan 2.2 (official repo, I2V via --image path)
def load_wan():
    sys.path.insert(0, WAN_ROOT)
    import wan
    import wan.modules.attention as WA
    import wan.modules.model as WM
    from wan.configs import WAN_CONFIGS

    # model.py calls flash_attention() directly and asserts flash-attn is installed; there is no wheel for Colab's
    # torch / Python 3.13. The repo's own attention() wrapper falls back to torch SDPA (same maths; batch of 1, no padding).
    if not (WA.FLASH_ATTN_2_AVAILABLE or WA.FLASH_ATTN_3_AVAILABLE):
        WM.flash_attention = WA.attention
        print("[wan22] flash-attn not available: using the SDPA fallback attention")

    cfg = WAN_CONFIGS["ti2v-5B"]
    kw = dict(config=cfg, checkpoint_dir=WAN_CKPT, device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False,
              use_sp=False, t5_cpu=False, init_on_cpu=False, convert_model_dtype=True)
    ok = set(inspect.signature(wan.WanTI2V.__init__).parameters)
    pipe = wan.WanTI2V(**{k: v for k, v in kw.items() if k in ok})
    return {"pipe": pipe, "cfg": cfg}


def gen_wan(h, image, prompt, seed, out, frames, steps=None):
    from PIL import Image
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS
    from wan.utils.utils import save_video

    s = SETTINGS["wan22"]
    cfg, pipe = h["cfg"], h["pipe"]
    key = f"{s['width']}*{s['height']}"
    kw = dict(input_prompt=prompt, img=Image.open(image).convert("RGB"), size=SIZE_CONFIGS[key],
              max_area=MAX_AREA_CONFIGS[key], frame_num=frames, shift=cfg.sample_shift, sample_solver="unipc",
              sampling_steps=steps or s["steps"], guide_scale=s["guidance"], n_prompt="", seed=seed,
              offload_model=False)
    ok = set(inspect.signature(pipe.generate).parameters)
    t0 = time.perf_counter()
    video = pipe.generate(**{k: v for k, v in kw.items() if k in ok})
    t1 = time.perf_counter()
    save_video(tensor=video[None], save_file=str(out), fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
    return {"sample": t1 - t0, "encode": time.perf_counter() - t1}


# ---------------------------------------------------------------- HunyuanVideo 1.5 (diffusers)
def find_hunyuan_repo() -> str:
    """HY_REPO env var wins. Otherwise list the community diffusers repos and take the best 480p I2V one."""
    if os.environ.get("HY_REPO"):
        return os.environ["HY_REPO"]
    from huggingface_hub import HfApi

    names = sorted(m.id for m in HfApi().list_models(author="hunyuanvideo-community", search="HunyuanVideo-1.5"))
    print("[hunyuanvideo15] community repos:", names)
    i2v = [n for n in names if "480p" in n and "i2v" in n]
    for pref in ("step_distilled", "distilled", ""):
        for n in i2v:
            if pref in n and (pref or "distilled" not in n):
                return n
    raise RuntimeError(f"no 480p i2v repo among {names}; set HY_REPO to the right id")


def load_hunyuan():
    import torch
    from diffusers import HunyuanVideo15ImageToVideoPipeline

    repo = find_hunyuan_repo()
    print("[hunyuanvideo15] using", repo)
    pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    steps = 8 if "step_distilled" in repo else 50          # step-distilled: 8-12 steps; the others need the full 50
    return {"pipe": pipe, "repo": repo, "steps": steps}


def gen_hunyuan(h, image, prompt, seed, out, frames, steps=None):
    import torch
    from diffusers.utils import export_to_video
    from PIL import Image

    s = SETTINGS["hunyuanvideo15"]
    pipe = h["pipe"]
    img = Image.open(image).convert("RGB").resize((s["width"], s["height"]), Image.LANCZOS)
    kw = dict(prompt=prompt, image=img, generator=torch.Generator("cpu").manual_seed(seed), num_frames=frames,
              num_inference_steps=steps or h["steps"], height=s["height"], width=s["width"])
    ok = set(inspect.signature(pipe.__call__).parameters)
    t0 = time.perf_counter()
    frames_out = pipe(**{k: v for k, v in kw.items() if k in ok}).frames[0]
    t1 = time.perf_counter()
    export_to_video(frames_out, str(out), fps=FPS)
    return {"sample": t1 - t0, "encode": time.perf_counter() - t1}


MODELS = {"wan22": (load_wan, gen_wan), "hunyuanvideo15": (load_hunyuan, gen_hunyuan)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODELS))
    ap.add_argument("--items", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--inputs", type=Path, default=INPUT_DIR)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--frames", type=int, default=NUM_FRAMES)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 item, results to <results>.smoke, own out folder")
    args = ap.parse_args(argv)

    model = args.model
    load, gen = MODELS[model]
    items = args.items or item_ids()
    if args.smoke:
        items = items[:1]
        args.results = args.results.with_name("w6_smoke.jsonl")
        args.out_root = args.out_root.parent / "w6_smoke"
    missing = [i for i in items if not (args.inputs / f"{i}.png").exists()]
    if missing:
        print("missing inputs:", missing, "- run prepare_inputs.py first")
        return 2

    import torch
    t0 = time.perf_counter()
    h = load()
    load_s = time.perf_counter() - t0
    info = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "settings": SETTINGS[model],
            "repo": h.get("repo", HF_REPO_WAN), "steps_used": h.get("steps", SETTINGS[model]["steps"])}
    print(f"[{model}] loaded in {load_s:.0f}s on {info['gpu']}")
    append(args.results, Record(PAIR, model, "all", "load_time_s", load_s, None, info))

    if not args.no_warmup:
        w = args.out_root / "_warmup" / f"{model}.mp4"
        w.parent.mkdir(parents=True, exist_ok=True)
        try:
            t = time.perf_counter()
            gen(h, str(args.inputs / f"{items[0]}.png"), prompt_of(items[0]), WARMUP_SEED, w, 25, steps=2)
            print(f"[{model}] warm-up {time.perf_counter() - t:.0f}s (not recorded)")
        except Exception as exc:
            print(f"[{model}] warm-up failed: {exc!r}")
            traceback.print_exc()

    n_ok = n_fail = 0
    for item in items:
        for seed in args.seeds:
            out = video_path(model, item, seed, args.out_root)
            side = out.with_suffix(".json")
            if side.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.cuda.empty_cache()
            try:
                with timed(args.results, PAIR, model, item, seed=seed, frames=args.frames) as ctx:
                    stages = gen(h, str(args.inputs / f"{item}.png"), prompt_of(item), seed, out, args.frames)
                    ctx["stages"] = stages
                for k, v in stages.items():
                    append(args.results, Record(PAIR, model, item, f"latency_{k}_s", float(v), seed))
                _, rsv = _cuda_gb()
                append(args.results, Record(PAIR, model, item, "peak_vram_reserved_gb", rsv, seed))
                ok = out.exists() and out.stat().st_size > 0
                append(args.results, Record(PAIR, model, item, "gen_ok", float(ok), seed))
                side.write_text(json.dumps({"ok": ok, "stages": stages}))
                n_ok += int(ok)
                n_fail += int(not ok)
                print(f"[{model}] {item} s{seed}: {'ok' if ok else 'NO FILE'} {stages}")
            except Exception as exc:
                append(args.results, Record(PAIR, model, item, "gen_ok", 0.0, seed, {"error": repr(exc)[:300]}))
                side.write_text(json.dumps({"ok": False, "error": repr(exc)[:300]}))
                n_fail += 1
                print(f"[{model}] {item} s{seed}: FAILED {exc!r}")
                traceback.print_exc()
    print(f"[{model}] finished: {n_ok} ok, {n_fail} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
