"""Week 2: generate edited images for one model on the shared item list.

    python pairs/w2_image_editing/run_w2.py --model flux1-kontext-dev --limit 1   # smoke test
    python pairs/w2_image_editing/run_w2.py --model flux1-kontext-dev --offload
    python pairs/w2_image_editing/run_w2.py --model qwen-image-edit-2509 --offload

Each call is timed (latency, peak VRAM) into results/w2.jsonl; finished items are
skipped so an interrupted Colab session resumes. Model id "identity" returns the
source unchanged and exists to test the pipeline on a CPU.

Fixed settings (recorded in every result row): one seed, same resized input, same prompt.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from a1eval import done_items, timed  # noqa: E402

MODELS = {
    "qwen-image-edit-2509": {"repo": "Qwen/Qwen-Image-Edit-2509", "steps": 40, "cfg": 4.0},
    "flux1-kontext-dev": {"repo": "black-forest-labs/FLUX.1-Kontext-dev", "steps": 28, "cfg": 2.5},
    "identity": {"repo": "-", "steps": 0, "cfg": 0.0},
}


def fit(img: Image.Image, long_side: int = 1024, mult: int = 16) -> Image.Image:
    s = long_side / max(img.size)
    w, h = (max(mult, round(img.width * s / mult) * mult), max(mult, round(img.height * s / mult) * mult))
    return img.resize((w, h), Image.LANCZOS)


def build(model: str, offload: bool):
    if model == "identity":
        return None
    import torch

    repo = MODELS[model]["repo"]
    if model == "qwen-image-edit-2509":
        from diffusers import QwenImageEditPlusPipeline

        pipe = QwenImageEditPlusPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    else:
        from diffusers import FluxKontextPipeline

        pipe = FluxKontextPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def edit(model: str, pipe, img: Image.Image, prompt: str, seed: int) -> Image.Image:
    if model == "identity":
        return img.copy()
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    cfg = MODELS[model]
    if model == "qwen-image-edit-2509":
        out = pipe(image=[img], prompt=prompt, negative_prompt=" ", true_cfg_scale=cfg["cfg"],
                   num_inference_steps=cfg["steps"], generator=gen)
    else:
        out = pipe(image=img, prompt=prompt, guidance_scale=cfg["cfg"], num_inference_steps=cfg["steps"],
                   width=img.width, height=img.height, generator=gen)
    return out.images[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--items", default=str(Path(__file__).with_name("items.csv")))
    ap.add_argument("--out-root", default=str(ROOT / "outputs"))
    ap.add_argument("--results", default=str(ROOT / "results" / "w2.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--long-side", type=int, default=1024)
    ap.add_argument("--offload", action="store_true", help="CPU offload to fit smaller GPUs (slower)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N items (smoke test)")
    args = ap.parse_args()

    items = list(csv.DictReader(open(args.items, encoding="utf-8")))
    if args.limit:
        items = items[: args.limit]
    out_dir = Path(args.out_root) / "w2" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    done = done_items(args.results, args.model)
    todo = [it for it in items if not (it["item_id"] in done and (out_dir / f"{it['item_id']}.png").exists())]
    print(f"{args.model}: {len(items)} items, {len(todo)} to run")
    if not todo:
        return

    pipe = build(args.model, args.offload)
    for it in todo:
        src = Path(it["source"])
        src = src if src.is_absolute() else ROOT / src
        img = fit(Image.open(src).convert("RGB"), args.long_side)
        with timed(args.results, "w2", args.model, it["item_id"], seed=args.seed,
                   prompt=it["prompt"], edit_type=it["edit_type"], repo=MODELS[args.model]["repo"],
                   steps=MODELS[args.model]["steps"], cfg=MODELS[args.model]["cfg"],
                   size=list(img.size), offload=args.offload):
            result = edit(args.model, pipe, img, it["prompt"], args.seed)
        result.save(out_dir / f"{it['item_id']}.png")
        print("done", it["item_id"])


if __name__ == "__main__":
    main()
