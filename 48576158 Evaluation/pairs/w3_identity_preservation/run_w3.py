"""Week 3: generate listing-style images of one identity with PuLID v1.1 or InstantID.

    python pairs/w3_identity_preservation/run_w3.py --model pulid-v1.1 --limit 1 --seeds 0   # smoke test
    python pairs/w3_identity_preservation/run_w3.py --model instantid
    python pairs/w3_identity_preservation/run_w3.py --model identity                          # CPU stub for tests

Each model runs as its authors ship it (system-level comparison, recorded in every row):
  pulid-v1.1  official PuLIDPipeline defaults: Lykon/dreamshaper-xl-lightning, dpmpp_sde, 4 steps, guidance 1.2, id_scale 1.0
  instantid   official quick start (infer.py): stabilityai/stable-diffusion-xl-base-1.0, IdentityNet (keypoints from the
              reference face) + IP-Adapter, 30 steps, guidance 5, controlnet scale 0.8, ip_adapter scale 0.8

The third-party repos are cloned by the notebook to /content/third_party (PuLID, InstantID).
Latency includes face analysis and identity-embedding extraction as well as sampling.
Finished (item, seed) pairs are skipped, so an interrupted Colab session resumes.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from a1eval import done_items, timed  # noqa: E402

PULID_COMMIT = "1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa"
INSTANTID_COMMIT = "2145b67f9607da6234702063692330185f374486"
MODELS = {
    "pulid-v1.1": {"base": "Lykon/dreamshaper-xl-lightning", "steps": 4, "cfg": 1.2, "id_scale": 1.0, "commit": PULID_COMMIT},
    "instantid": {"base": "stabilityai/stable-diffusion-xl-base-1.0", "steps": 30, "cfg": 5.0, "id_scale": 0.8, "commit": INSTANTID_COMMIT},
    "identity": {"base": "-", "steps": 0, "cfg": 0.0, "id_scale": 0.0, "commit": "-"},
}


def gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def shim_basicsr() -> None:
    """PuLID imports two tiny helpers from basicsr, which does not install on current Python/torch. Provide them."""
    if "basicsr" in sys.modules:
        return
    import cv2
    import torch

    def img2tensor(img, bgr2rgb=True, float32=True):
        if img.ndim == 3 and img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return t.float() if float32 else t

    def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
        t = tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        t = (t - min_max[0]) / (min_max[1] - min_max[0])
        img = t.numpy().transpose(1, 2, 0)
        if img.shape[2] == 3 and rgb2bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return (img * 255.0).round().astype(out_type)

    pkg, utils = types.ModuleType("basicsr"), types.ModuleType("basicsr.utils")
    utils.img2tensor, utils.tensor2img = img2tensor, tensor2img
    pkg.utils = utils
    sys.modules["basicsr"], sys.modules["basicsr.utils"] = pkg, utils


class Identity:
    """CPU stub: returns the reference unchanged (tests the plumbing only)."""

    def __call__(self, ref: Image.Image, prompt: str, negative: str, seed: int) -> Image.Image:
        return ref.copy()


class PuLID:
    def __init__(self, third_party: Path):
        import torch

        repo = third_party / "PuLID"
        shim_basicsr()
        sys.path.insert(0, str(repo))
        self._cwd = os.getcwd()
        os.chdir(repo)  # the repo uses ./models and ./eva_clip relative paths
        from pulid.pipeline_v1_1 import PuLIDPipeline

        cfg = MODELS["pulid-v1.1"]
        self.pipe = PuLIDPipeline(sdxl_repo=cfg["base"], sampler="dpmpp_sde")
        self.cfg = cfg
        self.torch = torch

    def __call__(self, ref: Image.Image, prompt: str, negative: str, seed: int) -> Image.Image:
        uncond, cond = self.pipe.get_id_embedding([np.array(ref.convert("RGB"))])
        out = self.pipe.inference(prompt, (1, 1024, 1024), negative, cond, uncond, id_scale=self.cfg["id_scale"],
                                  guidance_scale=self.cfg["cfg"], steps=self.cfg["steps"], seed=seed)
        self.pipe.debug_img_list = []  # the repo appends debug crops on every call
        return out[0]


class InstantID:
    def __init__(self, third_party: Path):
        import torch
        from huggingface_hub import hf_hub_download, snapshot_download

        repo = third_party / "InstantID"
        sys.path.insert(0, str(repo))
        ckpt = repo / "checkpoints"
        for f in ("ControlNetModel/config.json", "ControlNetModel/diffusion_pytorch_model.safetensors", "ip-adapter.bin"):
            hf_hub_download(repo_id="InstantX/InstantID", filename=f, local_dir=str(ckpt))
        snapshot_download("DIAMONIK7777/antelopev2", local_dir=str(repo / "models" / "antelopev2"))  # same mirror PuLID's code uses

        from diffusers.models import ControlNetModel
        from insightface.app import FaceAnalysis
        # Recent diffusers moved MultiControlNetModel; alias the old module path the repo imports.
        import diffusers.models.controlnets.multicontrolnet as _mcn
        sys.modules.setdefault("diffusers.pipelines.controlnet.multicontrolnet", _mcn)
        from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

        self.draw_kps = draw_kps
        self.app = FaceAnalysis(name="antelopev2", root=str(repo), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        cfg = MODELS["instantid"]
        controlnet = ControlNetModel.from_pretrained(str(ckpt / "ControlNetModel"), torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(cfg["base"], controlnet=controlnet, torch_dtype=torch.float16)
        self.pipe.cuda()
        self.pipe.load_ip_adapter_instantid(str(ckpt / "ip-adapter.bin"))
        self.cfg = cfg
        self.torch = torch

    def __call__(self, ref: Image.Image, prompt: str, negative: str, seed: int) -> Image.Image:
        import cv2

        ref = ref.convert("RGB").resize((1024, 1024))
        faces = self.app.get(cv2.cvtColor(np.array(ref), cv2.COLOR_RGB2BGR))
        if not faces:
            raise RuntimeError("no face found in reference")
        face = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
        kps = self.draw_kps(ref, face["kps"])
        gen = self.torch.Generator(device="cuda").manual_seed(seed)
        return self.pipe(prompt=prompt, negative_prompt=negative, image_embeds=face["embedding"], image=kps,
                         controlnet_conditioning_scale=self.cfg["id_scale"], ip_adapter_scale=self.cfg["id_scale"],
                         num_inference_steps=self.cfg["steps"], guidance_scale=self.cfg["cfg"], generator=gen).images[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--items", default=str(Path(__file__).with_name("items.csv")))
    ap.add_argument("--out-root", default=str(ROOT / "outputs"))
    ap.add_argument("--results", default=str(ROOT / "results" / "w3.jsonl"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--third-party", default="/content/third_party")
    ap.add_argument("--limit", type=int, default=0, help="only the first N items (smoke test)")
    args = ap.parse_args()

    # resolve every path before PuLID changes the working directory
    results = str(Path(args.results).resolve())
    out_dir = Path(args.out_root).resolve() / "w3" / args.model
    items_path = Path(args.items).resolve()
    third_party = Path(args.third_party).resolve()
    items = list(csv.DictReader(open(items_path, encoding="utf-8")))
    if args.limit:
        items = items[: args.limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    done = done_items(results, args.model)
    jobs = []
    for it in items:
        for seed in args.seeds:
            key = f"{it['item_id']}__s{seed}"
            if not (key in done and (out_dir / f"{key}.png").exists()):
                jobs.append((it, seed, key))
    print(f"{args.model}: {len(items) * len(args.seeds)} generations, {len(jobs)} to run")
    if not jobs:
        return

    ref_cache: dict[str, Image.Image] = {}
    for it, _, _ in jobs:
        p = Path(it["ref"])
        ref_cache[it["ref"]] = Image.open(p if p.is_absolute() else ROOT / p).convert("RGB")

    gen = {"identity": lambda: Identity(), "pulid-v1.1": lambda: PuLID(third_party), "instantid": lambda: InstantID(third_party)}[args.model]()
    cfg, gpu = MODELS[args.model], gpu_name()
    for it, seed, key in jobs:
        with timed(results, "w3", args.model, key, seed=seed, item=it["item_id"], identity=it["identity"], scenario=it["scenario"],
                   prompt=it["prompt"], base=cfg["base"], steps=cfg["steps"], cfg=cfg["cfg"], id_scale=cfg["id_scale"],
                   commit=cfg["commit"], gpu=gpu):
            img = gen(ref_cache[it["ref"]], it["prompt"], it["negative"], seed)
        img.save(out_dir / f"{key}.png")
        print("done", key, flush=True)


if __name__ == "__main__":
    main()
