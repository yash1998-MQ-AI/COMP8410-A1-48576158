"""Week 3: generate the synthetic reference faces (no real person involved).

    python pairs/w3_identity_preservation/make_synthetic_refs.py --out data/processed/w3_refs_candidates

Writes a few candidates per synthetic identity with SDXL base 1.0 (CreativeML Open RAIL++-M).
Look at them, then copy the one you pick to data/processed/w3_refs/<id>.jpg.
The identities differ in apparent age, gender presentation and skin tone so the
fairness check later is not run on one kind of face. Descriptions are of generated
people, not real ones.
"""

from __future__ import annotations

import argparse
from pathlib import Path

IDENTITIES = {
    "syn_a": "a woman in her mid thirties with East Asian features and shoulder-length black hair",
    "syn_b": "a man in his fifties with dark brown skin, short grey-flecked hair and a trimmed beard",
    "syn_c": "a woman in her twenties with light skin, freckles and wavy red-brown hair",
}
TEMPLATE = ("studio portrait photo of {who}, facing the camera, neutral relaxed expression, plain light grey background, "
            "soft even lighting, sharp focus, shot on 85mm lens")
NEGATIVE = "lowres, blurry, deformed, extra fingers, bad anatomy, watermark, text, sunglasses, hat, cartoon, painting"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/w3_refs_candidates")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()

    import torch
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16").to("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ident, who in IDENTITIES.items():
        for seed in args.seeds:
            g = torch.Generator("cpu").manual_seed(seed)
            img = pipe(TEMPLATE.format(who=who), negative_prompt=NEGATIVE, num_inference_steps=30, guidance_scale=6.0,
                       width=1024, height=1024, generator=g).images[0]
            img.save(out / f"{ident}_cand{seed}.png")
            print("wrote", f"{ident}_cand{seed}.png")


if __name__ == "__main__":
    main()
