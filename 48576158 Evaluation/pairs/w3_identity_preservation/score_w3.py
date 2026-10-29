"""Week 3: score every generated image and append metric rows to results/w3.jsonl.

Metrics (one row per generated image):
  id_facenet          primary identity match: cosine of FaceNet (VGGFace2) embeddings,
                      generated face vs the reference photo the model was given.
                      NaN when no face is found, so failures stay visible.
  face_found          1.0 if a face was detected in the generated image, else 0.0.
                      id_facenet alone cannot separate "wrong face" from "no face".
  id_arcface          secondary identity match: ArcFace (insightface buffalo_l).
                      Reported as a check, not as the headline: both models condition
                      on ArcFace embeddings internally, so this metric is closer to
                      their training objective than FaceNet is.
  clip_t              prompt adherence: CLIP text-image similarity with the item prompt.
                      Identity can always be bought by ignoring the prompt, so an
                      identity score is only meaningful read next to this.
  id_facenet_heldout  identity "me" only: mean cosine against photos of the same person
                      that were never given to either model (data/processed/w3_refs_extra).
                      Guards against a model simply copying the reference image.

    python pairs/w3_identity_preservation/score_w3.py
    python pairs/w3_identity_preservation/score_w3.py --calibrate   # scale reference values
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from a1eval import Record, append, load  # noqa: E402
from a1eval import metrics as M  # noqa: E402

ALL = ["id_facenet", "face_found", "id_arcface", "clip_t", "id_facenet_heldout"]


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def calibrate(items: list[dict], extra_dir: Path, out_path: Path) -> None:
    """Write the two ends of the FaceNet scale so the item scores can be read.

    within_person  same person, different photo (the realistic ceiling)
    between_person different people (the floor a model must stay well above)
    """
    refs = {it["identity"]: _resolve(it["ref"]) for it in items}
    emb = {}
    for name, path in refs.items():
        e = M.facenet_embedding(M.load_image(path))
        if e is not None:
            emb[name] = e
    for p in sorted(extra_dir.glob("*.jpg")):
        e = M.facenet_embedding(M.load_image(p))
        if e is not None:
            emb[p.stem] = e

    within = [M.cosine(emb["me"], emb[k]) for k in emb if k.startswith("me_extra") and "me" in emb]
    between = [M.cosine(emb[a], emb[b]) for a, b in combinations(sorted(refs), 2) if a in emb and b in emb]
    out = {
        "within_person": {"pairs": len(within), "mean": sum(within) / len(within) if within else None,
                          "values": [round(v, 4) for v in within]},
        "between_person": {"pairs": len(between), "mean": sum(between) / len(between) if between else None,
                           "values": [round(v, 4) for v in between]},
        "note": "FaceNet cosine, InceptionResnetV1/VGGFace2, MTCNN crop. Identities without a detected face are omitted.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(Path(__file__).with_name("items.csv")))
    ap.add_argument("--out-root", default=str(ROOT / "outputs"))
    ap.add_argument("--results", default=str(ROOT / "results" / "w3.jsonl"))
    ap.add_argument("--extra-dir", default=str(ROOT / "data" / "processed" / "w3_refs_extra"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--metrics", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--models", nargs="*", help="default: every folder under outputs/w3")
    ap.add_argument("--calibrate", action="store_true", help="also write results/w3_calibration.json")
    args = ap.parse_args()

    items = list(csv.DictReader(open(args.items, encoding="utf-8")))
    root = Path(args.out_root) / "w3"
    extra_dir = Path(args.extra_dir)

    if args.calibrate:
        calibrate(items, extra_dir, Path(args.results).with_name("w3_calibration.json"))

    models = args.models or sorted(p.name for p in root.iterdir() if p.is_dir())
    have = {(r.model, r.item_id, r.metric) for r in load(args.results)}

    ref_emb: dict[str, object] = {}   # reference photo -> FaceNet embedding
    ref_img: dict[str, object] = {}   # reference photo -> pixels, for ArcFace
    heldout: list = []

    def ref_of(path: Path):
        k = str(path)
        if k not in ref_img:
            ref_img[k] = M.load_image(path)
            ref_emb[k] = M.facenet_embedding(ref_img[k])
        return ref_img[k], ref_emb[k]

    n = 0
    for model in models:
        for it in items:
            for seed in args.seeds:
                key = f"{it['item_id']}__s{seed}"
                img_path = root / model / f"{key}.png"
                if not img_path.exists():
                    continue
                todo = [m for m in args.metrics if (model, key, m) not in have]
                if it["identity"] != "me" and "id_facenet_heldout" in todo:
                    todo.remove("id_facenet_heldout")
                if not todo:
                    continue

                gen = M.load_image(img_path)
                r_img, r_emb = ref_of(_resolve(it["ref"]))
                g_emb = M.facenet_embedding(gen) if {"id_facenet", "face_found", "id_facenet_heldout"} & set(todo) else None
                meta = {"identity": it["identity"], "scenario": it["scenario"], "item": it["item_id"]}

                for m in todo:
                    if m == "face_found":
                        v = 1.0 if g_emb is not None else 0.0
                    elif m == "id_facenet":
                        v = float("nan") if (g_emb is None or r_emb is None) else M.cosine(g_emb, r_emb)
                    elif m == "id_arcface":
                        v = M.identity_similarity(r_img, gen)
                    elif m == "clip_t":
                        v = M.clip_text_image(gen, it["prompt"])
                    else:  # id_facenet_heldout
                        if not heldout:
                            for p in sorted(extra_dir.glob("*.jpg")):
                                e = M.facenet_embedding(M.load_image(p))
                                if e is not None:
                                    heldout.append(e)
                        if g_emb is None or not heldout:
                            v = float("nan")
                        else:
                            sims = [M.cosine(g_emb, e) for e in heldout]
                            v = sum(sims) / len(sims)
                    append(args.results, Record("w3", model, key, m, float(v), seed, meta))
                    n += 1
    print(f"appended {n} metric rows to {args.results}")


if __name__ == "__main__":
    main()
