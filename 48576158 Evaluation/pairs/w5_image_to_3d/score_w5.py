"""Week 5 scoring: render every generated mesh from 8 fixed views and compare with the input photo.

    python pairs/w5_image_to_3d/score_w5.py --models trellis2 hunyuan3d --seeds 0 1 2

Per (model, item, seed) it logs, to results/w5.jsonl:
  clip_i_best / clip_i_mean      CLIP image-image similarity, render vs photo (best view / mean of 8 views)
  dino_best / dino_mean          the same with DINOv2 (better at shape and layout than CLIP)
  sil_iou_best                   silhouette IoU against the photo cut-out, best of 8 views
  clip_i_best_mismatch, dino_best_mismatch
                                 the same score against a DIFFERENT item's photo: shows the metric can tell objects apart
  n_faces, n_vertices, watertight, n_components, textured, glb_size_mb
There is no ground-truth 3D scan, so no Chamfer distance; these scores measure agreement with the photo only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from a1eval import Record, append, load  # noqa: E402
import render_splat as RS  # noqa: E402
from w5_common import (  # noqa: E402
    CLIP_NAME, CROP_SIZE, DINO_NAME, ELEV_DEG, INPUT_DIR, MODELS, N_POINTS, N_VIEWS, OUT_ROOT, PAIR, RENDER_SIZE,
    RESULTS, SEEDS, UP_AXIS, VIEWS_ROOT, item_ids, mesh_path,
)


class Embedder:
    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        from a1eval import metrics as M

        self.torch = torch
        self.clip, self.clip_proc, self.dev = M._clip(CLIP_NAME)
        self._img_embeds = M._image_embeds
        self.dino = AutoModel.from_pretrained(DINO_NAME).to(self.dev).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained(DINO_NAME)

    def __call__(self, images: list[np.ndarray]):
        torch = self.torch
        pil = [Image.fromarray(i) for i in images]
        with torch.no_grad():
            c = self.clip_proc(images=pil, return_tensors="pt").to(self.dev)
            ce = self._img_embeds(self.clip, c["pixel_values"])
            d = self.dino_proc(images=pil, return_tensors="pt", do_resize=False, do_center_crop=False).to(self.dev)
            de = self.dino(**d).last_hidden_state[:, 0]
        ce = (ce / ce.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        de = (de / de.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        return ce, de


def render_views(mesh_file: Path, up: str):
    """Returns (list of cropped views, list of masks, stats dict, textured flag)."""
    mesh = RS.load_mesh(mesh_file)
    stats = RS.mesh_stats(mesh)
    pts, col, nrm, textured = RS.sample_points(mesh, N_POINTS, seed=0)
    pts, nrm = RS.orient(pts, nrm, up)
    radius = float(np.linalg.norm(pts, axis=1).max())
    views, masks = [], []
    for k in range(N_VIEWS):
        rgb, m = RS.render(pts, col, nrm, 360.0 * k / N_VIEWS, ELEV_DEG, RENDER_SIZE, radius)
        v, mm = RS.crop_square(rgb, m, CROP_SIZE)
        views.append(v)
        masks.append(mm)
    stats["textured"] = float(textured)
    stats["glb_size_mb"] = mesh_file.stat().st_size / 1e6
    return views, masks, stats


def view_sheet(photo: np.ndarray, views: list[np.ndarray]) -> Image.Image:
    tiles = [photo] + views
    cols = 3 if len(tiles) == 9 else 5
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * CROP_SIZE, rows * CROP_SIZE), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(Image.fromarray(t), ((i % cols) * CROP_SIZE, (i // cols) * CROP_SIZE))
    return sheet


def limit_memory(gb: float = 40.0) -> None:
    """A runaway allocation should raise MemoryError (with a traceback) rather than get the whole kernel OOM-killed."""
    try:
        import resource

        # CUDA reserves a lot of address space, so cap the data segment, not the address space
        resource.setrlimit(resource.RLIMIT_DATA, (int(gb * 1024**3), resource.RLIM_INFINITY))
    except Exception as exc:
        print("memory guard not set:", exc)


def main() -> int:
    limit_memory(40.0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--items", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--inputs", type=Path, default=INPUT_DIR)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--views-root", type=Path, default=VIEWS_ROOT)
    ap.add_argument("--rescore", action="store_true", help="score again even if rows exist")
    args = ap.parse_args()

    all_items = item_ids()
    items = args.items or all_items
    emb = Embedder()
    photo = {}
    for it in all_items:
        p = args.inputs / f"{it}.png"
        if p.exists():
            img, mask = RS.load_photo(p, CROP_SIZE)
            ce, de = emb([img])
            photo[it] = {"img": img, "mask": mask, "clip": ce[0], "dino": de[0]}
    if not photo:
        print("no input photos found; run run_w5.py prep")
        return 2
    present = sorted(photo)

    scored = {(r.model, r.item_id, r.seed) for r in load(args.results) if r.metric == "dino_best"}
    n = 0
    for model in args.models:
        for it in items:
            for seed in args.seeds:
                f = mesh_path(model, it, seed, args.out_root)
                if not f.exists() or it not in photo or ((model, it, seed) in scored and not args.rescore):
                    continue
                try:
                    views, masks, stats = render_views(f, UP_AXIS[model])
                except Exception as exc:
                    print(f"{model} {it} s{seed}: could not render ({exc!r})")
                    append(args.results, Record(PAIR, model, it, "render_ok", 0.0, seed, {"error": repr(exc)[:200]}))
                    continue
                ce, de = emb(views)
                ph = photo[it]
                other = photo[present[(present.index(it) + 1) % len(present)]]
                clip_s, dino_s = ce @ ph["clip"], de @ ph["dino"]
                ious = [RS.iou(m, ph["mask"]) for m in masks]
                row = {
                    "clip_i_best": clip_s.max(), "clip_i_mean": clip_s.mean(),
                    "dino_best": dino_s.max(), "dino_mean": dino_s.mean(),
                    "sil_iou_best": max(ious),
                    "clip_i_best_mismatch": (ce @ other["clip"]).max(),
                    "dino_best_mismatch": (de @ other["dino"]).max(),
                    "render_ok": 1.0,
                }
                row.update(stats)
                meta = {"best_view_dino": int(np.argmax(dino_s)), "best_view_iou": int(np.argmax(ious))}
                for k, v in row.items():
                    append(args.results, Record(PAIR, model, it, k, float(v), seed, meta))
                sheet_dir = args.views_root / model
                sheet_dir.mkdir(parents=True, exist_ok=True)
                view_sheet(ph["img"], views).save(sheet_dir / f"{it}_s{seed}.jpg", quality=88)
                n += 1
                print(f"{model} {it} s{seed}: dino {row['dino_best']:.3f} clip {row['clip_i_best']:.3f} "
                      f"iou {row['sil_iou_best']:.3f} faces {stats['n_faces']}")
    print(f"scored {n} meshes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
