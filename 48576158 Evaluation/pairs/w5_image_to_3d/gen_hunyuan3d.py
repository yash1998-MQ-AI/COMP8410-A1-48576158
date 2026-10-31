"""Generate meshes with Hunyuan3D 2.1 (run inside the `hunyuan` environment, from the Hunyuan3D-2.1 repo root).

    python gen_hunyuan3d.py --items c01_topiary --seeds 0

Two stages, as in the project's demo: shape (hy3dshape DiT flow-matching pipeline, 3.3B) then texture
(hy3dpaint multi-view PBR pipeline, 6 views, 512 px). Latency counts both stages. Input is the same
background-removed RGBA PNG that TRELLIS.2 receives, so neither model's own background remover is used.
"""

from __future__ import annotations

import inspect
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "hy3dshape"))
sys.path.insert(0, str(REPO / "hy3dpaint"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_common import run  # noqa: E402
from w5_common import HF_ID  # noqa: E402


def load():
    import torch  # noqa: F401
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    try:
        from utils.torchvision_fix import apply_fix

        apply_fix()
    except Exception as exc:
        print("torchvision_fix not applied:", exc)

    shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(HF_ID["hunyuan3d"])
    conf = Hunyuan3DPaintConfig(6, 512)   # max_num_view=6, resolution=512 (the demo's values)
    for attr, rel in (("realesrgan_ckpt_path", "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"),
                      ("multiview_cfg_path", "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"),
                      ("custom_pipeline", "hy3dpaint/hunyuanpaintpbr")):
        if hasattr(conf, attr) and (REPO / rel).exists():
            setattr(conf, attr, str(REPO / rel))
    paint = Hunyuan3DPaintPipeline(conf)
    return {"shape": shape, "paint": paint}


def generate(h, image_path: str, seed: int, out_glb: str) -> dict:
    import torch
    import trimesh
    from PIL import Image

    image = Image.open(image_path).convert("RGBA")
    shape = h["shape"]
    kwargs = {}
    if "generator" in inspect.signature(shape.__call__).parameters:
        kwargs["generator"] = torch.Generator(device="cuda").manual_seed(seed)
    else:
        torch.manual_seed(seed)
    t0 = time.perf_counter()
    mesh = shape(image=image, **kwargs)[0]
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    work = Path(out_glb).with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    white = work / "shape.glb"
    mesh.export(str(white))
    obj_out = work / "textured.obj"
    paint_kwargs = {}
    if "save_glb" in inspect.signature(h["paint"].__call__).parameters:
        paint_kwargs["save_glb"] = False   # bpy is not installed; convert the OBJ with trimesh below
    result = h["paint"](mesh_path=str(white), image_path=image_path, output_mesh_path=str(obj_out), **paint_kwargs)
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    glb = Path(str(result)).with_suffix(".glb") if result else obj_out.with_suffix(".glb")
    if not glb.exists():
        glb = obj_out.with_suffix(".glb")
    if glb.exists():
        shutil.copy(glb, out_glb)
    else:   # the paint pipeline only wrote an .obj: convert so scoring reads the same format for both models
        src = Path(str(result)) if result and Path(str(result)).exists() else obj_out
        trimesh.load(str(src), force="mesh").export(out_glb)
    shutil.rmtree(work, ignore_errors=True)
    return {"shape": t1 - t0, "texture_export": t2 - t1}


if __name__ == "__main__":
    sys.exit(run("hunyuan3d", load, generate))
