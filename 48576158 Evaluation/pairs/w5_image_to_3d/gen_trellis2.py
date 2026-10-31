"""Generate meshes with TRELLIS.2-4B (run inside the `trellis2` environment, from the TRELLIS.2 repo root).

    python gen_trellis2.py --items c01_topiary --seeds 0

Follows the project's own example.py: pipeline.run(image) -> mesh, mesh.simplify(16777216), then
o_voxel.postprocess.to_glb(...) with the README's export settings (1M faces target, 4096 texture).
The input is the same background-removed RGBA PNG that Hunyuan3D receives.
"""

from __future__ import annotations

import inspect
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path.cwd()))   # the TRELLIS.2 repo root (run from there): trellis2 is a folder, not an installed package
# python puts the script's own folder on sys.path, not the current directory, so add the TRELLIS.2 checkout explicitly
sys.path.insert(0, os.environ.get("TRELLIS2_ROOT", "/content/TRELLIS.2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_common import run  # noqa: E402
from w5_common import HF_ID  # noqa: E402


def load():
    import torch  # noqa: F401
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipe = Trellis2ImageTo3DPipeline.from_pretrained(HF_ID["trellis2"])
    pipe.cuda()
    return pipe


def generate(pipe, image_path: str, seed: int, out_glb: str) -> dict:
    import torch
    import o_voxel
    from PIL import Image

    image = Image.open(image_path)
    kwargs = {}
    if "seed" in inspect.signature(pipe.run).parameters:
        kwargs["seed"] = seed
    else:
        torch.manual_seed(seed)
    t0 = time.perf_counter()
    mesh = pipe.run(image, **kwargs)[0]
    mesh.simplify(16777216)  # nvdiffrast limit, as in example.py
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000, texture_size=4096, remesh=True, remesh_band=1, remesh_project=0, verbose=False,
    )
    glb.export(out_glb, extension_webp=False)   # plain PNG textures so every viewer and trimesh can read them
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return {"shape": t1 - t0, "texture_export": t2 - t1}


if __name__ == "__main__":
    sys.exit(run("trellis2", load, generate))
