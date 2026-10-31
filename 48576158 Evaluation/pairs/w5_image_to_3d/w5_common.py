"""Settings shared by the Week 5 scripts. Both models get exactly the same inputs, seeds and scoring."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PAIR = "w5"
MODELS = ("trellis2", "hunyuan3d")
LABEL = {"trellis2": "TRELLIS.2-4B", "hunyuan3d": "Hunyuan3D 2.1"}
HF_ID = {"trellis2": "microsoft/TRELLIS.2-4B", "hunyuan3d": "tencent/Hunyuan3D-2.1"}
SEEDS = (0, 1, 2)
WARMUP_SEED = 99          # one throw-away call per model, so CUDA warm-up does not inflate latency

MANIFEST = HERE / "set_c_manifest.csv"
RESULTS = ROOT / "results" / "w5.jsonl"
OUT_ROOT = ROOT / "outputs" / "w5"          # meshes: <model>/<item>_s<seed>.glb
VIEWS_ROOT = ROOT / "outputs" / "w5_figures"  # contact sheets of the rendered views (git-ignored)

# scoring settings (identical for both models)
N_VIEWS = 8               # azimuths 0,45,...,315 degrees
ELEV_DEG = 15.0
RENDER_SIZE = 512
CROP_SIZE = 224           # square crop around the object, fed to CLIP / DINOv2
N_POINTS = 600_000        # surface samples used by the splat renderer
UP_AXIS = {"trellis2": "y", "hunyuan3d": "y"}   # check in the smoke test; change here if a model's mesh lies on its side
CLIP_NAME = "openai/clip-vit-base-patch32"
DINO_NAME = "facebook/dinov2-base"


def _work_dir() -> Path:
    env = os.environ.get("W5_WORK")
    if env:
        return Path(env)
    return Path("/content/w5_work") if Path("/content").exists() else Path(tempfile.gettempdir()) / "w5_work"


WORK_DIR = _work_dir()
INPUT_DIR = WORK_DIR / "inputs"   # <item>.png (RGBA, background removed) and <item>_mask.png


def item_ids() -> list[str]:
    with MANIFEST.open(encoding="utf-8") as fh:
        return [r["item_id"] for r in csv.DictReader(fh)]


def mesh_path(model: str, item: str, seed: int, out_root: Path = OUT_ROOT) -> Path:
    return out_root / model / f"{item}_s{seed}.glb"
