"""Settings shared by the Week 4 scripts. Both models use exactly these data and split settings."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAIR = "w4"
MODELS = ("nerfacto", "splatfacto")
LABEL = {"nerfacto": "Nerfacto", "splatfacto": "3DGS (splatfacto)"}

NERFSTUDIO_VERSION = "1.1.5"
DOWNSCALE = 2          # 1440x1080 source -> 720x540 for training and evaluation, for both models
EVAL_INTERVAL = 8      # every 8th frame is held out; never trained on
DEFAULT_ITERS = 30000  # each model's own default budget
SEEDS = (0, 1, 2)
MIN_REGISTERED = 0.90  # gate: share of input frames COLMAP must place

def _work_dir() -> Path:
    """Folder for COLMAP input and the processed scene. It must not contain spaces.

    ns-process-data builds its COLMAP commands as unquoted shell strings, so a path such as
    ".../48576158 Evaluation/..." is split in two and COLMAP fails with "image path does not exist".
    The repository folder name has a space (it is required by the submission rules), so the working
    data lives outside it. Override with the W4_WORK environment variable.
    """
    env = os.environ.get("W4_WORK")
    if env:
        return Path(env)
    return Path("/content/w4_work") if Path("/content").exists() else Path(tempfile.gettempdir()) / "w4_work"


WORK_DIR = _work_dir()
SCENE_DIR = WORK_DIR / "scene"
INPUT_DIR = WORK_DIR / "input"
NS_OUT = ROOT / "outputs" / "w4" / "_ns_runs"          # nerfstudio run folders (checkpoints, configs)
RESULTS = ROOT / "results" / "w4.jsonl"
SUMMARY = ROOT / "results" / "w4_summary.json"
SCENE_JSON = ROOT / "results" / "w4_scene.json"


def run_dir(model: str, seed: int, ns_out: Path = NS_OUT) -> Path:
    """Folder nerfstudio writes for one run (config.yml and nerfstudio_models/ live under it)."""
    return ns_out / f"{model}_s{seed}" / model / "run"
