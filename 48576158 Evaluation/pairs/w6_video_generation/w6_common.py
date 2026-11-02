"""Settings shared by the Week 6 scripts. Both models get the same photo, prompt, seed, frame count and fps."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PAIR = "w6"
MODELS = ("wan22", "hunyuanvideo15")
LABEL = {"wan22": "Wan2.2 TI2V-5B", "hunyuanvideo15": "HunyuanVideo 1.5 (480p I2V, step-distilled)"}
HF_ID = {"wan22": "Wan-AI/Wan2.2-TI2V-5B", "hunyuanvideo15": "tencent/HunyuanVideo-1.5"}

SEEDS = (0,)                 # one seed per item: budget decision (40-unit plan), stated in the report
WARMUP_SEED = 99
NUM_FRAMES = 73              # 4n+1 for both video VAEs; 3.04 s at 24 fps
FPS = 24
# Each model runs at its own recommended setting. This is a system-level comparison, not a matched-pixel one:
# Wan 704x1280 (native 720p-class), Hunyuan 480x848 with 8 distilled steps. Scoring downsizes every frame to 224 px.
SETTINGS = {
    "wan22": {"height": 704, "width": 1280, "steps": 50, "guidance": 5.0, "offload": "none"},
    "hunyuanvideo15": {"height": 480, "width": 848, "steps": 8, "guidance": 1.0, "offload": "model_cpu_offload"},
}
N_SCORE_FRAMES = 9           # evenly spaced frames used by CLIP / DINOv2
CROP = 224
CLIP_NAME = "openai/clip-vit-base-patch32"
DINO_NAME = "facebook/dinov2-base"

MANIFEST = HERE / "items.csv"
RESULTS = ROOT / "results" / "w6.jsonl"
OUT_ROOT = ROOT / "outputs" / "w6"            # <model>/<item>_s<seed>.mp4


def _work_dir() -> Path:
    env = os.environ.get("W6_WORK")
    if env:
        return Path(env)
    return Path("/content/w6_work") if Path("/content").exists() else Path(tempfile.gettempdir()) / "w6_work"


INPUT_DIR = _work_dir() / "inputs"          # 16:9 PNGs, same file fed to both models


def items() -> list[dict]:
    with MANIFEST.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def item_ids() -> list[str]:
    return [r["item_id"] for r in items()]


def prompt_of(item: str) -> str:
    return next(r["prompt"] for r in items() if r["item_id"] == item)


def video_path(model: str, item: str, seed: int, out_root: Path = OUT_ROOT) -> Path:
    return out_root / model / f"{item}_s{seed}.mp4"
