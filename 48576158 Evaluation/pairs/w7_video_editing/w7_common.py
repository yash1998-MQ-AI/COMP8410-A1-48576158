"""Settings shared by the Week 7 scripts.

Both models get the same input clip, the same instruction, the same seed, the same
resolution and the same number of frames. The one declared asymmetry is the edit mask:
Wan2.1-VACE is a control-driven editor and consumes a spatial mask, while Lucy-Edit is
instruction-only and has no mask input. The mask box is still used at scoring time for
BOTH models, so `bg_psnr` (fidelity outside the edit region) is measured identically.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PAIR = "w7"
MODELS = ("wan21_vace", "lucy_edit")
LABEL = {
    "wan21_vace": "Wan2.1-VACE-1.3B (mask-conditioned)",
    "lucy_edit": "Lucy-Edit-Dev (instruction-only)",
}
HF_ID = {
    "wan21_vace": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    "lucy_edit": "decart-ai/Lucy-Edit-Dev",
}

SEEDS = (0,)                 # one seed per item: budget decision, stated in the report
# Matched geometry: both pipelines default to 480x832 / 81 frames / CFG 5.0. Step count is
# reduced from the published default of 50 to 20 for BOTH models (compute budget) - declare this.
HEIGHT, WIDTH = 480, 832
MAX_FRAMES = 81              # must be 4n+1 for the Wan VAE used by both models
STEPS = 20              # budget decision: 50 steps for both models projects to ~55 units, 20 fits 29
GUIDANCE = 5.0
FPS = 24
NEGATIVE = ""

SETTINGS = {
    "wan21_vace": {"height": HEIGHT, "width": WIDTH, "steps": STEPS, "guidance": GUIDANCE,
                   "conditioning": "video + box mask", "offload": "model_cpu_offload"},
    "lucy_edit": {"height": HEIGHT, "width": WIDTH, "steps": STEPS, "guidance": GUIDANCE,
                  "conditioning": "video + text only", "offload": "model_cpu_offload"},
}

N_SCORE_FRAMES = 9
CROP = 224
CLIP_NAME = "openai/clip-vit-base-patch32"
DINO_NAME = "facebook/dinov2-base"

MANIFEST = HERE / "items.csv"
RESULTS = ROOT / "results" / "w7.jsonl"
OUT_ROOT = ROOT / "outputs" / "w7"           # <model>/<item>_s<seed>.mp4

# Edit families. Balanced design: 3 families x 2 clips, so each edit type is replicated.
# Each entry is (family, instruction, edit box as fractions of the frame x0,y0,x1,y1).
# EDIT_PLAN pins an edit to a specific clip (matched by lowercase file stem) because the
# instruction has to make sense for what is actually in frame - cycling a "replace the sofa"
# prompt onto a bathroom clip produces a meaningless result.
EDIT_PLAN = {
    # furniture swap - box covers the piece being replaced
    "living_area": ("furniture", "replace the brown leather sofa with a dark green velvet chesterfield sofa",
                    (0.00, 0.42, 0.60, 1.00)),
    "main_bedroom": ("furniture", "replace the bed with a low oak platform bed with white linen",
                     (0.20, 0.42, 0.90, 0.98)),
    # relight - whole frame, both clips have a clear window
    "bedroom_2": ("relight", "relight the room with warm golden-hour sunlight through the window",
                  (0.00, 0.00, 1.00, 1.00)),
    "bathroom": ("relight", "relight the room with warm golden-hour sunlight through the window",
                 (0.00, 0.00, 1.00, 1.00)),
    # surface change - box covers the lower part of the frame where the floor is
    "kitchen_and_main_living_area": ("surface", "change the floor to wide plank light oak timber flooring",
                                     (0.25, 0.60, 0.72, 1.00)),   # galley floor only: avoids the oven at bottom-left
    "storeroom_and_gym": ("surface", "change the floor to light grey epoxy garage flooring",
                          (0.00, 0.68, 1.00, 1.00)),   # tightened: equipment stands on the floor above this
}

# Fallback for any clip not named in EDIT_PLAN (cycled in order).
EDIT_BANK = [
    ("furniture", "replace the sofa with a dark green velvet chesterfield sofa", (0.18, 0.40, 0.82, 0.95)),
    ("relight", "relight the room with warm golden-hour sunlight through the window", (0.00, 0.00, 1.00, 1.00)),
    ("surface", "change the floor to wide plank light oak timber flooring", (0.00, 0.55, 1.00, 1.00)),
]

# Clips to use, in this order. Anything else under the source folder is ignored.
# Washing_machine_room is held out: it would unbalance the 3 x 2 design.
CLIP_ORDER = ["living_area", "main_bedroom", "bedroom_2", "bathroom",
              "kitchen_and_main_living_area", "storeroom_and_gym"]

START_FRAC = 0.15            # skip the first 15% of each clip: the opening second is the shakiest


def _work_dir() -> Path:
    env = os.environ.get("W7_WORK")
    if env:
        return Path(env)
    return Path("/content/w7_work") if Path("/content").exists() else Path(tempfile.gettempdir()) / "w7_work"


INPUT_DIR = _work_dir() / "inputs"           # 832x480 mp4s, the same file fed to both models


def items() -> list[dict]:
    with MANIFEST.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def item_ids() -> list[str]:
    return [r["item_id"] for r in items()]


def prompt_of(item: str) -> str:
    return next(r["prompt"] for r in items() if r["item_id"] == item)


def box_of(item: str) -> tuple[float, float, float, float]:
    r = next(r for r in items() if r["item_id"] == item)
    return (float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"]))


def clip_path(item: str) -> Path:
    return INPUT_DIR / f"{item}.mp4"


def video_path(model: str, item: str, seed: int, out_root: Path = OUT_ROOT) -> Path:
    return out_root / model / f"{item}_s{seed}.mp4"


def vae_frames(n: int, cap: int = MAX_FRAMES) -> int:
    """Largest 4k+1 frame count that is <= min(n, cap) and at least 5."""
    n = min(n, cap)
    return max(5, ((n - 1) // 4) * 4 + 1)


def box_pixels(box, width: int = WIDTH, height: int = HEIGHT) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (int(round(x0 * width)), int(round(y0 * height)),
            int(round(x1 * width)), int(round(y1 * height)))
