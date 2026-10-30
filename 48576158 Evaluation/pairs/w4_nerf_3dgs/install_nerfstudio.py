"""Install nerfstudio 1.1.5 on a recent Python (Colab ships 3.13) without letting it replace torch.

Why not a plain `pip install nerfstudio`: it requires `fpsample`, which has no wheel for Python 3.13 and is
only called by the optional "fps" camera sampler (the default sampler is "random"). A stub module stands in
for it. torch and torchvision are left alone so the GPU build the runtime ships with keeps working.

    python pairs/w4_nerf_3dgs/install_nerfstudio.py
"""

from __future__ import annotations

import importlib.metadata as md
import re
import site
import subprocess
import sys
from pathlib import Path

VERSION = "1.1.5"
SKIP = {"fpsample", "torch", "torchvision", "protobuf", "pymeshlab", "newrawpy"}

STUB = '''"""Stand-in for fpsample (no wheel for this Python). Only used by train_cameras_sampling_strategy="fps"."""


def bucket_fps_kdline_sampling(*args, **kwargs):
    raise RuntimeError("fpsample is stubbed; use the default random camera sampler")
'''


def requirement_name(req: str) -> str:
    """'viser ==0.2.7 ; extra == "x"' -> 'viser'"""
    return re.split(r"[ ;<>=!\[~]", req.strip(), maxsplit=1)[0].lower()


def pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


def main() -> None:
    pip(f"nerfstudio=={VERSION}", "--no-deps")
    reqs = []
    for r in md.requires("nerfstudio") or []:
        if "extra ==" in r:
            continue
        if requirement_name(r) in SKIP:
            continue
        reqs.append(r)
    print(f"installing {len(reqs)} requirements")
    pip(*reqs)

    target = Path(site.getsitepackages()[0]) / "fpsample.py"
    target.write_text(STUB, encoding="utf-8")
    print("wrote", target)


if __name__ == "__main__":
    main()
