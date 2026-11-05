"""Week 7 install. Both models are diffusers-native, so there is no repo to clone.

    python pairs/w7_video_editing/install_w7.py

Verified against diffusers 0.42.1, which ships both WanVACEPipeline and LucyEditPipeline.
Prints INSTALL DONE on success; any missing import is reported with the pip name to add.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

PIP = [
    "diffusers>=0.42.1",
    "transformers>=4.49",
    "accelerate",
    "safetensors",
    "sentencepiece",
    "ftfy",
    "imageio[ffmpeg]",
    "imageio-ffmpeg",
    "opencv-python-headless",
    "scikit-image",
    "pandas",
    "matplotlib",
    "tabulate",
]

# import name -> pip name
NEEDED = {
    "diffusers": "diffusers>=0.42.1",
    "transformers": "transformers>=4.49",
    "accelerate": "accelerate",
    "imageio": "imageio[ffmpeg]",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "cv2": "opencv-python-headless",
    "skimage": "scikit-image",
    "PIL": "pillow",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "tabulate": "tabulate",
}


def pip_install(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)


def main() -> int:
    pip_install(["-U", *PIP])

    missing = []
    for mod, pkg in NEEDED.items():
        try:
            importlib.import_module(mod)
        except Exception as exc:
            print(f"  missing {mod} ({type(exc).__name__}) -> installing {pkg}", flush=True)
            pip_install([pkg])
            try:
                importlib.invalidate_caches()
                importlib.import_module(mod)
            except Exception as exc2:
                missing.append(f"{mod} ({pkg}): {exc2!r}")

    import diffusers

    print(f"\ndiffusers {diffusers.__version__}")
    for cls in ("WanVACEPipeline", "LucyEditPipeline", "AutoencoderKLWan"):
        ok = hasattr(diffusers, cls)
        print(f"  {cls}: {'OK' if ok else 'NOT FOUND'}")
        if not ok:
            missing.append(f"diffusers.{cls} not available - upgrade diffusers")

    try:
        import torch

        print(f"\ntorch {torch.__version__}  cuda={torch.cuda.is_available()}  "
              f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    except Exception as exc:
        missing.append(f"torch: {exc!r}")

    if missing:
        print("\nPROBLEMS:")
        for m in missing:
            print("  -", m)
        return 1
    print("\nINSTALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
