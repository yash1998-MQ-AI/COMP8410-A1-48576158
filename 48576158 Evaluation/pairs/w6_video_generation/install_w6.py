"""Week 6 install, one environment, no compiled extensions (runs on a T4 for the import check, then again on the A100).

    python pairs/w6_video_generation/install_w6.py [--weights]

1. clone Wan2.2 (official repo, used for I2V with TI2V-5B) to /content/Wan2.2
2. pip install its requirements MINUS the pins that would downgrade the rest (torch, numpy, transformers, flash_attn, dashscope)
3. diffusers from source (HunyuanVideo15ImageToVideoPipeline is newer than the last PyPI release) + video IO
4. import loop: try the imports the generation script needs; on ModuleNotFoundError pip install the missing module and retry
5. --weights: download Wan2.2-TI2V-5B to /content/ckpt (HunyuanVideo-1.5 weights download on first use into HF_HOME=/content/hf)
Writes the pinned result to results/w6_ENV.txt (pip freeze) for the evidence folder.
"""
import argparse, os, re, subprocess, sys
from pathlib import Path

WAN = Path("/content/Wan2.2")
DROP = re.compile(r"^(torch|torchvision|numpy|transformers|tokenizers|flash[-_]attn|dashscope|opencv-python)\b", re.I)
MODULE_TO_PIP = {"cv2": "opencv-python-headless", "skimage": "scikit-image", "PIL": "pillow", "yaml": "pyyaml",
                 "sklearn": "scikit-learn", "av": "av"}
CHECK = ("import sys; sys.path.insert(0, '/content/Wan2.2'); import torch, diffusers, transformers; "
         "import wan; from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS; from wan.utils.utils import save_video; "
         "from diffusers import HunyuanVideo15ImageToVideoPipeline; from diffusers.utils import export_to_video; "
         "print('IMPORTS OK torch', torch.__version__, 'diffusers', diffusers.__version__, 'transformers', transformers.__version__)")


def sh(cmd, check=True):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check, capture_output=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", action="store_true")
    a = ap.parse_args()
    if not WAN.exists():
        sh(f"git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git {WAN}")
    sh(f"git -C {WAN} rev-parse HEAD > /content/wan_commit.txt")
    req = [l.strip() for l in (WAN / "requirements.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    keep = [l for l in req if not DROP.match(l)]
    print("dropping:", [l for l in req if DROP.match(l)])
    Path("/content/w6_req.txt").write_text("\n".join(keep) + "\n")
    sh("pip -q install -r /content/w6_req.txt")
    sh("pip -q install git+https://github.com/huggingface/diffusers imageio imageio-ffmpeg opencv-python-headless scikit-image tabulate accelerate sentencepiece protobuf")
    for attempt in range(10):
        r = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        print(out[-600:])
        if r.returncode == 0:
            break
        m = re.search(r"No module named '([\w\.]+)'", out)
        if not m:
            print("IMPORT FAILED with a non-missing-module error: read the text above"); sys.exit(1)
        mod = m.group(1).split(".")[0]
        sh(f"pip -q install {MODULE_TO_PIP.get(mod, mod)}")
    else:
        print("gave up after 10 attempts"); sys.exit(1)
    if a.weights:
        os.makedirs("/content/ckpt", exist_ok=True)
        sh("python -c \"from huggingface_hub import snapshot_download as s; s('Wan-AI/Wan2.2-TI2V-5B', local_dir='/content/ckpt/Wan2.2-TI2V-5B')\"")
    print("INSTALL DONE")


if __name__ == "__main__":
    main()
