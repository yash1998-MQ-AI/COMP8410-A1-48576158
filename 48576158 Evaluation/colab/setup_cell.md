# Colab setup

Colab cannot see your PC. Code comes from GitHub; private data (git-ignored) comes from Google Drive.
Sessions can disconnect, so results and outputs are saved on Drive and every run resumes from `results/`.

Easiest route: push this repo, then in Colab use File > Open notebook > GitHub > `yash1998-MQ-AI/COMP8410-A1-48576158` (tick 'Include private repos') and pick `48576158 Evaluation/colab/a1_week2.ipynb`. The dialog only lists .ipynb files, so before this notebook existed it showed 'No results'.

## One-time preparation
1. PC: commit and push everything (`git add .`, check `git status` shows no `data/raw` or `data/processed`, commit, push).
2. Drive: create `My Drive/comp8410_a1/w2_listing/` and upload the 15 images from `data/processed/w2_listing/`.
3. Colab: open https://colab.research.google.com, New notebook, then
   - Runtime > Change runtime type > A100 GPU (or L4 if A100 is not offered). Note the compute units per hour shown in the resources panel.
   - Left sidebar key icon (Secrets): add `HF_TOKEN` (Hugging Face token; first accept the FLUX.1 Kontext [dev] licence on its model page).
     If the repo is private, also add `GITHUB_TOKEN` (fine-grained token, read-only, this repo only). Switch "Notebook access" on for each.

## Cell 1: Drive, code, packages
```python
from google.colab import drive, userdata
import os
drive.mount('/content/drive')
BASE = '/content/drive/MyDrive/comp8410_a1'
for d in ['results', 'outputs', 'hf_cache', 'w2_listing']:
    os.makedirs(f'{BASE}/{d}', exist_ok=True)

REPO = 'yash1998-MQ-AI/COMP8410-A1-48576158'
try:
    tok = userdata.get('GITHUB_TOKEN')      # only needed for a private repo
except Exception:
    tok = None
url = f'https://{tok}@github.com/{REPO}.git' if tok else f'https://github.com/{REPO}.git'
!rm -rf /content/a1 && git clone {url} /content/a1
%cd "/content/a1/48576158 Evaluation"

!pip -q install -r requirements.txt
!pip -q install -U transformers accelerate sentencepiece protobuf
!pip -q install git+https://github.com/huggingface/diffusers
```

## Cell 2: link Drive folders, log in, check the GPU
```python
os.environ['HF_HOME'] = f'{BASE}/hf_cache'          # weights are cached on Drive
!rm -rf results outputs
!ln -s {BASE}/results results
!ln -s {BASE}/outputs outputs
!mkdir -p data/processed && ln -sfn {BASE}/w2_listing data/processed/w2_listing

from huggingface_hub import login
login(userdata.get('HF_TOKEN'))

!nvidia-smi --query-gpu=name,memory.total --format=csv
!python -m pytest -q tests          # all tests should pass
```

## Cell 3: smoke test, then the real run (Week 2)
```python
!python pairs/w2_image_editing/run_w2.py --model flux1-kontext-dev --limit 1 --offload
# if that works:
!python pairs/w2_image_editing/run_w2.py --model flux1-kontext-dev --offload
!python pairs/w2_image_editing/run_w2.py --model qwen-image-edit-2509 --offload
!python pairs/w2_image_editing/score_w2.py
!python -m a1eval summarize results/w2.jsonl --out report_assets/w2
```

## Rules for every session
- Keep the browser tab open and the screen awake (Colab Pro does not run in the background); work in sessions of about an hour.
- Per item, work already done is skipped (`a1eval.done_items`), so after a disconnect re-run Cell 1 and Cell 2, then the run line.
- After each session add a row to `cost_log.csv`: rate_per_hr = (compute units per hour shown in Colab) x A$0.1513, hours = session length. Then Runtime > Disconnect and delete runtime.
- Copy `results/*.jsonl` back to your PC (Drive syncs to `comp8410_a1/results`) and commit them: they are evidence.
