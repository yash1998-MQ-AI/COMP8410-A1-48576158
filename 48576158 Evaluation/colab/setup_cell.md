# Colab setup cell

Colab sessions can disconnect, so everything is saved to Google Drive and every run
resumes from the results file.

```python
# 1. Drive for outputs, results and cached weights
from google.colab import drive
drive.mount('/content/drive')
import os
BASE = '/content/drive/MyDrive/comp8410_a1'
os.makedirs(BASE, exist_ok=True)

# 2. Code (replace with your repo URL)
!git clone https://github.com/<your-username>/<your-repo>.git /content/a1 || (cd /content/a1 && git pull)
%cd /content/a1
!pip -q install -r requirements.txt -r requirements-gpu.txt

# 3. Point caches and output folders at Drive so a disconnect loses nothing
os.environ['HF_HOME'] = f'{BASE}/hf_cache'
!rm -rf results outputs && ln -s {BASE}/results results && ln -s {BASE}/outputs outputs
os.makedirs(f'{BASE}/results', exist_ok=True); os.makedirs(f'{BASE}/outputs', exist_ok=True)

# 4. Check which GPU was actually assigned, and log it in docs/EVIDENCE.md
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

Per item, skip work already done:

```python
from a1eval import done_items, timed
done = done_items('results/w2.jsonl', 'qwen-image-edit-2509')
for item in items:
    if item.id in done:
        continue
    with timed('results/w2.jsonl', 'w2', 'qwen-image-edit-2509', item.id, seed=0):
        out = pipe(...)
    out.save(f'outputs/w2/qwen-image-edit-2509/{item.id}.png')
```

After every session add a row to `cost_log.csv` (compute units used x unit price) and stop the runtime.
