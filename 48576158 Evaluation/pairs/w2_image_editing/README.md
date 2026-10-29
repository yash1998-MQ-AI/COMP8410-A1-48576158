# Week 2 - Instruction-based image editing (relighting, time-of-day, style)

Status: NOT STARTED  |  Models: Qwen-Image-Edit-2509 vs FLUX.1 Kontext [dev]  |  Compute: A100 80GB (or quantised build)

Scenario for this pair: Property listing photos: lighting, time-of-day, season and style edits. NO staging, clutter removal or fill (out of scope; used in an earlier assignment).

## A. Selection (rubric 1.5)
### A1. AI modelling techniques (0.5)
- Technique family and core algorithm:
- Lecture link (slides / practical):
- Why state of the art; what it superseded:

### A2. Modelling context (1.0) - justify each for the property-marketplace scenario
| Item | Qwen-Image-Edit-2509 | FLUX.1 Kontext [dev] |
|---|---|---|
| Objectives (technical + business, why) | | |
| Tasks (why) | | |
| Functions | | |
| Framework (architecture, workflow, process) | | |
| Algorithms (tie to lecture content) | | |
| Governance (safety, security, privacy, ethics) | | |
| Advantages vs competitors | | |

## B. Evaluation (rubric 2.0)
### B1. Methodology (0.5)
Test items (n = 12-15, list in config.yaml), fixed seeds, same prompts/resolution/hardware for both models.
Metrics planned: clip_t (edit/prompt alignment); lpips_bg or masked_psnr (unchanged-region preservation); latency_s; peak_vram_gb
Checklist of methodology bullets covered:
- [ ] role/appropriateness vs trends  - [ ] technical performance (quantified)  - [ ] business performance (quantified)
- [ ] competitiveness/uniqueness  - [ ] social/economic/environmental benefit  - [ ] user feedback (survey items)
- [ ] gaps/disadvantages  - [ ] governance and security  - [ ] safety/privacy/ethics risk
- [ ] fairness and trustworthiness  - [ ] difference made by advancements  - [ ] recommendations

### B2. Results (1.0)
- Settings, baselines, datasets, metrics (with justification):
- Table (from `report_assets/w2/summary.md`):
- Figures (from `report_assets/w2/*.png`):
- Failure cases (with likely cause tied to the lecture mechanism):
- Business metrics: cost per output, latency, VRAM, licence (assumptions stated):

### B3. Reflection and recommendation (0.5)
- Applicability to the scenario:
- Gaps / open issues / future directions:
- Alternatives for the organisation:

## Evidence
- Raw results: `results/w2.jsonl`
- Outputs: `outputs/w2/<model>/<item_id>.*`
- Environment: `pairs/w2_image_editing/ENV.txt` (pip freeze), model versions and download dates in `docs/EVIDENCE.md`
- Screen clip of the run: `clips/w2_*.mp4`
- GPU spend: `cost_log.csv`

## How to run (Colab)
Scope: lighting, time-of-day and weather instruction edits only (no staging, clutter removal, object insertion or fill).
15 items, one prompt and one fixed seed each, input resized to 1024 px long side, same prompt for both models.

1. Local, once: `python pairs/w2_image_editing/prepare_items.py` (builds `data/processed/w2_listing/` and `items.csv`).
   Copy `data/processed/w2_listing/` to Drive and link it into the repo at the same path (private photos are not in git).
2. Colab (see `colab/setup_cell.md` for the Drive/repo setup):
```
!pip -q install -U diffusers transformers accelerate sentencepiece protobuf
from huggingface_hub import login; login()   # FLUX.1 Kontext [dev] is gated: accept its licence on Hugging Face first
!python pairs/w2_image_editing/run_w2.py --model flux1-kontext-dev --limit 1 --offload       # smoke test, check GPU and speed
!python pairs/w2_image_editing/run_w2.py --model flux1-kontext-dev --offload
!python pairs/w2_image_editing/run_w2.py --model qwen-image-edit-2509 --offload
!python pairs/w2_image_editing/score_w2.py
!python -m a1eval summarize results/w2.jsonl --out report_assets/w2
```
3. Remove `--offload` on an 80 GB GPU for speed. Record the GPU, compute units used and library versions in `docs/EVIDENCE.md` and `cost_log.csv`.

Metrics: `edge_ssim` (layout kept), `clip_i` (content kept), `clip_gain` (moved toward the target description), plus `latency_s` and `peak_vram_gb`.
Status: generation code for the two models has NOT yet been run on a GPU; the pipeline, resume logic and edge metric are tested on CPU with the `identity` stub.
The generated edits show private rooms: decide per item what is published (street-facing images show the house front).
