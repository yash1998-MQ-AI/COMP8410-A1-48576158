# Week 6 - Video generation

Status: CODE WRITTEN, NOT YET RUN (Mon 2 Nov)  |  Models: Wan2.2 TI2V-5B vs HunyuanVideo 1.5 (480p I2V, step-distilled)  |  Compute: Colab A100-40GB, install check on T4

Scenario for this pair: promo/walkthrough clips for listings generated from ONE listing photo plus a camera-motion prompt (image-to-video).

Design (budget-driven, say so in the report): 6 listing photos (W2 copies, cropped 16:9), 1 seed, 73 frames at 24 fps (3.04 s), same photo / prompt / seed for both models.
Each model at its recommended setting: Wan 704x1280, 50 steps, CFG 5, T5 on CPU + model offload (needed to fit 40 GB); Hunyuan 480x848, 8 step-distilled steps if available, model CPU offload. This is a system-level comparison, not a matched-pixel one.
Metrics (`score_w6.py`): clip_t, clip_i/dino vs the source photo (mean, min), dino_consec, first_frame_ssim, motion_mad, jitter_ratio, luma_drift, mismatch controls, latency, VRAM, file size.
Files: `install_w6.py`, `prepare_inputs.py`, `gen_w6.py`, `score_w6.py`, `report_w6.py`, `items.csv`, notebook `colab/a1_week6.ipynb`, tests `tests/test_w6.py` (10 pass).

## A. Selection (rubric 1.5)
### A1. AI modelling techniques (0.5)
- Technique family and core algorithm:
- Lecture link (slides / practical):
- Why state of the art; what it superseded:

### A2. Modelling context (1.0) - justify each for the property-marketplace scenario
| Item | Wan 2.2 | HunyuanVideo 1.5 |
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
Metrics planned: clip_t; frame_flicker; latency_s; peak_vram_gb
Checklist of methodology bullets covered:
- [ ] role/appropriateness vs trends  - [ ] technical performance (quantified)  - [ ] business performance (quantified)
- [ ] competitiveness/uniqueness  - [ ] social/economic/environmental benefit  - [ ] user feedback (survey items)
- [ ] gaps/disadvantages  - [ ] governance and security  - [ ] safety/privacy/ethics risk
- [ ] fairness and trustworthiness  - [ ] difference made by advancements  - [ ] recommendations

### B2. Results (1.0)
- Settings, baselines, datasets, metrics (with justification):
- Table (from `report_assets/w6/summary.md`):
- Figures (from `report_assets/w6/*.png`):
- Failure cases (with likely cause tied to the lecture mechanism):
- Business metrics: cost per output, latency, VRAM, licence (assumptions stated):

### B3. Reflection and recommendation (0.5)
- Applicability to the scenario:
- Gaps / open issues / future directions:
- Alternatives for the organisation:

## Evidence
- Raw results: `results/w6.jsonl`
- Outputs: `outputs/w6/<model>/<item_id>.*`
- Environment: `pairs/w6_video_generation/ENV.txt` (pip freeze), model versions and download dates in `docs/EVIDENCE.md`
- Screen clip of the run: `clips/w6_*.mp4`
- GPU spend: `cost_log.csv`
