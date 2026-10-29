# Week 6 - Video generation

Status: NOT STARTED  |  Models: Wan 2.2 vs HunyuanVideo 1.5  |  Compute: A100 80GB (or 4090 with FP8)

Scenario for this pair: Promo clips for listings from a still or a prompt.

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
