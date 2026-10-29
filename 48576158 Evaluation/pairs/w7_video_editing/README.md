# Week 7 - Video editing

Status: NOT STARTED  |  Models: Wan2.1-VACE-14B vs Lucy-Edit-1.1-Dev  |  Compute: A100 80GB

Scenario for this pair: Walkthrough clips: swap furnishings, relight, change scenery.

## A. Selection (rubric 1.5)
### A1. AI modelling techniques (0.5)
- Technique family and core algorithm:
- Lecture link (slides / practical):
- Why state of the art; what it superseded:

### A2. Modelling context (1.0) - justify each for the property-marketplace scenario
| Item | Wan2.1-VACE-14B | Lucy-Edit-1.1-Dev |
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
Metrics planned: clip_t (edit alignment); frame_flicker; masked_psnr (unedited region); latency_s; peak_vram_gb
Checklist of methodology bullets covered:
- [ ] role/appropriateness vs trends  - [ ] technical performance (quantified)  - [ ] business performance (quantified)
- [ ] competitiveness/uniqueness  - [ ] social/economic/environmental benefit  - [ ] user feedback (survey items)
- [ ] gaps/disadvantages  - [ ] governance and security  - [ ] safety/privacy/ethics risk
- [ ] fairness and trustworthiness  - [ ] difference made by advancements  - [ ] recommendations

### B2. Results (1.0)
- Settings, baselines, datasets, metrics (with justification):
- Table (from `report_assets/w7/summary.md`):
- Figures (from `report_assets/w7/*.png`):
- Failure cases (with likely cause tied to the lecture mechanism):
- Business metrics: cost per output, latency, VRAM, licence (assumptions stated):

### B3. Reflection and recommendation (0.5)
- Applicability to the scenario:
- Gaps / open issues / future directions:
- Alternatives for the organisation:

## Evidence
- Raw results: `results/w7.jsonl`
- Outputs: `outputs/w7/<model>/<item_id>.*`
- Environment: `pairs/w7_video_editing/ENV.txt` (pip freeze), model versions and download dates in `docs/EVIDENCE.md`
- Screen clip of the run: `clips/w7_*.mp4`
- GPU spend: `cost_log.csv`
