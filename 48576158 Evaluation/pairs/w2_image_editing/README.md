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
