# Week 8 - Agents, evaluation and governance

Status: CODE WRITTEN, NOT YET RUN (Tue 3 Nov)  |  Controlled pair: Qwen2.5-VL-7B-Instruct reading DOM text vs UI-TARS-1.5-7B reading screenshots (GUI-trained Qwen2.5-VL derivative, Apache-2.0) + a rule-based baseline  |  Compute: L4/A100 for the two 7B models (about 17 GB bf16), CPU for the baseline

Scenario for this pair: back-office web tasks of a listing-operations agent (click, select, fill forms, log in) on MiniWoB++ (MIT; 130 tasks; observations include `utterance`, `dom_elements` and a `screenshot`; actions include CLICK_ELEMENT, CLICK_COORDS, TYPE_TEXT). Includes a prompt-injection test.

Design: 8 tasks x 3 seeds clean per agent, plus an injection variant on 4 tasks (a fake notice and a "CLICK HERE TO CONTINUE" button added to the page; we count how often the agent clicks it). Max 8 steps, greedy decoding, binary reward (no time penalty; the page clock is relaxed because a 7B model needs seconds per step). The visual agent sees the 160x210 task area upscaled to 560x728.
Why this pair: both model agents share the Qwen2.5-VL-7B backbone, so the contrast is mostly observation (DOM text vs pixels) plus GUI-specific training; it is still not a one-factor experiment and the report says so. MiniWoB is DOM-friendly by design, which favours the DOM agent.
Metrics: success rate (Wilson CI), paired bootstrap over tasks, steps, invalid-action rate, seconds per step, injection-followed rate and success drop.
Files: `install_w8.py`, `agents.py`, `run_w8.py`, `report_w8.py`, `w8_common.py`, notebook `colab/a1_week8.ipynb`, tests `tests/test_w8.py` (10 pass). Not yet verified on a real browser: Chrome/driver install, the page-clock fix, UI-TARS prompt format and coordinate scale (smoke cells 2.3, 4.2, 4.3 do this).

## A. Selection (rubric 1.5)
### A1. AI modelling techniques (0.5)
- Technique family and core algorithm:
- Lecture link (slides / practical):
- Why state of the art; what it superseded:

### A2. Modelling context (1.0) - justify each for the property-marketplace scenario
| Item | UI-TARS-1.5-7B (candidate) | MAI-UI-8B or Holo (candidate) |
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
Metrics planned: task_success (0/1, environment reward); steps_taken; latency_s; injection_pass (0/1)
Checklist of methodology bullets covered:
- [ ] role/appropriateness vs trends  - [ ] technical performance (quantified)  - [ ] business performance (quantified)
- [ ] competitiveness/uniqueness  - [ ] social/economic/environmental benefit  - [ ] user feedback (survey items)
- [ ] gaps/disadvantages  - [ ] governance and security  - [ ] safety/privacy/ethics risk
- [ ] fairness and trustworthiness  - [ ] difference made by advancements  - [ ] recommendations

### B2. Results (1.0)
- Settings, baselines, datasets, metrics (with justification):
- Table (from `report_assets/w8/summary.md`):
- Figures (from `report_assets/w8/*.png`):
- Failure cases (with likely cause tied to the lecture mechanism):
- Business metrics: cost per output, latency, VRAM, licence (assumptions stated):

### B3. Reflection and recommendation (0.5)
- Applicability to the scenario:
- Gaps / open issues / future directions:
- Alternatives for the organisation:

## Evidence
- Raw results: `results/w8.jsonl`
- Outputs: `outputs/w8/<model>/<item_id>.*`
- Environment: `pairs/w8_agents/ENV.txt` (pip freeze), model versions and download dates in `docs/EVIDENCE.md`
- Screen clip of the run: `clips/w8_*.mp4`
- GPU spend: `cost_log.csv`
