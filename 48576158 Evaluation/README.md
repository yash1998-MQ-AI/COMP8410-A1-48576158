# COMP8410 Assignment 1 - Evaluation code and data (Student ID 48576158)

Evaluation of seven pairs (14 open-weights models) for a national online property
marketplace: image editing, identity preservation, NeRF/3DGS, image-to-3D, video
generation, video editing, and agents. One shared harness (`a1eval/`) is used by
every pair so results are directly comparable.

## Status
| Pair | Models | Status |
|---|---|---|
| w2 image editing | Qwen-Image-Edit-2509 vs FLUX.1 Kontext [dev] | not started |
| w3 identity | PuLID vs InstantID | not started |
| w4 NeRF / 3DGS | 3DGS (splatfacto) vs Nerfacto | not started |
| w5 image to 3D | TRELLIS(.2) vs Hunyuan3D 2.1 | not started |
| w6 video generation | Wan 2.2 vs HunyuanVideo 1.5 | not started |
| w7 video editing | Wan2.1-VACE-14B vs Lucy-Edit-1.1-Dev | not started |
| w8 agents | two open GUI agents on MiniWoB++ | not started |

(Model choices are provisional until licences and VRAM are confirmed; see `docs/EVIDENCE.md`.)

## Layout
```
a1eval/        shared package: record schema, run logging, metrics, aggregation
pairs/<w>/     one folder per pair: README.md (report template), config.yaml
results/       raw measurements, one JSONL per pair (append-only)
outputs/       generated images/videos/meshes: outputs/<pair>/<model>/<item>.*
report_assets/ summary tables and figures produced from results/
survey/        blind A/B user-study scripts (items, analysis)
data/          inputs and provenance (data/README.md)
tools/         helpers (EXIF stripping)
docs/          rubric checklist, evidence log, declaration
colab/         setup cell for Colab runs
tests/         unit tests for the harness
cost_log.csv   GPU spend ledger (hard cap A$60)
```

## Quick start (CPU, to check the harness)
```
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests                            # expected: all tests pass
```

## Replicating a pair
1. `pip install -r requirements-gpu.txt` plus the model-specific packages in the pair's README.
2. Fixed inputs and seeds are in `pairs/<w>/config.yaml`; input data and licences in `data/README.md`.
3. Run the pair's generation script (added per pair) - each call is wrapped in
   `a1eval.timed(...)`, which appends latency and peak VRAM to `results/<w>.jsonl`;
   metrics are appended as `Record` rows. Interrupted runs resume via `a1eval.done_items`.
4. Aggregate: `python -m a1eval summarize results/<w>.jsonl --out report_assets/<w>`
5. Spend so far: `python -m a1eval cost cost_log.csv`

Expected runtimes and GPU requirements per pair are recorded in each pair's README once measured.

## User study
```
python survey/make_ab_items.py --outputs outputs --per-pair 2    # items + private answer key
python survey/analyse_survey.py --responses survey/responses.csv --key survey/answer_key.json
```
Half of the raters get version V1 and half V2 (left/right mirrored) to balance position bias.
`survey/answer_key.json` and raw responses are not committed.

## Reproducibility notes
- Seeds, model versions (repo id + commit/date), library versions and hardware are logged per pair.
- Generated outputs are saved, so metrics can be recomputed without re-running generation.
- Licences matter for the business analysis; each model's licence terms are recorded in `docs/EVIDENCE.md`.
