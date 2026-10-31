# Week 5: single image to 3D (TRELLIS.2-4B vs Hunyuan3D 2.1)

Scenario: a national online property marketplace wants a 3D asset of a piece of furniture or decor from the single photo a seller uploads.

## Data
Set C: 13 single-object photos taken by me (`set_c_manifest.csv`: furniture, decor, household, two transparent/reflective cases on purpose). Raw photos stay out of git (they carry GPS EXIF); `prepare_inputs.py` writes cleaned copies, `run_w5.py prep` cuts the object out (rembg) so **both models receive the identical RGBA image**.

## Method
- Generation: 13 items x 3 seeds x 2 models, one unrecorded warm-up call per model. Each model runs in its own environment (`install_envs.sh`) with the authors' own example settings (`gen_trellis2.py`, `gen_hunyuan3d.py`).
- Scoring (`score_w5.py`, `render_splat.py`): both meshes are rendered by the same CPU renderer from 8 fixed azimuths, then compared with the photo: DINOv2 and CLIP similarity (best view, mean of views) and silhouette IoU (best view). A mismatch control (score against a different item's photo) checks the metrics can tell objects apart.
- Cost per mesh (latency, peak GPU memory), geometry facts (faces, watertight, components, textured) and failure rate are logged by `gen_common.py`.
- Report: `report_w5.py` (paired bootstrap over items, seeds averaged per item).

## What this does not measure
No ground-truth scans, so no geometric accuracy (Chamfer). Scores measure agreement with the photo, mostly from the visible side; the back is never checked against anything. The renderer is a point-splat approximation (albedo with mild headlight shading), shared by both models; absolute numbers are not comparable with papers. Two A/B survey items complement the metrics (see `survey/`).

## Governance notes to carry into the report
- TRELLIS.2: model and code MIT, but the pipeline loads other models and libraries with their own terms (check what Cell B3 prints; nvdiffrast / nvdiffrec are NVIDIA-licensed).
- Hunyuan3D 2.1: Tencent Hunyuan community licence, not MIT; read `LICENSE` (printed by the install step, copied to `results/`) for territory and use limits before saying anything about marketplace use.

## Run
See `colab/a1_week5.ipynb` (Part A on a T4 builds and packs the environments; Part B on an A100 runs everything). `python -m pytest tests/test_w5.py` runs the CPU-only tests on fake meshes.
