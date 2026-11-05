# Week 7 - Video editing

Status: **BUILT, awaiting clips + run**  |  Models: Wan2.1-VACE-1.3B vs Lucy-Edit-Dev  |  Compute: A100-40GB (or L4)

Scenario for this pair: walkthrough clips - swap furnishings, relight, change scenery.

## Why these two models

Both edit an existing clip from a text instruction, and both are diffusers-native, so there is
no repo to clone and no flash-attn build (the Week 6 lesson). They differ in **conditioning**:

| | Wan2.1-VACE-1.3B | Lucy-Edit-Dev |
|---|---|---|
| HF id | `Wan-AI/Wan2.1-VACE-1.3B-diffusers` | `decart-ai/Lucy-Edit-Dev` |
| Pipeline | `WanVACEPipeline` | `LucyEditPipeline` |
| Conditioning | video **+ spatial edit mask** (+ optional reference images) | video **+ text only** |
| Base | Wan2.1 | Wan2.2 5B |

Matched settings for both: **480x832, 50 steps, CFG 5.0, same clip, same instruction, same seed.**
Both use the same Wan VAE family, so the frame count must be 4n+1 (`vae_frames()` enforces this).

**The declared asymmetry.** VACE receives the edit box as a mask; Lucy-Edit has no mask input.
That is the modality contrast under test, not a mistake - but it means VACE is given spatial
information Lucy must infer from the instruction. The mask box is used at **scoring** time for
both models, so `bg_psnr` / `bg_ssim` (fidelity outside the edit region) are measured
identically. Say this plainly in the report; do not claim a pure architecture comparison.

## Files

| File | What it does |
|---|---|
| `w7_common.py` | matched settings, item manifest access, `vae_frames()`, `box_pixels()`, `EDIT_BANK` |
| `prepare_inputs.py` | any folder of clips -> 832x480 mp4s at 4n+1 frames + `items.csv` |
| `gen_w7.py` | both pipelines, resumable via sidecar json, `--smoke` for 1 item / 6 steps |
| `score_w7.py` | CLIP/DINOv2 + pixel metrics; `--rescore` to recompute |
| `report_w7.py` | paired-bootstrap tables, plot, `summary.md` into `report_assets/w7/` |
| `install_w7.py` | pip only; verifies both pipeline classes are importable |
| `../../tests/test_w7.py` | 13 CPU tests of the pure functions |
| `../../colab/a1_week7.ipynb` | run order; 2.4 shows the inputs before any GPU spend |

## Metrics

| Metric | Meaning | Better |
|---|---|---|
| `clip_t_gain` | CLIP text-frame cosine for the instruction, output minus input | higher |
| `clip_t_mismatch` | output against a *different* item's instruction (validity control) | --- |
| `bg_psnr`, `bg_ssim` | fidelity **outside** the edit box (capped at 100 dB when identical) | higher |
| `box_mad`, `bg_mad` | change magnitude inside / outside the box | --- |
| `edit_ratio` | `box_mad / bg_mad`, capped at 1000 - is the change localised | higher |
| `flicker`, `flicker_excess` | abs second temporal difference; excess over the input clip | lower |
| `motion_mad` | abs frame difference - catches an output that froze (the Week 6 trap) | --- |
| `dino_consec`, `dino_src_mean` | temporal coherence; content kept vs the input clip | higher |
| `latency_s`, `peak_vram_gb`, `file_mb` | run cost | lower |

`box_mad` near zero means the model did nothing - check it before reading any quality metric.

## Run order

```
1.1  setup                         2.1  install          2.2  tests
2.3  prepare_inputs (set SRC)      2.4  LOOK at inputs
3.1  smoke VACE   3.2 smoke Lucy   3.3  look + budget gate
4.1  full VACE    4.2 full Lucy    4.3 score   4.4 report   4.5 strips
5.1  cost log                      5.2  disconnect
```

## A. Selection (rubric 1.5)

### A1. AI modelling techniques (0.5)

Both are diffusion-transformer video editors built on the Wan latent-video architecture: the
clip is encoded by a causal 3D VAE, edited in latent space under text conditioning, and
decoded. VACE adds a control branch that accepts a mask (and reference images) so the edit is
spatially targeted; Lucy-Edit conditions on the source video and instruction alone.

- Technique family and core algorithm:
- Lecture link (slides / practical):
- Why state of the art; what it superseded:

> **To complete:** verify the architecture claims above against the actual papers/model cards
> before citing them, then fill the three bullets.

### A2. Modelling context (1.0) - justify each for the property-marketplace scenario

| Item | Wan2.1-VACE-1.3B | Lucy-Edit-Dev |
|---|---|---|
| Objectives (technical + business, why) | | |
| Tasks (why) | | |
| Functions | | |
| Framework (architecture, workflow, process) | | |
| Algorithms (tie to lecture content) | | |
| Governance (safety, security, privacy, ethics) | | |
| Advantages vs competitors | | |

> **To complete.** The governance row is the strongest cell here: editing a walkthrough clip of
> a real property changes what a buyer sees of a dwelling they may purchase, so disclosure of
> edited media and a prohibition on altering structural features belong in this row.

## B. Evaluation (rubric 2.0)

### B1. Methodology (0.5)

Items listed in `items.csv` (generated from the clip folder), fixed seed, identical prompts,
resolution, step count, guidance and hardware for both models. Metrics as tabled above, with a
mismatch control on `clip_t` and a paired bootstrap over items for every difference.

Checklist of methodology bullets covered:
- [ ] role/appropriateness vs trends  - [ ] technical performance (quantified)  - [ ] business performance (quantified)
- [ ] competitiveness/uniqueness  - [ ] social/economic/environmental benefit  - [ ] user feedback (survey items)

### B2. Results (1.0)

> Generated by `report_w7.py` into `report_assets/w7/` once the run completes.

### B3. Reflection and recommendation (0.5)

> **To complete after the run.** Check first whether `box_mad` is non-trivial for both models;
> if either did nothing, that is the finding. Then address: the mask asymmetry as a confound,
> whether localisation (`edit_ratio`) or instruction adherence (`clip_t_gain`) matters more for
> the marketplace's fidelity obligation, added flicker as a deployment blocker, and the small
> item count.
