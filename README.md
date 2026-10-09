# REC-10: Week 1 reproducible-workflow lecture and course repository template

**Approved:** 2026-10-09 · **Priority:** P1, critical path · **Status:** draft for faculty review (release gate not yet passed)
**Findings addressed:** G5, H9 (gap analysis); FIND-07 (loader, split, and eval defects); the removed `torch.set_deterministic` API in the old helpers; LO3 and LO10 support.

## Contents

| Path | What it is |
|---|---|
| `dl-course-template/` | Starter repository for Labs 1–5 and the capstone (publish as a GitHub template repository) |
| `lecture/week01_reproducible_workflow.qmd` | Lecture source (Quarto), 19 slides with speaker notes and a 30-minute hands-on activity |
| `lecture/week01_reproducible_workflow.html` / `.pptx` | Rendered versions (reveal.js and PowerPoint) |

## What the template replaces or consolidates

| Original (GitHub repos `CSCI-xx52-Deep-Learning--main` and `CSCI4852_6852_F23_DL-main`) | Problem | In the template |
|---|---|---|
| 5 diverging copies of `helper_train.py`, `helper_dataset.py`, and related helpers across lab7–lab15 | Duplication; behavior differs by lab | One tested package, `src/dlcourse/` |
| `lab7/mlp-softmax-pyscripts/helper.py` `set_deterministic()` | Calls `torch.set_deterministic`, which no longer exists in PyTorch | `set_seed(..., deterministic=True)` using `torch.use_deterministic_algorithms` |
| `settings.yaml` with keys containing spaces; values hard-coded in notebooks | Settings scattered and unvalidated | `configs/default.yaml` plus typed, validated CLI overrides |
| CNN-MNIST notebook: no shuffling, no validation split, eval without `no_grad`, kwargs passed to the dataset (FIND-07) | Correctness | Seeded shuffled train loader; disjoint seeded splits; eval under `set_grad_enabled(False)`; best checkpoint chosen on validation, test used once |
| No record of versions, device, or commit | Not reproducible | `env.json` per run (versions, GPU, git commit with a dirty flag) |
| Single-run reporting | No variance estimate | `summarize_runs.py` gives mean ± std across seeds |
| `profileTorch.py` (F23 repo) | Ad hoc | Run time and peak GPU memory recorded in `results.json` |

Source copies in `00_source_materials/` are untouched. The template is a rewrite, not a copy; credit to the STAT 453 origin of the helper design is given in the template README.

## Verification performed (2026-10-09; REC-09 environment, Windows 11, RTX 2080 Super)

| Check | Result |
|---|---|
| Fresh copy → `pip install -e .` → `python -m pytest` | **8 passed** (about 6 s, CPU): config validation, disjoint splits, split depends on seed, training reproducible, training learns, checkpoint round-trip, parameter count, CLI end-to-end plus summary plus plot |
| MNIST, 3 seeds, 3 epochs, GPU | Test accuracy **96.73% ± 0.23%**; 109,386 parameters; about 36 s per run; 18.7 MB peak GPU memory |
| GPU determinism: same seed twice with `deterministic=true` | **Identical** val loss and test accuracy |
| Synthetic default, 3 seeds, CPU | 86.75% ± 2.17% (overlap tuned so seed variation is visible; the first draft hit 100%, which made the seed exercise meaningless) |
| `notebooks/00_quickstart.ipynb` executed headless | Runs end to end |
| Lecture renders | HTML and PPTX produced without errors |

Fixed during testing: fractional-epoch axis ticks; synthetic data too easy; lecture example numbers aligned with the measured run.

## Not yet verified / faculty actions

- **GitHub Actions workflow:** the repo is published, but the workflow was pushed as `workflows/tests.yml` instead of `.github/workflows/tests.yml`, so it has never run (the API shows 0 runs). Fix: move the file, then confirm the green check. Also enable *Settings → Template repository*. Optional: remove `.vscode/settings.json` and `uv.lock`; the lock file only lists the package itself and pins nothing.
- `Makefile` targets are for OSC, Linux, and macOS. On Windows, students run the commands directly (the README shows both).
- `environment.yml` (copied from REC-09) is untested under conda. It is covered by the faculty's OSC run (REC-09 Q1).
- **Lecture review:** visual check of the slides in your template/theme. In the PPTX export, the two table slides are split onto continuation slides (4 and 14); merge them in PowerPoint if preferred.
- **Done:** the Git slide now uses https://github.com/alinutzal/dl-course-template (Use this template → clone your own repo). Slides approved by faculty ("Fine").

## Syllabus alignment (2026-10-09)

Faculty decision: the materials follow the syllabus, never the reverse. Checked against syllabus Week 1 ("PyTorch setup; tensors, devices, data splits, reproducibility"; readings Ketkar Ch 1–2):

| Change | Why |
|---|---|
| Added two **tensor** slides (shape, dtype, device, reshape, reductions, broadcasting, matmul, `.to(device)`, NumPy round trip); every output shown was produced by running the code in the course environment | "Tensors" is listed in syllabus Week 1 and was missing |
| Agenda now follows the syllabus order: setup → tensors → workflow and devices → splits | Syllabus order |
| Seed guidance now says "results across multiple runs when randomness matters; CSCI 6952 capstone: at least 3 seeds with variability reported" (lecture slides 6 and 20; template README checklist) | The earlier "≥ 3 seeds when comparing" for everyone was stricter than the syllabus |

**PowerPoint export fix.** Quarto's PPTX output gave code and body text a large default size. Code wrapped mid-statement and two slides overflowed. These were present in the earlier version too; you may have reviewed the HTML. `postprocess_pptx.py` now applies body 18 pt, inline code 16 pt, code blocks 12 pt, and leaves titles alone. All 21 slides were rendered with PowerPoint and checked.

Build: `quarto render week01_reproducible_workflow.qmd --to revealjs`, `quarto render week01_reproducible_workflow.qmd --to pptx`, then `python postprocess_pptx.py week01_reproducible_workflow.pptx`.

**Faculty action:** the template README change (seed checklist line) needs to be pushed to github.com/alinutzal/dl-course-template.
