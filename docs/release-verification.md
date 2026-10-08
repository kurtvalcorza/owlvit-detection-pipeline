# Release verification

`tutorials/owlvit_detection_colab.ipynb` (`TASK-INFERENCE`, **standalone** carrier) is a **release candidate** until the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, the tiny-model test, JSON validation, code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate record for the notebook.

The upstream snapshot is pinned to `cbc355fb364588351c5d51c7f74465e8e7ec6f72` and the notebook is regenerated with that revision and manifest, so it can run. One default-path execution of a previous blob is recorded below (2026-09-25, Kaggle T4); it needed a manual restart after the in-kernel install cell. The current blob builds an isolated hash-locked uv environment instead and carries the 2026-10-02 review fixes (OVT-*), so it needs its own one-pass run; release step 7 (REL12) has not been exercised, so the status stays `Candidate`.

## Automatic coverage (static and unit, every pull request)

CI installs the pinned CPU-only torch wheel and the other runtime pins, then runs:

- `ruff check src tests tools`;
- `pytest`: snapshot verification and staging against synthetic manifests, the unpinned refusals, the import boundary, prompt and image validation, the evaluation report, the notebook parity checks, the weight-facts check, the pin tool against a fake Hub, and `tests/test_tiny_model.py` — a tiny random-weight OWL-ViT with a toy CLIP vocabulary taken through the real processor, model, post-processing and `detect`;
- `tools/validate_release_assets.py`: model card 1.2 structure and front matter, the pin state across the package, the manifest and the documents, identity consistency, the weight facts, the release-status tokens, and the notebook's structure, carried module, parity, markers, BYOD gate and location field;
- `tools/build_notebook.py --check`.

These are source, provenance and unit checks. None of them loads the pinned checkpoint, so none of them is execution evidence.

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. confirm the snapshot is pinned (`MODEL_REVISION` is a 40-hex commit and every manifest entry has a SHA-256) and that static CI is green on the exact commit under review;
2. open that exact notebook revision in a new Linux x86_64 runtime (Colab or Kaggle; CPU is sufficient) with **no repository checkout** and a clean model cache, and run every cell in order in one kernel — no restart is expected, and a run that needs one does not pass;
3. run the notebook top-to-bottom without editing implementation cells, with every form field at its default (`USE_BYOD = False`, `threshold = 0.1`);
4. verify that Section 1 builds (or reuses) the isolated hash-locked environment, reports `NOTEBOOK_SOURCE.repository_revision` equal to `metadata.dimer.generated_from.revision`, and that the environment's versions equal the inline `PINS`;
5. verify that every default-path stage completes: the isolated environment; the carried module; staging of all eight manifest entries and `verify_snapshot`; `validate_inputs` with the two rejection findings (over 48 characters; over 16 CLIP tokens); `detect`; the `sample-sanity` evaluation report; and the five outputs in `outputs/`;
6. record the returned labels, scores and per-object `box_iou` values, the `per_reference_best_box` rows (each shape's best score and IoU from the threshold-0 pass, which give the box count at 0.05 and 0.02 without a rerun), and the wall time. No value is asserted in advance: a missed shape is a finding to record, not a failure by itself;
7. exercise the BYOD branch once with a photograph through `BYOD_IMAGE_PATH` and once with an incompatible input (for example 17 phrases, or a phrase over 16 CLIP tokens), and record both outcomes (REL12);
8. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), model identifier and revision, whether the model cache was clean, and any warning judged harmless with the reason;
9. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/owlvit_detection_colab.ipynb` (verify with `git rev-parse <commit>:tutorials/owlvit_detection_colab.ipynb`). Each record uses the fields Date, Subject, Runtime, Procedure, Observed result and Caveats.

| Date (UTC) | Subject (commit / notebook blob) | Runtime | Procedure | Observed result | Caveats |
|---|---|---|---|---|---|
| 2026-09-25 (23:34:37–23:39:55) | commit `14be73f` / blob `fff9ff981ddc` (`NOTEBOOK_SOURCE.repository_revision` `bec396c`, the source revision the notebook was generated from, equal to `metadata.dimer.generated_from.revision`; `bec396c..14be73f` changes only the notebook and `tests/test_pin_snapshot.py`) | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-owlvit-detection` v1), batch run; image `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461` (image torch 2.10.0+cu128, transformers 5.0.0), Python 3.12.13, Tesla T4 15360 MiB, driver 580.159.04; after the inline pins torch 2.14.0+cu130 (CUDA 13.0), transformers 4.57.6; device `cuda:0`, float32 | serial suite: blob fetched at the 40-char SHA and Git-blob verified, HF cache clean at start, no repository checkout; top-to-bottom with every form field at its default (`USE_BYOD = False`, `threshold = 0.1`); `google/owlvit-base-patch32` at `cbc355fb364588351c5d51c7f74465e8e7ec6f72` (apache-2.0), all 8 manifest files fetched (614,579,712 B) and `verify_snapshot` passed (`verified_files` 8), model loaded from the local snapshot | **PASSED only after a manual restart** (not a one-pass Run all; RUN1/RUN10; superseded blob) — 314.8 s wall (pass 1 252.7 s stopped after the install cell with `RuntimeError: Core dependencies changed while older modules were loaded` (cuda-bindings 12.9.4 → 13.4.3, numpy 2.0.2 → 2.5.3), kernel restarted after the install cell as the notebook instructs; pass 2 62.0 s), 8/8 post-restart code cells ok; `validate_inputs` accepted the sample and recorded the over-long-phrase rejection (49 chars > `MAX_PROMPT_CHARS` 48); synthetic 640×480 scene (`rgb_sha256` `527ab8a8384c…`) with phrases `a black rectangle`, `a red circle`, `a blue triangle` at threshold 0.1: `detect` took 0.53 s and returned **1 detection** — `a red circle`, score 0.4487, box [378.1, 137.5, 560.0, 321.7]; per-reference `box_iou`: red circle 0.9671 (label matches), black rectangle 0.0 and blue triangle 0.0 (**not detected**; the only box is the circle's, label does not match); `evaluation_report` verdict `sample-sanity`, `baselines` empty; five outputs written, sha256 `owlvit_detection_result.json` `31e01a60936d…`, `owlvit_detection_evaluation_report.json` `2f1ae4a190bd…`, `owlvit_detection_input_manifest.json` `1f4fdf228b94…`, `owlvit_detection_detections.csv` `7c57c3cc74fb…`, `owlvit_detection_annotated.png` `dfb73a9c4d24…` | One drawn scene, three reference boxes, one runtime, no dispersion estimate; geometry sanity evidence, not a detection benchmark. Two of the three drawn shapes were not detected at the default threshold 0.1 — recorded as a finding (step 6), not hidden. The count of boxes at threshold 0.05 (step 6) is not produced by the default path and was not captured in this run. The pip resolver's conflict warnings for preinstalled Kaggle packages were harmless (the notebook's version check passed after the restart). **REL12 not exercised** (step 7: BYOD photograph and an incompatible 17-phrase input), so the status stays `Candidate`. No tokens or secrets recorded. |
