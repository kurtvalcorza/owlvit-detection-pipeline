# Release verification

`tutorials/owlvit_detection_colab.ipynb` (`TASK-INFERENCE`, **standalone** carrier) is a **release candidate** until the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, the tiny-model test, JSON validation, code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER Notebook Specification 2.1 (REL8). This file is the durable release-gate record for the notebook.

The upstream snapshot is not yet pinned, so the notebook cannot run yet: its model cell raises before any download. Pinning (`python tools/pin_snapshot.py`) and regenerating the notebook come before any execution recorded here.

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
2. open that exact notebook revision in a new runtime (Colab or Kaggle; CPU is sufficient) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells, with every form field at its default (`USE_BYOD = False`, `threshold = 0.1`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to `metadata.dimer.generated_from.revision`, and that the installed versions equal the inline `PINS`;
5. verify that every default-path stage completes: the pinned install; the carried module; staging of all eight manifest entries and `verify_snapshot`; `validate_inputs` with the over-long-phrase rejection finding; `detect`; the `sample-sanity` evaluation report; and the five outputs in `outputs/`;
6. record the returned labels, scores and per-object `box_iou` values, the count of boxes at threshold 0.05 as well as 0.1, and the wall time. No value is asserted in advance: a missed shape is a finding to record, not a failure by itself;
7. exercise the BYOD branch once with a photograph through `BYOD_IMAGE_PATH` and once with an incompatible input (for example 17 phrases), and record both outcomes (REL12);
8. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), model identifier and revision, whether the model cache was clean, and any warning judged harmless with the reason;
9. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/owlvit_detection_colab.ipynb` (verify with `git rev-parse <commit>:tutorials/owlvit_detection_colab.ipynb`). Each record uses the fields Date, Subject, Runtime, Procedure, Observed result and Caveats.

| Date (UTC) | Subject (commit / notebook blob) | Runtime | Procedure | Observed result | Caveats |
|---|---|---|---|---|---|
| — | — | — | — | No execution recorded. The snapshot is not yet pinned. | — |
