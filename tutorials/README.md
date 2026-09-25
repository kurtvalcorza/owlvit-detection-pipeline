# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/owlvit-detection-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/owlvit-detection-pipeline/blob/main/tutorials/owlvit_detection_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fowlvit--base--patch32-ffcc4d?style=flat)](https://huggingface.co/google/owlvit-base-patch32)
[![Upstream](https://img.shields.io/badge/Upstream-google--research%2Fscenic%20(owl__vit)-181717?style=flat&logo=github&logoColor=white)](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)
[![arXiv](https://img.shields.io/badge/arXiv-2205.06230-b31b1b.svg)](https://arxiv.org/abs/2205.06230)

Notebook specification: **DIMER Notebook Specification 2.1**. The notebook is **standalone** (§4) and declares its profile and pedagogical mode (§3.4). `tools/build_notebook.py` generates it from `tools/notebook_template.py`, and it carries the package's modules, the model identity, the snapshot manifest and the runtime pins, so the exported `.ipynb` works without this repository. Do not edit the notebook by hand: edit the package or the template and regenerate (`python tools/build_notebook.py`; CI and the validator enforce `--check`).

| Notebook | Profile | Mode | Carrier | Capability | Default runtime | Sample | BYOD | Run-all | Release status |
|---|---|---|---|---|---|---|---|---|---|
| `owlvit_detection_colab.ipynb` | `TASK-INFERENCE` | `GUIDED` | standalone (generated) | zero-shot, text-prompted detection with `google/owlvit-base-patch32`: score-ordered boxes labelled with the matched phrase under a caller-owned threshold, per-object `box_iou` against drawn references as sanity evidence, JSON/CSV/PNG exports | CPU (CUDA used automatically when present) | automatic (640×480 scene with three shapes drawn in code) | one image plus your own phrases, off by default; location field `BYOD_IMAGE_PATH`; uploads stay in the runtime | not yet run on the pinned snapshot (`cbc355f`); see `../docs/release-verification.md` | **Candidate** |

## Conformance notes

- **Standalone carrier (§4):** the default path performs no clone, repository install or repository import. Section 2 carries `pipeline.py` verbatim (tagged `metadata.dimer.embedded_module`; the only rewrite makes the default weights directory working-directory-relative). Section 3 carries the model identity and the manifest inline and asserts that they agree with the module before anything is fetched. Section 1 carries the exact `pyproject.toml` runtime pins.
- **Parity (PAR1–PAR3):** `tests/test_notebook_parity.py` and `tools/validate_release_assets.py` fail when the carried module, the inline manifest or the inline pins differ from the repository, or when the notebook differs from the generator's output.
- **Model acquisition (MOD1–MOD8):** the snapshot is pinned to an immutable revision (the three loaders would raise before any download if `MODEL_REVISION` were reset to `"unpinned"`). `stage_missing_files(WEIGHTS_DIR, allow_download=True)` fetches only the absent manifest entries at the immutable revision, `verify_snapshot` re-hashes every entry, and `from_pretrained(weights_dir=WEIGHTS_DIR)` loads with `local_files_only=True` and `trust_remote_code=False`.
- **Stages:** `validate_inputs` writes the input manifest, including one rejection finding from an over-long phrase; `detect` returns score-ordered boxes; `evaluation_report` writes a `sample-sanity` report with per-object `box_iou` on the drawn scene, or `not-measurable` on BYOD. No adaptation happens, as NOTEBOOK_SPEC §13 recommends for a zero-shot task.
- **Scores (UNC1–UNC4):** every `score` is an uncalibrated sigmoid of the best image–text logit; `threshold` is a `# @param` field passed explicitly on every call; there is no non-maximum suppression.
- **Non-interactive execution (EXE1–EXE4):** the threshold, the BYOD switch, the BYOD prompts and the BYOD image location are form fields; outputs go to `outputs/` under the working directory.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the clean-runtime execution requirement; a release review must confirm that a recorded clean run in `docs/release-verification.md` matches the notebook revision under review before the status is promoted to `Release-grade`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
