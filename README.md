# OWL-ViT base/32 open-vocabulary detection pipeline

DIMER pipeline for **OWL-ViT with a ViT-B/32 image encoder** (`google/owlvit-base-patch32`), a CLIP model turned into a detector that finds objects named by free-text phrases. The pipeline loads the checkpoint only from a digest-verified local snapshot and returns pixel-space boxes, each labelled with the phrase it matched and an uncalibrated sigmoid score under a caller-owned threshold. It performs no training.

> **The upstream snapshot is pinned** to Hub commit `cbc355fb364588351c5d51c7f74465e8e7ec6f72` (pinned 2026-09-25). The manifest records every file's byte size and SHA-256, and the LFS digest matched the Hub's record. Default-path execution recorded on 2026-09-25 (Kaggle T4); REL12 BYOD exercise pending before promotion (see [Release status](#release-status)).

## Upstream alignment

- Model: `google/owlvit-base-patch32`
- Revision: `cbc355fb364588351c5d51c7f74465e8e7ec6f72`
- Upstream weight license: Apache-2.0
- Upstream task: zero-shot, text-conditioned object detection
- Repository adaptation: **none**; inference only

## Quick start

```python
from PIL import Image
from owlvit_detection_pipeline import OwlViTDetectionPipeline

pipe = OwlViTDetectionPipeline.from_pretrained(allow_download=True)   # stages + verifies weights/owlvit-base-patch32
result = pipe.detect(Image.open("desk.jpg"), ["a photo of a mug", "a photo of a keyboard"], threshold=0.1)
for det in result["detections"]:                                      # sorted by score; boxes are xyxy pixels
    print(det["label"], det["box"], round(det["score"], 3))
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`, and run `pytest` for the offline test suite (no weights needed; `tests/test_tiny_model.py` runs a tiny random-weight OWL-ViT with a toy vocabulary through the real processor and post-processing).

## Pinning the snapshot

The snapshot is pinned (see [Upstream alignment](#upstream-alignment)). To move to a newer upstream commit, from the repository root with network access to huggingface.co:

1. Run `python tools/pin_snapshot.py` (or `--revision <commit>`). It resolves `main` to a commit, downloads the eight manifest files at that commit into `weights/owlvit-base-patch32/`, checks the LFS file against the Hub's SHA-256, and writes the commit and digests into the manifest and `MODEL_REVISION`.
2. Commit, then run `python tools/build_notebook.py` and commit the regenerated notebook.
3. Update the commit and digests cited in `README.md`, `MODEL_CARD.md`, `STATUS.md`, `docs/WEIGHTS.md`, `tutorials/README.md` and `docs/release-verification.md`.
4. Run `python tools/validate_release_assets.py` and `pytest`. A new pin invalidates any recorded execution, so the status returns to Candidate until the new commit is run.

## Weights layout

```
weights/owlvit-base-patch32/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256 (8 files)
  preprocessor_config.json   # resize to 768x768, no crop, CLIP normalisation
  config.json, tokenizer files, README.md   # staged with the weights
  model.safetensors          # git-ignored, 612,983,940 bytes
```

## Input ceilings and threshold

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_PROMPTS = 16`, `MAX_PROMPT_CHARS = 48`, `MAX_TEXT_TOKENS = 16`, `MAX_DETECTIONS = 576` (one box per 32×32 patch of the 768×768 input), `DETECTION_THRESHOLD = 0.1` (the pinned README example's value). There is no non-maximum suppression. See `MODEL_CARD.md` for who owns the threshold and what the score means.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/owlvit-detection-pipeline/blob/main/tutorials/owlvit_detection_colab.ipynb)

`tutorials/owlvit_detection_colab.ipynb` is declared `TASK-INFERENCE` / `GUIDED` under DIMER Notebook Specification 2.2 and is **standalone** (§4): `tools/build_notebook.py` generates it, and it carries the package module, the model identity, the manifest and the runtime pins, so it runs without this repository. Its default `Run all` path draws a 640×480 scene with three shapes, validates the image and prompts into an input manifest, detects, writes a `sample-sanity` evaluation report with per-object `box_iou`, and exports JSON, CSV and an annotated PNG. Section 1 builds an isolated, hash-locked Python 3.12.12 environment with uv (`tutorials/requirements-colab.lock.txt`) and runs every later cell there, so nothing is installed into the kernel and no restart is needed; supported runtimes are Linux x86_64 (Colab, Kaggle, Linux Jupyter). The BYOD branch is off by default and has a location field (`BYOD_IMAGE_PATH`). See `tutorials/README.md` and `docs/release-verification.md`.

## Release status

**Candidate.** The snapshot is pinned (`cbc355f`). The notebook now builds an isolated hash-locked uv environment (no kernel install, no restart) and carries the 2026-10-02 review fixes; the review-fix blob `114ea53bf6ae` (commit `4932410`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-08 (Colab CLI 0.7.4 sequential execution, 9/9 code cells, 101.4 s wall; 1 detection, `a red circle` 0.4487, `box_iou` 0.9671; the rectangle and triangle were localised (IoU 0.9235 / 0.8641) but scored 0.0373 / 0.0985, under 0.1). REL12 BYOD exercise pending before promotion. The previous blob `fff9ff981ddc` (commit `14be73f`) ran on Kaggle T4 on 2026-09-25 with BYOD off, but only after a manual kernel restart after its install cell; on the drawn sample scene at threshold 0.1 it found 1 of 3 drawn shapes (`a red circle`, score 0.4487, `box_iou` 0.9671) and missed the black rectangle and the blue triangle; one scene, one runtime. REL12 BYOD exercise pending before promotion: release step 7 has not been run. Static checks, unit tests and the tiny-model test do not constitute notebook execution evidence; `docs/release-verification.md` defines the release gate.

## Documentation

- `MODEL_CARD.md`: MODEL_CARD_SPEC 1.2 card, provenance, input/output contract.
- `docs/WEIGHTS.md`: weight provenance, pinning and hosting notes.
- `STATUS.md`: release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
