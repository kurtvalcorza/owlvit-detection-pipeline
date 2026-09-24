# OWL-ViT base/32 open-vocabulary detection pipeline

DIMER pipeline for **OWL-ViT with a ViT-B/32 image encoder** (`google/owlvit-base-patch32`), a CLIP model turned into a detector that finds objects named by free-text phrases. The pipeline loads the checkpoint only from a digest-verified local snapshot and returns pixel-space boxes, each labelled with the phrase it matched and an uncalibrated sigmoid score under a caller-owned threshold. It performs no training.

> **The upstream snapshot is not yet pinned.** `MODEL_REVISION` is `"unpinned"` and the manifest records byte sizes but no SHA-256 digests. Every weight operation refuses to run until `python tools/pin_snapshot.py` has recorded the commit and digests (see [Pinning the snapshot](#pinning-the-snapshot)).

## Upstream alignment

- Model: `google/owlvit-base-patch32`
- Revision: not yet pinned (`unpinned`)
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

From the repository root, with network access to huggingface.co:

1. Run `python tools/pin_snapshot.py`. It resolves `main` to a commit, downloads the eight manifest files at that commit into `weights/owlvit-base-patch32/`, checks the LFS file against the Hub's SHA-256, and writes the commit and digests into the manifest and `MODEL_REVISION`.
2. Commit, then run `python tools/build_notebook.py` and commit the regenerated notebook.
3. Replace the "not yet pinned" statements in `README.md`, `MODEL_CARD.md`, `STATUS.md` and `docs/WEIGHTS.md` with the commit and digests.
4. Run `python tools/validate_release_assets.py` and `pytest`. The validator fails while any document still says the snapshot is not yet pinned.

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

`tutorials/owlvit_detection_colab.ipynb` is declared `TASK-INFERENCE` / `GUIDED` under DIMER Notebook Specification 2.1 and is **standalone** (§4): `tools/build_notebook.py` generates it, and it carries the package module, the model identity, the manifest and the runtime pins, so it runs without this repository. Its default `Run all` path draws a 640×480 scene with three shapes, validates the image and prompts into an input manifest, detects, writes a `sample-sanity` evaluation report with per-object `box_iou`, and exports JSON, CSV and an annotated PNG. The BYOD branch is off by default and has a location field (`BYOD_IMAGE_PATH`). See `tutorials/README.md` and `docs/release-verification.md`.

## Release status

**Candidate.** The snapshot is not yet pinned and no execution with the pinned weights is recorded. Static checks, unit tests and the tiny-model test do not constitute notebook execution evidence; `docs/release-verification.md` defines the release gate.

## Documentation

- `MODEL_CARD.md`: MODEL_CARD_SPEC 1.2 card, provenance, input/output contract.
- `docs/WEIGHTS.md`: weight provenance, pinning and hosting notes.
- `STATUS.md`: release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
