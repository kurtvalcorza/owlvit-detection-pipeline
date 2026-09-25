---
license: apache-2.0
model_card_spec: "1.2"
pipeline_tag: zero-shot-object-detection
base_model: google/owlvit-base-patch32
date_published: "2022-05"
date_published_source: "the pinned README's Model Date (May 2022), which matches the OWL-ViT paper (arXiv:2205.06230, submitted 2022-05-12); the date the Hugging Face conversion was first published is not established by this repository"
---

# OWL-ViT base/32 — Open-Vocabulary Object Detection (Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fowlvit--base--patch32-ffcc4d?style=flat)](https://huggingface.co/google/owlvit-base-patch32)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-google--research%2Fscenic%20(owl__vit)-181717?style=flat&logo=github&logoColor=white)](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2205.06230-b31b1b.svg)](https://arxiv.org/abs/2205.06230)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

> [!IMPORTANT]
> The upstream snapshot is pinned to Hub commit `cbc355fb364588351c5d51c7f74465e8e7ec6f72`, and the manifest records every file's SHA-256. Default-path execution recorded on 2026-09-25 (Kaggle T4); REL12 BYOD exercise pending before promotion. The measured values under Metrics come from that one run on one drawn scene with three reference boxes, one runtime. They are sanity evidence, not a benchmark.

---

## Interactive Colab Tutorials

- **Task inference tutorial**:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/owlvit-detection-pipeline/blob/main/tutorials/owlvit_detection_colab.ipynb) [`owlvit_detection_colab.ipynb`](https://github.com/kurtvalcorza/owlvit-detection-pipeline/blob/main/tutorials/owlvit_detection_colab.ipynb)
  *Text-prompted detection on a scene drawn in code with the pinned `google/owlvit-base-patch32` weights: score-ordered boxes labelled with the matched phrase under a caller-owned `threshold`, and per-object `box_iou` against the drawn references as sanity evidence only — no mAP.*

---

#### Description

`google/owlvit-base-patch32` is the Hugging Face Transformers release of OWL-ViT (Vision Transformer for Open-World Localization) with a ViT-B/32 image encoder, from "Simple Open-Vocabulary Object Detection with Vision Transformers" (Minderer et al., arXiv:2205.06230). The snapshot's `config.json` declares `OwlViTForObjectDetection` with a CLIP image tower (12 layers, hidden size 768, 32×32 patches at 768×768 input) and a CLIP text tower (12 layers, hidden size 512, at most 16 tokens), projected to 512 dimensions. This repository has not counted the parameters.

OWL-ViT turns CLIP into a detector. It removes the image tower's final token pooling and attaches a box head and a class head to every image token. The class head scores each token's embedding against text embeddings of the caller's phrases, so the vocabulary is whatever the caller writes. At inference the processor resizes the image to 768×768 without preserving the aspect ratio. The 24×24 = 576 image tokens each propose one box. Each box is labelled with the phrase whose sigmoid image–text score is highest, and it is kept when that score reaches the threshold. There is no non-maximum suppression.

No adaptation happens in this repository: there is no training, fine-tuning, in-context conditioning or preprocessing fitting. The upstream authors trained CLIP from scratch and then fine-tuned it end to end with the detection heads on public detection datasets, per the pinned README.

What this repository adds to the upstream weights:

- `verify_snapshot` and `stage_missing_files`: manifest checks and staging of the pinned files, both refusing to run if `MODEL_REVISION` is ever reset to the `"unpinned"` sentinel;
- `OwlViTDetectionPipeline.from_pretrained`: loading from the verified directory only, with `trust_remote_code=False` and `local_files_only=True`;
- `detect`: prompt normalisation (`format_prompts`), image and threshold checks, and score-sorted pixel-space output labelled with the matched phrase;
- `validate_inputs` and `evaluation_report`: the validation and single-image evaluation stages, and `box_iou`, the overlap primitive;
- `tools/pin_snapshot.py`, `tools/build_notebook.py` and `tools/validate_release_assets.py`: pinning, notebook generation and static release checks.

#### Intended Use and Limitations

The uses below are the ones the package was built to support. Everything else is out of scope (Out-of-scope use cases) or prohibited (Use cases).

###### Primary Intended Uses

The task is zero-shot, text-prompted object detection. `detect` takes one `PIL.Image.Image`, 1–16 short phrases and a threshold. It returns boxes in input pixels, each labelled with the phrase it matched best and a sigmoid score, sorted by score.

Envisioned applications are finding objects that no fixed class list names, in research and prototyping. Examples are labelling a new object category in a small image set before training a closed-set detector, triaging images for a phrase, and cropping regions for downstream classification.

The intended role is a zero-configuration baseline for open-vocabulary detection and a comparison point for the OWLv2 pipeline, its successor. A reader's own application can embed `OwlViTDetectionPipeline` as a candidate-proposal stage whose output a person or a later model checks.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision researchers, data annotators preparing labelled sets, students and instructors. Settings envisioned are research prototypes, teaching, and self-hosted applications that run the code in this repository.

A user is expected to know the following before relying on the output:

- a label only says which of the caller's phrases scored best; it does not say the object is present;
- a `score` is an uncalibrated sigmoid, and scores for different phrases are not comparable probabilities;
- results depend on the wording of each phrase, and a phrase longer than 16 CLIP tokens is truncated;
- there is no non-maximum suppression, so one object can appear as several overlapping boxes at a low threshold;
- precision, recall and average precision can only be measured on labelled images the user supplies;
- plain drawn shapes are outside the training distribution: in the recorded tutorial run only the red circle of three drawn shapes was found at the default threshold 0.1.

###### Out-of-scope use cases

1. **Capability boundary:** no masks (the public `sam2-segmentation-pipeline` repository covers segmentation), no tracking, OCR or captioning, and no training. Image-guided (one-shot) detection exists upstream but is not exposed by this package. For a fixed class list with a trained head, the public `detr-detection-pipeline` and `rtdetr-detection-pipeline` repositories apply.
2. **Input boundary:** `validate_image` rejects anything that is not a `PIL.Image.Image` (`TypeError`), and sides below `MIN_IMAGE_SIDE = 16` px or above `MAX_IMAGE_SIDE = 4096` px. `format_prompts` rejects 0 or more than `MAX_PROMPTS = 16` phrases, empty phrases, phrases longer than `MAX_PROMPT_CHARS = 48` characters and duplicates after normalisation. Thresholds outside `[0, 1]` are rejected.
3. **Input boundary:** the 768×768 resize distorts every non-square image. With 32-pixel patches, an object smaller than about one patch at that scale is poorly represented. At most `MAX_DETECTIONS = 576` boxes can be returned.
4. **Decision boundary:** not for decisions that act on detections without a person reviewing them, such as security alerts, content moderation, safety interlocks, or medical or industrial inspection, and not without precision and recall measured locally at the chosen threshold.

#### Factors

###### Groups

The pipeline is human-centric wherever a caller's phrase names people or their attributes. The CLIP backbone was trained on web image–caption data that, per the pinned README, is "more representative of people and societies most connected to the internet". Neither the upstream authors nor this repository evaluated detection per demographic group.

Phrases that describe people by appearance, clothing, age or other attributes invite associations learned from web captions. Those associations can encode stereotypes. Whether and how that shifts recall or scores for this checkpoint is unknown, not known to be absent. The operator who prompts for people owns a per-group audit on their own images, stratified by the attributes their phrases mention, before relying on the output.

###### Instrumentation

The upstream data comes from web-crawled image–caption pairs for CLIP pre-training, and from public detection datasets such as COCO and OpenImages for the detection fine-tune, per the pinned README. Those images come from consumer cameras and web publishing, with annotation conventions set by each dataset. Inference images arrive from whatever produced them.

Resolution, blur, compression, exposure, viewpoint and aspect ratio all change the visual evidence. The resize to 768×768 alters every input. The text side is an instrument too: tokenisation lower-cases the phrase and caps it at 16 tokens. The pipeline checks only type, size, prompt count and prompt length. It cannot tell a rendered scene from a photograph, and it cannot tell an empty frame from a full one.

###### Environment

**Operating environment.** Python 3.12 with the pins in `pyproject.toml`: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`. Computation is float32. The code runs on CPU and uses CUDA automatically when available. One run with the pinned weights is recorded: Kaggle Tesla T4, 2026-09-25 UTC, torch 2.14.0+cu130 (CUDA 13.0), transformers 4.57.6, `cuda:0`. The whole notebook took 314.8 s wall including installs, one kernel restart and the 613 MB weight download; one `detect` call on the 640×480 sample took 0.53 s. No memory or throughput figure was measured.

**Data environment.** The model assumes photographs of scenes whose objects can be named in a short phrase of the kind found in web captions. Drawn graphics, documents, aerial, medical, thermal and microscopy images are distribution shifts. So are phrases in languages other than English and specialist terminology. When these assumptions fail, the model still returns boxes and scores, and the pipeline reports no signal that anything has shifted.

#### Metrics

###### Performance Measures

The pipeline reports no accuracy measure. Each detection carries `score`, the sigmoid of the best image–text logit for its patch, which is a ranking signal within one image and one prompt set.

`box_iou(a, b)` is the intersection-over-union of two xyxy boxes. It is the primitive every detection metric is built on, and it is the only measure the repository computes. `evaluation_report(result, ground_truth_boxes)` covers one image. With a reference box per phrase, it reports one `box_iou` per reference against the best-overlapping detection, with that detection's label and whether the label agrees, under the verdict `sample-sanity`. Without references it returns `not-measurable` and names the labelled data that would be needed.

Average precision, precision and recall are not implemented, because they need a labelled image set with a phrase vocabulary that matches the prompts. The caller must supply that set. The OWL-ViT paper reports results on LVIS and COCO; those are upstream-reported, and this repository does not reproduce them.

Values measured by this repository (one run on Kaggle Tesla T4, 2026-09-25 UTC; exact notebook blob `fff9ff981ddc`, commit `14be73f`; one pass, no dispersion estimate):

- **Drawn sample scene** (640×480, synthetic, three drawn shapes, phrases `a black rectangle`, `a red circle`, `a blue triangle`, threshold 0.1): **1 detection** — `a red circle`, score 0.4487, same-shape `box_iou` 0.9671 against the drawn box. The black rectangle and the blue triangle were **not detected** (`box_iou` 0.0; the only box is the circle's). So 1 of 3 drawn shapes was found. This is a `sample-sanity` check on one drawn image, not a detection evaluation.
- **Input validation:** the over-long phrase probe (49 characters) was rejected against `MAX_PROMPT_CHARS` 48.

The BYOD branch was not exercised in this run, and no box count at a lower threshold was recorded.

###### Decision thresholds

`detect` applies one threshold: a patch's box is kept when the sigmoid of its best image–text logit is at least `threshold`. Taking the best phrase per patch is an implicit argmax over the caller's phrases. The default, `DETECTION_THRESHOLD = 0.1`, is the value in the pinned README's example. It was not tuned or calibrated by this repository, and no acceptance threshold is set anywhere.

The deployment owns the threshold, and it must be re-set for each prompt set, because scores depend on phrasing. Lower it when a missed object costs more than a false box. Raise it when a false box triggers downstream action. Because there is no non-maximum suppression, a caller that needs one box per object must add suppression or clustering of its own.

###### Approaches to uncertainty and variability

The only measured values are three `box_iou` values from one image and one run, so there is no estimation procedure and no dispersion. Inference is deterministic on a fixed device and dtype: there is no sampling and no seed. CUDA kernel selection can move scores slightly and reorder near-ties across hardware.

A `score` is uncalibrated. A caller who needs calibrated confidence must fit a calibration map per prompt set on labelled images from the deployment. A caller who needs an uncertainty estimate for a metric must evaluate over many labelled images and resample.

#### Ethical considerations and biases

No external ethics board, red team, or population-specific review has examined this repository or, to our knowledge, the upstream checkpoint. Nothing below implies that one did.

###### Data

Per the pinned README, the CLIP backbone was trained on publicly available image–caption data, largely from web crawling and datasets such as YFCC100M. The detection heads and backbone were then fine-tuned on public detection datasets such as COCO and OpenImages. The crawled data was not enumerated by the authors. Personal data, and possibly copyrighted material, are present in it by construction or cannot be ruled out; neither was audited here.

This repository distributes code, tests, documentation, and one small configuration file copied from the upstream snapshot (`preprocessor_config.json`). It does not distribute `model.safetensors` (612,983,940 bytes) or the tokenizer files, which are staged locally and git-ignored. It ships no photographs: the tutorial scene is drawn in code.

The operator must audit the images and phrases they submit for personal, proprietary or restricted content; the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit or housing. Neither this repository, the upstream authors, nor any regulator has validated or certified it for any of them.

Foreseeable but unintended uses include prompting for people by appearance in surveillance footage and prompting for hazards or weapons in security screening. Such uses would be admissible only with human review of every acted-on detection, locally measured precision and recall per prompt set and per group, a documented threshold and re-validation policy, and any regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** while `MODEL_REVISION` is `"unpinned"`, `verify_snapshot`, `stage_missing_files` and `from_pretrained` raise before any download or model import. Once pinned, `stage_missing_files` refuses a manifest whose `modelId` or `revision` differs from the package constants. It fetches only manifest-listed files, and only with `allow_download=True`. `verify_snapshot` checks every file's byte size and SHA-256 and refuses an entry with no recorded digest. `from_pretrained` loads only the verified directory, with `local_files_only=True` and `trust_remote_code=False`. The upstream `pytorch_model.bin` pickle is not in the manifest and is never staged.
- **Tests of those refusals:** tests assert that an unpinned package, a missing snapshot and a tampered digest are refused before `torch` or `transformers` is imported.
- **Input integrity:** `validate_inputs` and `detect` share one checker for the image, the prompts and the threshold. `format_prompts` enforces the prompt count, length and distinctness limits. `detect` raises on a malformed backend detection, a label that is not one of the normalised phrases, or more than 576 detections.
- **Reproducibility:** exact `==` pins in `pyproject.toml`, carried into the notebook and checked by the parity tests. Every result records `model_id` and `model_revision`.
- **Refusals:** no download without the explicit flag, no pickle deserialisation, no remote model code, and no silent fallback to a different checkpoint.
- **Statistical mitigations:** none applies, because no training or fitting happens in this repository.

###### Risks and harms

- **Confident boxes for absent objects.** Each phrase is scored against every patch, so a phrase for something absent can still produce a box above a low threshold. The operator and downstream consumers bear the harm of a fabricated detection.
- **Prompt sensitivity.** Rewording a phrase changes scores and results, so two operators prompting for the same object can get different output. A threshold tuned on one wording does not transfer to another.
- **Duplicate boxes.** Without non-maximum suppression, counts built on raw detections overstate the number of objects.
- **Biased associations.** Phrases about people can surface associations learned from web captions, and people in the processed images bear the harm of unequal error.
- **Surveillance.** Free-text prompts make it easy to search images for people with particular attributes. The people in the processed images bear the harm of misuse.
- **Automation bias.** A labelled box looks authoritative even when its score is low and uncalibrated. Operators who skip review turn a model error into a decision error.
- **Resource use.** The fixed 768×768 pass costs the same for small and large inputs, and a video stream can saturate a shared host.

###### Use cases

The following uses are prohibited even where the model would work:

- prompting for people by appearance, clothing, perceived demographic attributes or identity in order to surveil, track, profile or score them;
- unlawful discrimination in employment, housing, credit, insurance, education, healthcare access or law enforcement;
- processing images the operator has no right to process, or in breach of consent, privacy or data-protection obligations;
- deceptive uses that present detections as verified facts or as evidence;
- autonomous physical control, moderation or safety decisions driven by unreviewed detections;
- any use that violates the upstream Apache-2.0 licence or the terms of the deployment running the pipeline.

## Immutable provenance

- Model: `google/owlvit-base-patch32`
- Revision: `cbc355fb364588351c5d51c7f74465e8e7ec6f72` (pinned 2026-09-25 by `python tools/pin_snapshot.py`, which resolved the Hub's `main` to this commit, downloaded every manifest file at it, and recorded each file's SHA-256).
- Snapshot manifest: `weights/owlvit-base-patch32/dimer-base-manifest.json`, 8 files, `totalBytes` 614579712. The byte sizes and digests describe the files at the pinned commit.
- `model.safetensors`: 612,983,940 bytes; SHA-256 `4dbe0399f0b7d7c8dddf1535a98769cc30743bebc877aea681998c8d984ce52b` (matches the Hub's LFS record).
- `config.json`: 4,418 bytes; `OwlViTForObjectDetection`, ViT-B/32 image tower at 768×768, text tower with 16 positions, projection 512.
- `preprocessor_config.json`: 392 bytes; resize to 768×768, `do_center_crop` false, CLIP mean and standard deviation.
- Tokenizer: `vocab.json` (1,059,962 bytes), `merges.txt` (524,619 bytes), `tokenizer_config.json` (775 bytes), `special_tokens_map.json` (460 bytes); `CLIPTokenizer`, `model_max_length` 16.
- `README.md`: 5,146 bytes; the upstream model card.
- Loader: `OwlViTForObjectDetection.from_pretrained(<verified dir>, local_files_only=True, trust_remote_code=False)`, with `OwlViTProcessor` from the same directory.

## Input/output contract

- `OwlViTDetectionPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)`: stage (only with `allow_download=True`), verify, load. `OwlViTDetectionPipeline.from_components(model, processor, device)` wraps an already-loaded model and processor.
- `detect(image, prompts, *, threshold=0.1) -> dict`: keys `detections` (list of `{"box": [x0, y0, x1, y1], "label": str, "score": float}`, input pixels, descending score, at most 576), `queries` (the normalised phrases), `threshold`, `width`, `height`, `model_id`, `model_revision`.
- `validate_inputs(image, prompts, *, threshold=0.1, names=None) -> dict`; `evaluation_report(result, ground_truth_boxes=None, *, sample_kind="synthetic") -> dict`; `format_prompts(prompts) -> list[str]`; `box_iou(a, b) -> float`.
- Constants: `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_PROMPTS = 16`, `MAX_PROMPT_CHARS = 48`, `MAX_TEXT_TOKENS = 16`, `MAX_DETECTIONS = 576`, `DETECTION_THRESHOLD = 0.1`.

## Verification records

Default-path execution recorded on 2026-09-25 (Kaggle T4): exact notebook blob `fff9ff981ddc` at commit `14be73f`, 314.8 s, 8/8 post-restart code cells, BYOD off; measured values are under Metrics. REL12 BYOD exercise pending before promotion. The offline test suite runs a tiny random-weight OWL-ViT with a toy CLIP vocabulary through the real processor, model and post-processing; that exercises the code path and is not a result about this model. `docs/release-verification.md` holds the release gate and the record table.

## References

- Minderer et al. Simple Open-Vocabulary Object Detection with Vision Transformers. ECCV 2022. https://arxiv.org/abs/2205.06230
- Radford et al. Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021. https://arxiv.org/abs/2103.00020
- Upstream code: https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit
- Upstream card: https://huggingface.co/google/owlvit-base-patch32
- Transformers OWL-ViT documentation: https://huggingface.co/docs/transformers/model_doc/owlvit
