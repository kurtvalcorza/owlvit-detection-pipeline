"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.1 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "owlvit_detection_pipeline",
    "repo_name": "owlvit-detection-pipeline",
    "stem": "owlvit_detection",
    "notebook_name": "owlvit_detection_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (generator /2.2): managed CPython, a
    # size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "OwlViTDetectionPipeline",
    "weights_key": "owlvit-base-patch32",
    "runtime_imports": ["torch", "transformers"],
    "title": "OWL-ViT base/32 — DIMER open-vocabulary object detection tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/owlvit-detection-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/owlvit-detection-pipeline/blob/main/tutorials/owlvit_detection_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fowlvit--base--patch32-ffcc4d?style=flat",
            "https://huggingface.co/google/owlvit-base-patch32",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-google--research%2Fscenic%20(owl__vit)-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2205.06230-b31b1b.svg", "https://arxiv.org/abs/2205.06230"),
    ],
    "capability": "zero-shot (open-vocabulary, text-prompted) object detection using the pinned `google/owlvit-base-patch32` weights",
    "intro": (
        "OWL-ViT (Vision Transformer for Open-World Localization) turns a CLIP model into a detector by removing the image tower's "
        "final pooling and attaching a small box head and class head to every output token. At inference the image is resized "
        "to 768×768 **without preserving its aspect ratio and without padding**, and split into 24×24 = 576 patches by the CLIP "
        "ViT-B/32 image tower; each of the caller's phrases is encoded by the CLIP text tower "
        "into one query embedding; every image patch proposes one box and is scored against every query by a **sigmoid** of "
        "the image–text logit; the box survives when its best query score reaches the threshold and is labelled with that "
        "query. The pipeline returns each surviving box in xyxy pixel coordinates of the input image, the phrase it matched, "
        "and its score. **No adaptation occurs:** no training, fine-tuning, in-context conditioning, or preprocessing "
        "fitting happens in this notebook — the upstream checkpoint supplies the weights, processor and tokenizer, and the "
        "carried pipeline module adds snapshot verification, prompt and image validation with named ceilings, a fixed output "
        "contract and the `box_iou`, `validate_inputs` and `evaluation_report` helpers. The default sample is a synthetic "
        "scene drawn in code; the IoU values reported for it are sanity evidence against the boxes you drew, not a benchmark claim."
    ),
    "guided": {"opening": [(
        "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter and has met bounding boxes, and wants to see how an open-vocabulary detector finds objects named by free-text phrases, what its scores and threshold mean, and why a box with a label is not proof that the object is there. The audience is students and practitioners deciding whether text-prompted detection fits their own images; no prior experience with OWL-ViT or CLIP is assumed — each term is explained where it first matters and again in the **Glossary**. CPU is enough; nothing is trained.\n\n**Input → Model → Output.**\n\n| | |\n|---|---|\n| Input | one image (resized to 768 × 768 without padding) and 1–16 free-text phrases; the default is a drawn 640 × 480 scene of a black rectangle, a red disc and a blue triangle |\n| Model | OWL-ViT base/32: a CLIP image tower whose 576 patches each propose one box, a CLIP text tower that embeds each phrase, and a sigmoid score per (box, phrase) |\n| Output | the boxes whose best score reaches the caller-owned threshold, each with its phrase and score (no non-maximum suppression), an annotated image, and an evaluation report that is `sample-sanity` only against boxes you drew yourself |\n\n**How to use this notebook.** Choose any runtime, then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed (the recorded hosted run of the previous version needed one; this version removes it). Sections 1–3 are **infrastructure** — the isolated environment, the carried module and the verified snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer from the recorded Kaggle T4 run of 25 September 2026. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the drawn scene (or your image) → 5 validation and a deliberate refusal *(core concept: the input contract)* → 6 detect *(core concept: sigmoid scores, a caller-owned threshold, no NMS)* → 7 the evaluation report and `box_iou` *(evaluation practice)* → 8 export and the annotated image *(engineering)* → conclude."
    )]},
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, draw a synthetic scene with known object boxes, validate the image and the "
        "prompts into an input manifest through the pipeline's own validation stage, run text-prompted detection through "
        "the public API with an explicit caller-owned threshold, read sigmoid scores and the absence of non-maximum "
        "suppression correctly, produce an evaluation report that is `sample-sanity` with `box_iou` only when reference "
        "boxes exist and `not-measurable` otherwise, exercise an optional BYOD path, and export machine-readable detections "
        "plus an annotated image and provenance."
    ),
    "exclusions": (
        "instance or semantic segmentation (see the sibling SAM 2 pipeline), tracking, OCR, captioning, closed-set "
        "detection with a fixed class list (see the sibling RT-DETR pipeline), image-guided (one-shot) detection, mAP or "
        "precision/recall evaluation (which needs a labelled box set), or any training. Prompts are free text limited to "
        "16 CLIP tokens each, so a long phrase is silently truncated; and there is no non-maximum suppression, so one "
        "object can surface as several overlapping boxes at a low threshold — the score, not the label, is your only signal."
    ),
    "prerequisites": [
        '- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with OWL-ViT or CLIP. Sigmoid scores, thresholds, non-maximum suppression, IoU and the evaluation verdicts are explained where they are first used and again in the Glossary.',
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the Python version of the kernel itself does not matter and nothing is installed into it; a Windows or macOS kernel is not supported. The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. Runtimes are not measured in this revision; the cost is dominated by the fixed 768×768 ViT-B/32 pass, not by the input image size. The pinned `torch==2.14.0` install and the ~613 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what a bounding box in xyxy pixel coordinates is; what intersection-over-union measures; what a sigmoid score is and why it is not a probability.",
        "- **Data:** the default sample is a deterministic 640×480 scene drawn in code (grey background, a black rectangle, a red disc, a blue triangle) with three prompts naming them, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image file decodable by Pillow (PNG/JPEG/WebP and similar), any colour mode, sides between 16 and 4096 px, plus your own comma-separated prompt phrases (1–16 distinct phrases, at most 48 characters each; the upstream convention is `a photo of a <thing>`). Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Draw the synthetic scene or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own reference boxes: a deterministic 640×480 RGB scene is "
                "drawn in code — grey background, a black filled rectangle at `[80, 120, 280, 360]`, a red filled disc whose "
                "bounding box is `[380, 140, 560, 320]` and a blue filled triangle whose bounding box is `[200, 380, 360, 460]` — "
                "and the prompts name them (`a black rectangle`, `a red circle`, `a blue triangle`). The drawn boxes are the reference for the `box_iou` sanity check "
                "later; they are not a labelled dataset, so nothing here is a precision/recall measurement. The image digest is "
                "printed for the record. BYOD is optional and disabled by default; when enabled, set `BYOD_IMAGE_PATH` to an image "
                "file (or leave it empty on Colab to get an upload dialog) and set "
                "`BYOD_PROMPTS` to the phrases you want found — no reference boxes exist for it, so the evaluation report will be "
                "`not-measurable`.\n\n"
                "The detection threshold is a **caller-owned request parameter**, not a pipeline constant: a patch's box survives "
                "when the sigmoid of its best image–text logit reaches it. The package default (`DETECTION_THRESHOLD = 0.1`) "
                "follows the pinned README's usage example, not a calibration; it is exposed here as a form parameter and passed "
                "explicitly on every call. Nothing is validated in this cell — the next section hands the image, the prompts and "
                "the threshold to the pipeline's own validation stage, which is the only checker. Look for a dictionary naming "
                "the sample kind, the image size and digest, the prompts, the threshold, and the drawn reference boxes."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import os\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_IMAGE_PATH = ''  # @param {{type:\"string\"}}\n"
                "BYOD_PROMPTS = 'a photo of a cat, a photo of a remote control'  # @param {{type:\"string\"}}\n"
                "threshold = 0.1  # @param {{type:\"number\"}}\n\n"
                "if USE_BYOD and BYOD_IMAGE_PATH:\n"
                "    # Location field (EXE2): read the file directly, no upload dialog.\n"
                '    if not os.path.isfile(BYOD_IMAGE_PATH):\n'
                "        raise FileNotFoundError(f'BYOD_IMAGE_PATH {{BYOD_IMAGE_PATH!r}} does not exist or is not a file (relative paths start at {{os.getcwd()}}).')\n"
                '    image_name = os.path.basename(BYOD_IMAGE_PATH)\n'
                '    image = Image.open(BYOD_IMAGE_PATH)\n'

                "    image.load()\n"
                'if USE_BYOD and not BYOD_IMAGE_PATH:\n'
                '    try:\n'
                '        from google.colab import files\n'
                '    except ImportError:\n'
                "        raise RuntimeError('USE_BYOD is on but BYOD_IMAGE_PATH is empty, and the upload dialog exists only in Google Colab: copy the image into this runtime and set BYOD_IMAGE_PATH.') from None\n"
                '    uploaded = files.upload()\n'
                '    if len(uploaded) != 1:\n'
                "        raise ValueError(f'Upload exactly one image file (received {{len(uploaded)}} files; a cancelled dialog sends none). Run this cell again.')\n"
                '    image_name = next(iter(uploaded))\n'
                '    image = Image.open(io.BytesIO(uploaded[image_name]))\n'
                '    image.load()\n'

                "if USE_BYOD:\n"
                "    prompts = [phrase.strip() for phrase in BYOD_PROMPTS.split(',') if phrase.strip()]\n"
                "    drawn_boxes = None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic synthetic scene: no randomness, so no seed is needed and the digest is stable.\n"
                "    image = Image.new('RGB', (640, 480), (128, 128, 128))\n"
                "    draw = ImageDraw.Draw(image)\n"
                "    drawn_boxes = {{'a black rectangle': [80.0, 120.0, 280.0, 360.0], 'a red circle': [380.0, 140.0, 560.0, 320.0], 'a blue triangle': [200.0, 380.0, 360.0, 460.0]}}\n"
                "    draw.rectangle(drawn_boxes['a black rectangle'], fill=(30, 30, 30))\n"
                "    draw.ellipse(drawn_boxes['a red circle'], fill=(220, 30, 30))\n"
                "    draw.polygon([(280, 380), (200, 460), (360, 460)], fill=(30, 60, 220))\n"
                "    prompts = list(drawn_boxes)\n"
                "    image_name = 'synthetic_scene_640x480.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256, 'prompts': prompts, 'threshold': threshold, 'drawn_boxes': drawn_boxes}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `detect` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px, 1..`MAX_PROMPTS` distinct phrases of at most "
                "`MAX_PROMPT_CHARS` characters each, and a threshold in `[0, 1]` — canonicalises the phrases through the "
                "package's `format_prompts` (whitespace-collapsed, lower-cased, trailing period removed, one text query each) and "
                "returns an **input manifest** naming the schema and ceilings (including the 576-patch ceiling on detections and "
                "the 16-token CLIP limit per query), the input's observed mode and size, the queries actually sent to the "
                "tokenizer, the threshold, and the verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To "
                "show what rejection looks like, the cell also validates a deliberately over-long phrase and records the "
                "pipeline's own error message as a finding. Inside the pipeline the image is converted to RGB and resized to "
                "768×768 without padding by the processor; boxes are mapped back to input pixels, and nothing else is dropped or altered."
                '\n\n**Predict before running:** the cell also validates a deliberately bad phrase. Which rule will refuse it, and will the model ever see it?'
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PROMPTS': MAX_PROMPTS, 'MAX_PROMPT_CHARS': MAX_PROMPT_CHARS, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS, 'MAX_DETECTIONS': MAX_DETECTIONS, 'DETECTION_THRESHOLD': DETECTION_THRESHOLD}}}})\n"
                "input_manifest = validate_inputs(image, prompts, threshold=threshold, names=[image_name])\n"
                "# Demonstrate rejection on a request that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, ['x' * (MAX_PROMPT_CHARS + 1)])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'over-long-prompt-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>The phrase is longer than `MAX_PROMPT_CHARS` (48 characters; the recorded run used a 49-character phrase), so `validate_inputs` refuses it and the refusal is recorded as a finding in the input manifest — before any model work. The same checks run inside `detect`, so what is accepted here is exactly what the model may receive.</details>'
            ),
        },
        {
            "md": (
                "## 6. Detect and read the scores correctly\n\n"
                "`detect` returns a dict with `detections` — a list of `{{box, label, score}}` **ordered by descending score**, "
                "`box` in xyxy pixel coordinates of the input, `label` the matched query text — plus `queries`, the threshold "
                "used, `width`, `height` and the model identity. At most 576 boxes can ever be returned (one per image patch). "
                "Each `score` is a **sigmoid of the best image–text logit, not a calibrated probability**: it was never fitted "
                "to the frequency with which a box is correct, so 0.9 does not mean \"90 % likely\", and the scores of different "
                "queries for one patch are independent. The threshold you passed is the only decision rule; the pipeline ships "
                "0.1 as a default (the README example's value), not as a calibration, and the caller owns it per deployment — "
                "raise it when false boxes cost more than missed ones, lower it for recall. **There is no non-maximum "
                "suppression**: neighbouring patches can propose overlapping boxes for the same object, and at a low threshold "
                "the same object appears more than once. Inference is deterministic on a fixed device and dtype (no sampling, "
                "`torch.inference_mode`); CUDA kernel selection can move scores in the third or fourth decimal place and reorder "
                "near-ties. No run with the pinned weights has been recorded for this checkpoint yet, so look at the boxes and scores "
                "yourself: they are one observation on one drawn scene, not a calibration point."
                '\n\n**Predict before running:** the scene holds three clean, high-contrast shapes and three prompts that name them. At threshold 0.1, how many boxes will come back, and will each shape get one?'
            ),
            "code": (
                "import time\n\n"
                "t0 = time.time()\n"
                "result = pipe.detect(image, prompts, threshold=threshold)\n"
                "elapsed = time.time() - t0\n"
                "print({{'n_detections': len(result['detections']), 'queries': result['queries'], 'threshold': result['threshold'], 'device': pipe.device, 'seconds': round(elapsed, 2)}})\n"
                "for rank, det in enumerate(result['detections'], start=1):\n"
                "    print(f\"{{rank:>2}}. score {{det['score']:.4f}}  label {{det['label']!r}}  box {{[round(v, 1) for v in det['box']]}}\")"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Fewer than you might expect. In the recorded run `detect` returned **one** box: `a red circle`, score 0.4487, box [378.1, 137.5, 560.0, 321.7]. The black rectangle and the blue triangle were **not detected** at 0.1. Drawn shapes on grey are far from the photographs the model learned from, and short prompts score lower than the upstream `a photo of …` form: a clean scene is not an easy scene for an open-vocabulary detector.</details>'
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No detection metric "
                "is reported by default: mean average precision needs a labelled box set with a matching vocabulary, and this "
                "repository ships none. The repository's only metric helper is `box_iou(a, b)` (intersection-over-union of two "
                "xyxy boxes), the building block a caller would use to compute mAP on their own labelled data; when reference "
                "boxes are supplied the report carries one `box_iou` entry per reference — its value, which detection matched it "
                "best and whether that detection's label agrees — with the verdict `sample-sanity`. On the synthetic path those "
                "references are shapes **you drew yourself**, so a high IoU proves only that the input contract, query "
                "formatting, forward pass and coordinate mapping (including the non-uniform resize) round-trip. On BYOD no reference "
                "exists, the verdict is `not-measurable`, and the report states what would make the task measurable: labelled "
                "boxes on your own images with a phrase vocabulary matching the prompts. The report is written to "
                "`outputs/{stem}_evaluation_report.json`."
                "\n\n**Predict before running:** which verdict will the report give for the drawn scene, and what IoU will the red circle's box reach against the disc you drew?"
            ),
            "code": (
                "report = evaluation_report(result, drawn_boxes, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No reference boxes exist for this input, so box_iou is not computed; inspect the annotated PNG instead.')"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>`sample-sanity`, because reference boxes exist (you drew them). In the recorded run the red circle's IoU was 0.9671 with a matching label, and the rectangle and triangle scored 0.0 (no box). One drawn scene with three references is geometry sanity evidence — the coordinate mapping works — not a detection benchmark; a BYOD image with no reference boxes gets `not-measurable`.</details>"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the full result (score-ordered detections with boxes and labels, the queries, "
                "the threshold), the evaluation report, the input manifest, the sample identity, digest and drawn boxes, the "
                "notebook's source (repository, revision, embedded module digest, generator), the model identifier, the immutable "
                "model revision, the model licence, and the runtime identity (Python, `torch`, `transformers`, device). The "
                "detections are also written as CSV with explicit `image`, `rank`, `label`, `score`, `x0`, `y0`, `x1`, `y1` "
                "columns so score ordering survives downstream use, and an annotated PNG draws every returned box for visual "
                "inspection (a supplement to, not a replacement for, the machine-readable files). No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "annotated = image.convert('RGB').copy()\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "for det in result['detections']:\n"
                "    draw.rectangle(det['box'], outline=(0, 255, 0), width=2)\n"
                "    draw.text((det['box'][0] + 2, det['box'][1] + 2), f\"{{det['label']}} {{det['score']:.2f}}\", fill=(0, 255, 0))\n"
                "annotated.save('outputs/{stem}_annotated.png')\n"
                "payload = {{\n"
                "    'prediction': result,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'prompts': prompts, 'drawn_boxes': drawn_boxes}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_detections.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'rank', 'label', 'score', 'x0', 'y0', 'x1', 'y1'])\n"
                "    for rank, det in enumerate(result['detections'], start=1):\n"
                "        writer.writerow([image_name, rank, det['label'], f\"{{det['score']:.6f}}\", *[f\"{{v:.2f}}\" for v in det['box']]])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The boxes are matched to free-text phrases: the label tells you which query the box scored best against, not that the "
        "object is really there, and the sigmoid score is uncalibrated. The threshold is a request parameter you own; the "
        "default is the README usage example, not a tuned operating point. On the synthetic scene the `box_iou` values in the "
        "evaluation report compare detections to shapes you drew yourself and the verdict is `sample-sanity`, which proves only "
        "that the input contract, query formatting, forward pass and coordinate mapping work; they say nothing about "
        "photographs, small or occluded objects, crowded scenes, or vocabulary the model has never seen, and a BYOD result is a "
        "single-image observation with the verdict `not-measurable`. Prompts longer than 16 CLIP tokens are truncated silently, "
        "the image is squashed to 768×768 so a wide or tall image is distorted and each of its 32×32-pixel patches covers a "
        "different share of the scene, and without non-maximum suppression one object can surface as several boxes at a low "
        "threshold. The pipeline provides no "
        "segmentation, tracking, OCR, captioning, mAP evaluation, or training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** lower `threshold` to 0.05 and count the duplicate boxes; rephrase the prompts in the upstream form (`a photo of a red circle`) and compare scores; add a prompt for "
        "something that is not in the scene (`a green star`) and see whether anything surfaces; enable `USE_BYOD` with a "
        "photograph, hand-label a few objects and pass them to `evaluation_report` to see the verdict switch to `sample-sanity` — "
        "the first step towards a real precision/recall number.\n\n"
        '## Troubleshooting\n\n'
        '- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n'
        '- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n'
        '- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n'
        '- **"The isolated environment\'s Python process exited"** — usually out of memory; restart the session and choose **Run all**.\n'
        '- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete the folder Section 3 prints as `weights_dir` and run Section 3 again.\n'
        '- **A shape is not detected** — that is a finding, not an error: lower `threshold` (and expect duplicate boxes), or rephrase the prompt (`a photo of a …`).\n'
        '- **BYOD: "BYOD_IMAGE_PATH … does not exist" / "the upload dialog exists only in Google Colab" / "Upload exactly one image file"** — set `BYOD_IMAGE_PATH` to an image in the runtime (it works on Kaggle and Jupyter); on Colab an empty path opens the dialog, and a cancelled dialog stops with that message.\n'
        '- **A `ValueError` from `validate_inputs`** — it names the rule: a phrase longer than 48 characters, more than 16 phrases, or an image side outside the ceilings.\n\n'
        '## Glossary\n\n'
        '- **Open-vocabulary detection** — detecting objects named by free-text phrases instead of a fixed class list.\n'
        '- **Patch / box head / class head** — the image is cut into 24 × 24 = 576 patches; each proposes one box (box head) and is scored against every phrase (class head).\n'
        '- **Sigmoid score** — one (box, phrase) logit squashed into 0..1 on its own; uncalibrated, not a probability that the object is there.\n'
        "- **Threshold** — the caller-owned cut-off a box's best score must reach; `0.1` follows the upstream example, not a calibration.\n"
        '- **Non-maximum suppression (NMS)** — merging overlapping boxes; none is applied, so one object can appear as several boxes.\n'
        '- **IoU** — intersection over union between a detected box and a reference box (1 = identical).\n'
        '- **`sample-sanity` / `not-measurable`** — the verdict against boxes you drew yourself, and without any reference boxes.\n'
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock; every later cell runs there.\n'
        '- **BYOD** — bring your own data: your image and phrases through the same cells.\n\n'
        '## Conclusion (your notes)\n\nOptional — fill in from **your** run:\n\n'
        '- At threshold ___ the model returned ___ box(es); the shapes found were ___ and missed were ___.\n'
        '- The best IoU was ___ for ___; the verdict was ___.\n'
        '- Lowering the threshold to ___ changed ___.\n'
        '- One reason this does not tell me how the model works on my photographs: ___.\n\n'
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/owlvit-detection-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/owlvit-detection-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/owlvit-detection-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code (Scenic, OWL-ViT project): https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit\n"
        "- Simple Open-Vocabulary Object Detection with Vision Transformers (Minderer et al., 2022): https://arxiv.org/abs/2205.06230"
    ),
}
