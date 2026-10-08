"""Open-vocabulary (text-prompted) object detection with the pinned ``google/owlvit-base-patch32``
checkpoint (OWL-ViT, ViT-B/32).

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``),
always with ``trust_remote_code=False``: the OWL-ViT architecture comes from the pinned ``transformers``
release, the weights are SafeTensors, and no model-repository code is executed.

Until ``tools/pin_snapshot.py`` has recorded an immutable revision and every file's SHA-256, the package
refuses to stage, verify or load weights: an unpinned snapshot is never trusted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "google/owlvit-base-patch32"
MODEL_REVISION = "cbc355fb364588351c5d51c7f74465e8e7ec6f72"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "owlvit-base-patch32"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
UNPINNED = "unpinned"
PIN_COMMAND = "python tools/pin_snapshot.py"

# Threshold: the value the pinned README's usage example passes to post_process_object_detection
# (threshold=0.1). It gates a sigmoid over the best text query per image patch that is not calibrated;
# the deployment owns tuning it on its own labelled data.
DETECTION_THRESHOLD = 0.1
# The ViT-B/32 image tower sees a 768x768 image as 24x24 = 576 patch tokens, each of which is one
# detection candidate, so no image can yield more than this many boxes.
MAX_DETECTIONS = 576
# Input ceilings. The processor resizes the image to 768x768 without preserving its aspect ratio and
# without padding (preprocessor_config.json: size [768, 768], do_center_crop false), so image cost is
# bounded; each text query is tokenised by the CLIP tokenizer, whose text tower has 16 positions (start and
# end tokens included). The processor does not truncate a longer query: it fails inside the model with a
# tensor-shape error, so a loaded pipeline refuses such a phrase before the model runs (review OVT-M2).
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
MAX_PROMPTS = 16
MAX_PROMPT_CHARS = 48
MAX_TEXT_TOKENS = 16
# IoU at or above which a reference counts as localised by a box (the conventional AP50 overlap); used only
# to tell a score miss from a localisation miss in the sanity report, never as a quality claim.
LOCALISATION_IOU = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_pinned() -> bool:
    """True once MODEL_REVISION names an immutable 40-hex commit."""
    revision = MODEL_REVISION
    return len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)


def _require_pinned(action: str) -> None:
    if not is_pinned():
        raise RuntimeError(
            f"refusing to {action}: {MODEL_ID} has no pinned revision yet (MODEL_REVISION = "
            f"{MODEL_REVISION!r}); run `{PIN_COMMAND}` to record the commit and every file's SHA-256"
        )


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    _require_pinned("verify the snapshot")
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        if not entry.get("sha256"):
            raise ValueError(f"{entry['path']}: manifest records no sha256; run `{PIN_COMMAND}`")
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    _require_pinned("stage weights")
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two xyxy pixel boxes; the building block for any caller-side mAP."""
    if len(a) != 4 or len(b) != 4:
        raise ValueError("boxes must be [x0, y0, x1, y1]")
    if a[2] < a[0] or a[3] < a[1] or b[2] < b[0] or b[3] < b[1]:
        raise ValueError("boxes must satisfy x0 <= x1 and y0 <= y1")
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def format_prompts(prompts: Sequence[str]) -> list[str]:
    """Validate a list of phrases and normalise them to the OWL-ViT query form: stripped, lower-cased, one
    text query per phrase (the upstream example uses "a photo of a cat"-style queries; the pipeline
    passes the caller's phrases through unchanged apart from case and whitespace)."""
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a list of phrases, not a single string")
    if not 1 <= len(prompts) <= MAX_PROMPTS:
        raise ValueError(f"prompt count {len(prompts)} outside 1..MAX_PROMPTS {MAX_PROMPTS}")
    cleaned: list[str] = []
    for phrase in prompts:
        if not isinstance(phrase, str):
            raise TypeError(f"prompt must be str, got {type(phrase).__name__}")
        text = " ".join(phrase.split()).strip().rstrip(".").strip().lower()
        if not text:
            raise ValueError("prompt phrases must not be empty")
        if len(text) > MAX_PROMPT_CHARS:
            raise ValueError(
                f"prompt {text[:12]!r}... is {len(text)} chars > MAX_PROMPT_CHARS {MAX_PROMPT_CHARS}"
            )
        cleaned.append(text)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("prompt phrases must be distinct after normalisation")
    return cleaned


def check_prompt_tokens(queries: Sequence[str], count_tokens: Callable[[str], int]) -> list[int]:
    """Return each query's CLIP token count (start and end tokens included); raise ValueError naming the first
    query over MAX_TEXT_TOKENS. Hyphens, digits and brackets cost extra tokens, so a phrase within
    MAX_PROMPT_CHARS can still be too long for the text tower (review OVT-M2)."""
    counts: list[int] = []
    for query in queries:
        n_tokens = int(count_tokens(query))
        if n_tokens > MAX_TEXT_TOKENS:
            raise ValueError(
                f"prompt {query!r} is {n_tokens} CLIP tokens (start and end tokens included) > "
                f"MAX_TEXT_TOKENS {MAX_TEXT_TOKENS}; shorten it "
                "(hyphens, digits and brackets each cost extra tokens)"
            )
        counts.append(n_tokens)
    return counts


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


def _check_threshold(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
    return float(value)


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one PIL.Image.Image (any mode, converted to RGB) plus 1..MAX_PROMPTS free-text phrases",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "prompts": [1, MAX_PROMPTS],
    "prompt_chars": [1, MAX_PROMPT_CHARS],
    "prompt_tokens_per_query": [1, MAX_TEXT_TOKENS],
    "threshold": [0.0, 1.0],
    "max_detections": MAX_DETECTIONS,
    "preprocessing": (
        "image converted to RGB and resized to 768x768 without preserving the aspect ratio and without "
        "padding (CLIP mean/std); phrases stripped and lower-cased into one CLIP text query each "
        "(format_prompts); a query over 16 CLIP tokens is refused (check_prompt_tokens), never truncated; "
        "returned boxes are mapped back to input pixels"
    ),
}


def _check_inputs(
    image: Any, prompts: Any, threshold: Any, count_tokens: Callable[[str], int] | None = None
) -> tuple[Image.Image, list[str], float]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``detect`` and ``validate_inputs`` both route through this function so their acceptance
    criteria cannot diverge. With ``count_tokens`` (a loaded pipeline's ``count_tokens``) the
    16-token text limit is checked too.
    """
    rgb = validate_image(image)
    queries = format_prompts(prompts)
    checked = _check_threshold("threshold", threshold)
    if count_tokens is not None:
        check_prompt_tokens(queries, count_tokens)
    return rgb, queries, checked


def validate_inputs(
    image: Image.Image,
    prompts: Sequence[str],
    *,
    threshold: float = DETECTION_THRESHOLD,
    names: Sequence[str] | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Rejection is reported by raising exactly as ``detect`` would; a caller that wants the finding
    recorded catches the exception and stores ``str(exc)`` under ``findings``. Pass the loaded
    pipeline's ``count_tokens`` so the 16-token text limit is checked here, before any model call;
    without it the manifest records ``prompt_tokens`` as null (not checked).
    """
    _rgb, queries, checked = _check_inputs(image, prompts, threshold, count_tokens)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (detect takes one image)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": names[0] if names else "image-0",
                "mode": image.mode,
                "size": list(image.size),
                "n_prompts": len(prompts),
            }
        ],
        "queries": queries,
        "prompt_tokens": check_prompt_tokens(queries, count_tokens) if count_tokens is not None else None,
        "threshold": checked,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any],
    ground_truth_boxes: Mapping[str, Sequence[float]] | None = None,
    *,
    sample_kind: str = "synthetic",
    candidates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``ground_truth_boxes`` (phrase -> xyxy reference box) the report carries one ``box_iou``
    entry per reference as sample-sanity geometry evidence; without them the verdict is
    ``not-measurable`` and the report says what labelled data would make the task measurable.
    A reference that no returned box overlaps gets ``matched_label`` null (review OVT-m6). With
    ``candidates`` (the same request detected at threshold 0) the report adds
    ``per_reference_best_box`` (see ``best_box_per_reference``).
    """
    detections = list(result["detections"])
    base = {
        "task": "zero-shot (open-vocabulary, text-prompted) object detection",
        "decision_rule": (
            "each of the 576 image patches proposes one box labelled with its best-matching text query; "
            "the box survives when the sigmoid of that best image-text logit reaches the threshold; the "
            "score is an uncalibrated sigmoid, not a probability, and is not exclusive across queries"
        ),
        "threshold": result.get("threshold", DETECTION_THRESHOLD),
        "sample_kind": sample_kind,
        "n_detections": len(detections),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if not ground_truth_boxes:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no ground-truth boxes were supplied for the evaluated image",
            "needs": (
                "labelled boxes on your own images with a phrase vocabulary matching the prompts, "
                "scored per object with box_iou and aggregated into precision/recall or mean average "
                "precision at a stated IoU threshold; no such labelled set ships with this repository"
            ),
        }
    metrics = []
    for phrase, box in ground_truth_boxes.items():
        ious = [box_iou(det["box"], box) for det in detections]
        best = max(range(len(ious)), key=ious.__getitem__) if ious else None
        if best is not None and ious[best] <= 0.0:
            best = None  # no returned box overlaps this reference: nothing was matched to it
        entry = {
            "id": "box_iou",
            "reference": phrase,
            "value": ious[best] if best is not None else 0.0,
            "matched_label": detections[best]["label"] if best is not None else None,
            "label_matches_reference": (detections[best]["label"] == phrase) if best is not None else False,
            "estimation": "one reference box per phrase on a single scene, no dispersion estimate",
        }
        if best is None:
            entry["note"] = "no overlapping detection at this threshold"
        metrics.append(entry)
    extra: dict[str, Any] = {}
    if candidates is not None:
        extra["per_reference_best_box"] = best_box_per_reference(
            candidates, ground_truth_boxes, base["threshold"]
        )
    return {
        **base,
        "metrics": metrics,
        **extra,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(metrics)} reference box(es) on one tutorial sample; geometry sanity evidence, "
            "not a detection benchmark"
        ),
        "needs": (
            "a labelled box set from the deployment domain with a matching phrase vocabulary for any "
            "mean-average-precision or precision/recall claim"
        ),
    }


def best_box_per_reference(
    candidates: Mapping[str, Any],
    ground_truth_boxes: Mapping[str, Sequence[float]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Tell a score miss from a localisation miss (review OVT-M3).

    ``candidates`` is ``detect`` on the same image and prompts at threshold 0, so every patch's box is
    returned with its best-matching phrase. For each reference phrase this reports the highest-scoring
    candidate labelled with that phrase, its score, its IoU with the reference, whether that score reaches
    ``threshold``, and a reading: found at the threshold, localised but scored under it, or not localised
    (best box IoU below ``LOCALISATION_IOU``).
    """
    rows: list[dict[str, Any]] = []
    for phrase, box in ground_truth_boxes.items():
        query = format_prompts([phrase])[0]
        own = [det for det in candidates["detections"] if det["label"] == query]
        if not own:
            rows.append(
                {
                    "reference": phrase,
                    "best_score": None,
                    "best_box": None,
                    "best_box_iou": None,
                    "reaches_threshold": False,
                    "reading": "no image patch matched this phrase best",
                }
            )
            continue
        top = max(own, key=lambda det: det["score"])
        iou = box_iou(top["box"], box)
        reaches = top["score"] >= threshold
        if iou < LOCALISATION_IOU:
            reading = f"not localised: the best-scoring box overlaps the reference at IoU {iou:.2f}"
        elif reaches:
            reading = "found: the best-scoring box reaches the threshold and covers the reference"
        else:
            reading = "localised but scored under the threshold: a score miss, not a coordinate error"
        rows.append(
            {
                "reference": phrase,
                "best_score": top["score"],
                "best_box": top["box"],
                "best_box_iou": iou,
                "reaches_threshold": reaches,
                "reading": reading,
            }
        )
    return rows


def _model_weight_digest(root: Path) -> str | None:
    """The manifest-recorded SHA-256 of the verified weight file (printed by the notebook, review OVT-m4)."""
    with open(root / MANIFEST_NAME, encoding="utf-8") as fh:
        manifest = json.load(fh)
    for entry in manifest["files"]:
        if entry["path"].endswith(".safetensors"):
            return entry.get("sha256")
    return None


@dataclass
class OwlViTDetectionPipeline:
    """Text-prompted (open-vocabulary) object detection over the pinned OWL-ViT base/32 checkpoint."""

    _runner: Callable[[Image.Image, list[str], float], list[dict[str, Any]]]
    device: str
    # CLIP token count of one query (start and end tokens included) from the loaded tokenizer; detect and
    # validate_inputs refuse a query over MAX_TEXT_TOKENS with it. None for an injected runner.
    count_tokens: Callable[[str], int] | None = None
    weight_sha256: str | None = None

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> OwlViTDetectionPipeline:
        _require_pinned("load the model")
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if not (root / MANIFEST_NAME).is_file():
            raise FileNotFoundError(
                f"no snapshot manifest at {root}; stage {MODEL_ID}@{MODEL_REVISION} "
                f"under weights/{MODEL_KEY} "
                "(allow_download=True fetches the manifest-listed files)"
            )
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        common = {"trust_remote_code": False, "local_files_only": True}
        processor = OwlViTProcessor.from_pretrained(str(root), **common)
        model = OwlViTForObjectDetection.from_pretrained(str(root), **common)
        pipe = cls.from_components(model, processor, resolved_device)
        pipe.weight_sha256 = _model_weight_digest(root)
        return pipe

    @classmethod
    def from_components(cls, model: Any, processor: Any, device: str) -> OwlViTDetectionPipeline:
        """Wrap an already-loaded OWL-ViT model and processor (the verified snapshot, or a test double)."""
        import torch

        model = model.to(device).eval()

        def runner(image: Image.Image, queries: list[str], threshold: float) -> list[dict]:
            # One text query per phrase, padded to the longest; detect has already refused any query over
            # MAX_TEXT_TOKENS, which the processor would not truncate.
            inputs = processor(images=image, text=[queries], return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            # OWL-ViT resizes without padding, so boxes (normalised to the resized square) are scaled by the
            # original (height, width) directly.
            result = processor.post_process_grounded_object_detection(
                outputs, threshold=threshold, target_sizes=[image.size[::-1]], text_labels=[queries]
            )[0]
            return [
                {"box": [float(v) for v in box.tolist()], "label": str(label), "score": float(score)}
                for box, label, score in zip(
                    result["boxes"], result["text_labels"], result["scores"], strict=True
                )
            ]

        tokenizer = getattr(processor, "tokenizer", None)
        count_tokens = None
        if tokenizer is not None:

            def count_tokens(query: str) -> int:
                return len(tokenizer(query, verbose=False)["input_ids"])

        return cls(runner, device, count_tokens)

    def detect(
        self,
        image: Image.Image,
        prompts: Sequence[str],
        *,
        threshold: float = DETECTION_THRESHOLD,
    ) -> dict[str, Any]:
        """Detect the phrases in `prompts`; boxes are xyxy pixel coordinates in the input image."""
        rgb, queries, checked = _check_inputs(image, prompts, threshold, self.count_tokens)
        detections = self._runner(rgb, queries, checked)
        if len(detections) > MAX_DETECTIONS:
            raise RuntimeError(
                f"backend returned {len(detections)} detections > MAX_DETECTIONS {MAX_DETECTIONS}"
            )
        for det in detections:
            if set(det) != {"box", "label", "score"} or len(det["box"]) != 4 or det["label"] not in queries:
                raise RuntimeError(f"backend returned a malformed detection: {det!r}")
        return {
            "detections": sorted(detections, key=lambda d: -d["score"]),
            "queries": queries,
            "threshold": checked,
            "width": rgb.width,
            "height": rgb.height,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
