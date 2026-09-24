"""The real OWL-ViT runner on a tiny random-weight model and a toy CLIP vocabulary: no checkpoint, no network.

It exercises processor → model → post_process_grounded_object_detection → detect, including the
no-padding box scaling. It says nothing about detection quality.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from PIL import Image  # noqa: E402

from owlvit_detection_pipeline import MAX_DETECTIONS, OwlViTDetectionPipeline  # noqa: E402


def _tiny(tmp_path) -> OwlViTDetectionPipeline:
    from transformers import (
        CLIPTokenizer,
        OwlViTConfig,
        OwlViTForObjectDetection,
        OwlViTImageProcessor,
        OwlViTProcessor,
    )

    # OWL-ViT treats a query as present when its first token id is > 0, as the real CLIP vocabulary's
    # <|startoftext|> (49406) is; the toy vocabulary keeps that property.
    vocab = {"!": 0, "<|startoftext|>": 1, "<|endoftext|>": 2}
    for token in [*"abcdefghijklmnopqrstuvwxyz", *(c + "</w>" for c in "abcdefghijklmnopqrstuvwxyz")]:
        vocab.setdefault(token, len(vocab))
    (tmp_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (tmp_path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = CLIPTokenizer(
        str(tmp_path / "vocab.json"), str(tmp_path / "merges.txt"), pad_token="!", model_max_length=16
    )
    image_processor = OwlViTImageProcessor(
        size={"height": 64, "width": 64}, crop_size={"height": 64, "width": 64}, do_center_crop=False
    )
    torch.manual_seed(0)
    text = {"hidden_size": 16, "intermediate_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2}
    config = OwlViTConfig(
        text_config={
            **text,
            "vocab_size": len(vocab),
            "max_position_embeddings": 16,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
        vision_config={**text, "image_size": 64, "patch_size": 32},
        projection_dim=16,
    )
    model = OwlViTForObjectDetection(config)
    processor = OwlViTProcessor(image_processor=image_processor, tokenizer=tokenizer)
    return OwlViTDetectionPipeline.from_components(model, processor, "cpu")


def test_runner_returns_one_box_per_patch_labelled_with_a_query(tmp_path):
    pipe = _tiny(tmp_path)
    result = pipe.detect(Image.new("RGB", (120, 80), "white"), ["stop sign", "clock"], threshold=0.0)
    detections = result["detections"]
    assert len(detections) == 4 <= MAX_DETECTIONS  # a 64x64 input at patch 32 has 2x2 patches
    assert {d["label"] for d in detections} <= {"stop sign", "clock"}
    assert [d["score"] for d in detections] == sorted((d["score"] for d in detections), reverse=True)
    assert all(len(d["box"]) == 4 and d["box"][0] <= d["box"][2] for d in detections)


def test_runner_threshold_filters_and_is_repeatable(tmp_path):
    pipe = _tiny(tmp_path)
    image = Image.new("RGB", (64, 96), "white")
    everything = pipe.detect(image, ["clock"], threshold=0.0)["detections"]
    again = pipe.detect(image, ["clock"], threshold=0.0)["detections"]
    assert everything == again
    cut = sorted(d["score"] for d in everything)[len(everything) // 2]
    kept = pipe.detect(image, ["clock"], threshold=cut)["detections"]
    assert (
        kept
        and all(d["score"] > cut or abs(d["score"] - cut) < 1e-9 for d in kept)
        and len(kept) <= len(everything)
    )
