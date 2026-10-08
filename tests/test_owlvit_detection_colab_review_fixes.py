"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings on owlvit_detection_colab (OVT-*).

They need only CI's dependencies. The real-tokenizer check runs only where the pinned snapshot is staged
(`weights/owlvit-base-patch32`, or OWLVIT_WEIGHTS_DIR); the token rule itself is exercised on CI through the
tiny random-weight model's real CLIPTokenizer and through fake token counters.
"""
# ruff: noqa: E501  -- notebook source fragments are kept on single lines

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from PIL import Image

import owlvit_detection_pipeline as pkg
from owlvit_detection_pipeline import (
    LOCALISATION_IOU,
    MAX_TEXT_TOKENS,
    OwlViTDetectionPipeline,
    best_box_per_reference,
    check_prompt_tokens,
    evaluation_report,
    validate_inputs,
)
from owlvit_detection_pipeline import pipeline as pipeline_module

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "owlvit_detection_colab.ipynb"
SODA = "a photo of a red-and-white 330-ml soda can"  # 42 characters, 18 CLIP tokens (review probe P3)
CARABAO = "a photo of a carabao (bubalus bubalis) calf"  # 43 characters, 17 CLIP tokens (review probe P5)
SCENE = {
    "a black rectangle": [80.0, 120.0, 280.0, 360.0],
    "a red circle": [380.0, 140.0, 560.0, 320.0],
    "a blue triangle": [200.0, 380.0, 360.0, 460.0],
}


def _fake_counter(query: str) -> int:
    """Stand-in token counts: the review's measured counts for the two long phrases, 5 otherwise."""
    return {SODA: 18, CARABAO: 17}.get(query, 5)


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _code(nb: dict, marker: str) -> str:
    cells = [_src(c) for c in nb["cells"] if c["cell_type"] == "code" and marker in _src(c)]
    assert len(cells) == 1, marker
    return cells[0]


def _package_namespace() -> dict:
    return {name: getattr(pkg, name) for name in pkg.__all__}


# ---- OVT-M2: the 16-token CLIP limit is checked before the model runs, never "silently truncated" ----


def test_ovt_m2_check_prompt_tokens_names_phrase_count_and_limit() -> None:
    assert check_prompt_tokens(["a red circle"], _fake_counter) == [5]
    with pytest.raises(ValueError) as err:
        check_prompt_tokens(["a cat", SODA], _fake_counter)
    message = str(err.value)
    assert SODA in message and "18 CLIP tokens" in message and f"MAX_TEXT_TOKENS {MAX_TEXT_TOKENS}" in message


def test_ovt_m2_validate_inputs_refuses_long_phrase_before_any_model_import(forbid_model_imports) -> None:
    image = Image.new("RGB", (64, 64))
    with pytest.raises(ValueError, match="18 CLIP tokens"):
        validate_inputs(image, [SODA], count_tokens=_fake_counter)
    manifest = validate_inputs(image, ["a red circle"], count_tokens=_fake_counter)
    assert manifest["prompt_tokens"] == [5] and manifest["verdict"] == "accepted"
    # Without a counter the manifest says the token rule was not checked rather than implying it passed.
    assert validate_inputs(image, ["a red circle"])["prompt_tokens"] is None


def test_ovt_m2_detect_refuses_before_runner() -> None:
    calls = []
    pipe = OwlViTDetectionPipeline(lambda *a: calls.append(a) or [], "cpu", _fake_counter)
    with pytest.raises(ValueError, match="MAX_TEXT_TOKENS 16"):
        pipe.detect(Image.new("RGB", (64, 64)), [CARABAO, "a cat"])
    assert calls == []


def test_ovt_m2_tiny_model_real_tokenizer_refuses_instead_of_tensor_error(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from test_tiny_model import _tiny

    pipe = _tiny(tmp_path)
    assert pipe.count_tokens is not None
    # The toy vocabulary has no merges: one token per letter, plus start and end tokens.
    assert pipe.count_tokens("clock") == 7
    image = Image.new("RGB", (64, 64), "white")
    with pytest.raises(ValueError, match=r"is 17 CLIP tokens .* > MAX_TEXT_TOKENS 16"):
        pipe.detect(image, ["abcdefghijklmno"], threshold=0.0)  # before the fix: a tensor-shape error inside the model
    assert pipe.detect(image, ["abcdefghijklmn"], threshold=0.0)["queries"] == ["abcdefghijklmn"]  # exactly 16


def _snapshot_dir() -> Path:
    return Path(os.environ.get("OWLVIT_WEIGHTS_DIR", pipeline_module.DEFAULT_WEIGHTS_DIR))


@pytest.mark.skipif(not (_snapshot_dir() / "vocab.json").is_file(), reason="pinned snapshot tokenizer not staged")
def test_ovt_m2_real_clip_tokenizer_counts() -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.CLIPTokenizerFast.from_pretrained(str(_snapshot_dir()), local_files_only=True)

    def count(query: str) -> int:
        return len(tokenizer(query, verbose=False)["input_ids"])

    assert count(SODA) == 18 and count(CARABAO) == 17
    assert all(count(p) <= MAX_TEXT_TOKENS for p in SCENE)


def test_ovt_m2_no_truncation_claim_anywhere(nb: dict) -> None:
    text = _markdown(nb) + (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8") + (ROOT / "src/owlvit_detection_pipeline/pipeline.py").read_text(encoding="utf-8")
    assert not re.search(r"silently truncated|truncated silently|is truncated;|pads/truncates", text)


def test_ovt_m2_section5_refuses_byod_phrase_and_records_token_probe(nb: dict, tmp_path, monkeypatch) -> None:
    code = _code(nb, "input_manifest = validate_inputs(")
    assert "count_tokens=pipe.count_tokens" in code and SODA in code
    monkeypatch.chdir(tmp_path)

    class Pipe:
        count_tokens = staticmethod(_fake_counter)

    ns = {**_package_namespace(), "image": Image.new("RGB", (640, 480)), "prompts": list(SCENE), "threshold": 0.1, "image_name": "scene.png", "pipe": Pipe()}
    exec(compile(code, "<section 5>", "exec"), ns)
    findings = {f["input"]: f["message"] for f in ns["input_manifest"]["findings"]}
    assert set(findings) == {"over-long-prompt-probe", "over-16-token-prompt-probe"}
    assert "MAX_PROMPT_CHARS 48" in findings["over-long-prompt-probe"]
    assert "18 CLIP tokens" in findings["over-16-token-prompt-probe"]
    # Review acceptance check: the BYOD phrase stops in Section 5 with a message naming it and the limit.
    ns["prompts"] = [SODA]
    with pytest.raises(ValueError, match=re.escape(SODA) + r".*MAX_TEXT_TOKENS 16"):
        exec(compile(code, "<section 5>", "exec"), ns)


# ---- OVT-M3: tell a score miss from a localisation miss; no duplicate prediction the scene cannot show ----


def _cands(*rows: tuple[str, float, list[float]]) -> dict:
    return {"detections": [{"label": label, "score": score, "box": box} for label, score, box in rows]}


def test_ovt_m3_best_box_per_reference_readings() -> None:
    cands = _cands(
        ("a red circle", 0.4487, [378.1, 137.5, 560.0, 321.7]),
        ("a black rectangle", 0.0373, [72.9, 112.6, 281.9, 361.3]),
        ("a black rectangle", 0.01, [0.0, 0.0, 10.0, 10.0]),
        ("a blue triangle", 0.2, [0.0, 0.0, 50.0, 50.0]),
    )
    rows = {r["reference"]: r for r in best_box_per_reference(cands, {**SCENE, "a green star": [0, 0, 1, 1]}, 0.1)}
    assert rows["a red circle"]["reaches_threshold"] and rows["a red circle"]["reading"].startswith("found")
    rect = rows["a black rectangle"]
    assert rect["best_score"] == 0.0373 and rect["best_box_iou"] >= LOCALISATION_IOU and not rect["reaches_threshold"]
    assert "score miss" in rect["reading"]
    assert rows["a blue triangle"]["reading"].startswith("not localised")
    assert rows["a green star"]["best_score"] is None


def test_ovt_m3_and_m6_report_on_the_recorded_default_result() -> None:
    result = {"threshold": 0.1, "detections": [{"label": "a red circle", "score": 0.4487, "box": [378.1, 137.5, 560.0, 321.7]}]}
    cands = _cands(("a red circle", 0.4487, [378.1, 137.5, 560.0, 321.7]), ("a blue triangle", 0.0985, [188.7, 380.7, 371.6, 460.4]), ("a black rectangle", 0.0373, [72.9, 112.6, 281.9, 361.3]))
    report = evaluation_report(result, SCENE, candidates=cands)
    metrics = {m["reference"]: m for m in report["metrics"]}
    # OVT-m6: zero overlap is not "matched" to the circle any more.
    for missed in ("a black rectangle", "a blue triangle"):
        assert metrics[missed]["matched_label"] is None and metrics[missed]["value"] == 0.0
        assert metrics[missed]["note"] == "no overlapping detection at this threshold"
    assert metrics["a red circle"]["matched_label"] == "a red circle" and "note" not in metrics["a red circle"]
    readings = {r["reference"]: r["reading"] for r in report["per_reference_best_box"]}
    assert "score miss" in readings["a black rectangle"] and "score miss" in readings["a blue triangle"]
    assert "per_reference_best_box" not in evaluation_report(result, SCENE)


def test_ovt_m3_notebook_explains_zero_iou_and_drops_duplicate_prediction(nb: dict) -> None:
    md = _markdown(nb)
    assert "**0.0 means no box above the threshold touched it**" in md
    assert "score miss" in md and "IoU 0.9235" in md and "IoU 0.8641" in md
    assert "count the duplicate boxes" not in md and "and expect duplicate boxes" not in md
    code = _code(nb, "report = evaluation_report(")
    assert "candidates = pipe.detect(image, prompts, threshold=0.0) if drawn_boxes else None" in code
    assert "for row in report.get('per_reference_best_box', []):" in code


def test_ovt_m3_section7_prints_best_box_per_reference(nb: dict, tmp_path, monkeypatch, capsys) -> None:
    code = _code(nb, "report = evaluation_report(")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    result = {"threshold": 0.1, "detections": [{"label": "a red circle", "score": 0.4487, "box": [378.1, 137.5, 560.0, 321.7]}]}

    class Pipe:
        def detect(self, image, prompts, threshold):
            assert threshold == 0.0
            return _cands(("a red circle", 0.4487, [378.1, 137.5, 560.0, 321.7]), ("a blue triangle", 0.0985, [188.7, 380.7, 371.6, 460.4]))

    ns = {**_package_namespace(), "json": json, "result": result, "drawn_boxes": SCENE, "sample_kind": "synthetic", "image": None, "prompts": list(SCENE), "threshold": 0.1, "pipe": Pipe()}
    exec(compile(code, "<section 7>", "exec"), ns)
    out = capsys.readouterr().out
    assert "'a blue triangle': best score 0.0985" in out and "score miss" in out
    assert "'a black rectangle': best score none" in out
    saved = json.loads((tmp_path / "outputs" / "owlvit_detection_evaluation_report.json").read_text(encoding="utf-8"))
    assert len(saved["per_reference_best_box"]) == 3


# ---- OVT-m1, OVT-m4, OVT-m5 ----


def test_ovt_m1_no_stale_status_text(nb: dict) -> None:
    text = "\n".join(_src(c) for c in nb["cells"])
    assert "not measured in this revision" not in text and "No run with the pinned weights" not in text
    assert "314.8 s" in text and "0.53 s" in text


def test_ovt_m4_section3_prints_real_weight_source(nb: dict, tmp_path) -> None:
    code = _code(nb, "pipe = OwlViTDetectionPipeline.from_pretrained(")
    assert "'local-snapshot'" not in code and "'weights_dir': str(WEIGHTS_DIR)" in code and "'weight_sha256'" in code
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps({"files": [{"path": "config.json", "sha256": "a"}, {"path": "model.safetensors", "sha256": "b" * 64}]}), encoding="utf-8")
    assert pipeline_module._model_weight_digest(tmp_path) == "b" * 64


def test_ovt_m5_declares_notebook_spec_2_2(nb: dict) -> None:
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.2"
