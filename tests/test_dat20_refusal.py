"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import io
import json
import os
import re
import signal
import subprocess
import sys
import types
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from PIL import Image

from owlvit_detection_pipeline import pipeline as pipeline_module

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/owlvit_detection_colab.ipynb"
BYOD_CELL = "owlvit_detection-12"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


STEM = "owlvit_detection"
EARLIER = {"image": "old", "input_manifest": "old", "result": "old", "report": "old", "payload": "old", "annotated": "old"}


def _seed_outputs(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    out.mkdir()
    for name in ("input_manifest.json", "evaluation_report.json", "result.json", "annotated.png", "detections.csv"):
        (out / f"{STEM}_{name}").write_text("x", encoding="utf-8")


def _namespace() -> dict:
    ns = {k: v for k, v in vars(pipeline_module).items() if not k.startswith("__")}
    ns.update(EARLIER, pipe=types.SimpleNamespace(count_tokens=lambda text: len(text.split())))
    return ns


def _section4(monkeypatch, tmp_path, path: str, *, prompts: str | None = None, source: str | None = None) -> dict:
    monkeypatch.chdir(tmp_path)
    _seed_outputs(tmp_path)
    source = (source or _cell(BYOD_CELL)).replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_IMAGE_PATH = ''", f"BYOD_IMAGE_PATH = {path!r}", 1)
    if prompts is not None:
        source = source.replace("BYOD_PROMPTS = 'a photo of a cat, a photo of a remote control'", f"BYOD_PROMPTS = {prompts!r}", 1)
    ns = _namespace()
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def _cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "tiny.png")
    Image.new("RGB", (64, 48)).save(tmp_path / "ok.png")
    return {
        "text file (the scripted /etc/hostname run)": ("hostname", None, "UnidentifiedImageError"),
        "folder": ("folder", None, "does not exist or is not a file"),
        "missing": ("nope.jpg", None, "does not exist or is not a file"),
        "too small": ("tiny.png", None, "ValueError"),
        "no prompts": ("ok.png", " , ", "ValueError"),
    }


@pytest.mark.parametrize("case", ["text file (the scripted /etc/hostname run)", "folder", "missing", "too small", "no prompts"])
def test_byod_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    file_name, prompts, rule = _cases(tmp_path)[case]
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / file_name), prompts=prompts)
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line
    assert "run this cell again" in line and "earlier results of this session and their outputs/ files were removed" in line
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == []


def test_refusal_clears_the_session(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    monkeypatch.chdir(tmp_path)
    source = _cell(BYOD_CELL).replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_IMAGE_PATH = ''", "BYOD_IMAGE_PATH = 'nope.jpg'", 1)
    ns = _namespace()
    with pytest.raises(ValueError, match="^nope.jpg: BYOD_IMAGE_PATH"):
        exec(compile(source, "<section 4>", "exec"), ns)
    assert not set(EARLIER) & set(ns), "an earlier run's results survived the refusal"
    assert "pipe" in ns


@pytest.mark.parametrize(("uploaded", "file_name", "rule"), [({}, "(no file)", "received 0 files"), ({"a.png": b"1", "b.png": b"2"}, "a.png, b.png", "received 2 files")])
def test_cancelled_or_multiple_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name, rule):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "")
    assert _refusal_line(caught.value).startswith(f"Refused: {file_name}: Upload exactly one image file") and rule in str(caught.value)


def test_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    Image.new("RGB", (64, 48)).save(tmp_path / "ok.png")
    source = _cell(BYOD_CELL).replace("        validate_inputs(image, prompts,", "        raise RuntimeError('internal bug')\n        validate_inputs(image, prompts,", 1)
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / "ok.png"), source=source)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / f"{STEM}_result.json").exists()  # nothing is removed for an internal error


def test_good_image_still_loads(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    Image.new("RGB", (64, 48)).save(tmp_path / "ok.png")
    ns = _section4(monkeypatch, tmp_path, str(tmp_path / "ok.png"))
    assert ns["image_name"] == "ok.png" and ns["sample_kind"] == "BYOD" and ns["prompts"] == ["a photo of a cat", "a photo of a remote control"]


def test_default_path_does_not_enter_the_refusal_wrap():
    source = _cell(BYOD_CELL)
    wrap = source[source.index("if USE_BYOD:\n") : source.index("else:\n    # Deterministic synthetic scene")]
    assert "Image.new('RGB', (640, 480)" not in wrap and "refuse_byod(" in wrap


@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs a POSIX /etc/hostname, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_hostname(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD=True;BYOD_IMAGE_PATH='/etc/hostname': a text file present in every fresh runtime."""
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "/etc/hostname")
    assert _refusal_line(caught.value).startswith("Refused: hostname: not an image file Pillow can open (UnidentifiedImageError: ")


# --- the catch is narrow: only bad input is refused; an internal TypeError/OSError keeps its traceback -------------


def test_the_byod_catch_is_narrow_in_the_notebook_source():
    source = _cell(BYOD_CELL)
    branch = source[source.index("if USE_BYOD:\n") : source.index("else:\n    # Deterministic synthetic scene")]
    caught = re.findall(r"^\s*except (.+?)(?: as \w+)?:$", branch, flags=re.M)
    assert caught == ["OSError", "ImportError", "OSError", "BYODRefusal", "ValueError"], caught
    assert "            try:\n                image = Image.open(BYOD_IMAGE_PATH)\n                image.load()\n            except OSError as exc:\n" in branch
    assert "            try:\n                image = Image.open(io.BytesIO(uploaded[image_name]))\n                image.load()\n            except OSError as exc:\n" in branch


@pytest.mark.parametrize("upload", [False, True], ids=["path", "upload"])
def test_a_truncated_image_is_still_one_refused_line(monkeypatch, tmp_path, upload):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 30)).save(buffer, format="PNG")
    data = buffer.getvalue()[:-40]
    (tmp_path / "cut.png").write_bytes(data)
    if upload:
        colab = types.ModuleType("google.colab")
        colab.files = types.SimpleNamespace(upload=lambda: {"cut.png": data})
        monkeypatch.setitem(sys.modules, "google.colab", colab)
    else:
        monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "" if upload else str(tmp_path / "cut.png"))
    line = _refusal_line(caught.value)
    assert line.startswith("Refused: cut.png: not an image file Pillow can open ("), line
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == []


@pytest.mark.parametrize("error", [TypeError("internal helper bug"), OSError("internal I/O bug")], ids=["TypeError", "OSError"])
def test_an_internal_error_in_the_byod_branch_keeps_its_traceback(monkeypatch, tmp_path, error):
    import traceback

    monkeypatch.setitem(sys.modules, "google.colab", None)
    Image.new("RGB", (64, 48)).save(tmp_path / "ok.png")
    monkeypatch.chdir(tmp_path)
    _seed_outputs(tmp_path)
    source = _cell(BYOD_CELL).replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_IMAGE_PATH = ''", f"BYOD_IMAGE_PATH = {str(tmp_path / 'ok.png')!r}", 1)
    ns = _namespace()

    def broken_count_tokens(text):
        raise error

    ns["pipe"] = types.SimpleNamespace(count_tokens=broken_count_tokens)
    with pytest.raises(type(error)) as caught:
        exec(compile(source, "<section 4>", "exec"), ns)
    assert caught.value is error and not getattr(caught.value, "dimer_refusal", False)
    rendered = "".join(traceback.format_exception(caught.value))
    assert rendered.startswith("Traceback") and "broken_count_tokens" in rendered and "Refused:" not in rendered
    assert (tmp_path / "outputs" / f"{STEM}_result.json").exists() and ns["result"] == "old"  # nothing is cleared
