# owlvit_detection_colab.ipynb — Notebook Review (Framework v1)

**Readiness: Needs revision.** The notebook is a short `TASK-INFERENCE` lesson: zero-shot, text-prompted detection
on a drawn scene with an input manifest, a `sample-sanity` IoU report and JSON/CSV/PNG exports. The reviewed blob is
the exact blob recorded as passing on Kaggle T4, and a CPU run in this review reproduced that run's single detection
to four decimals. Three problems hold it back:

- Run all needs a manual restart after the install cell (OVT-M1).
- The phrase check accepts realistic BYOD phrases that the model then cannot tokenize. Detection crashes with a raw
  tensor-shape error, although the notebook says over-long phrases are "silently truncated" (OVT-M2).
- The default sanity demonstration reports two of its three drawn shapes as `box_iou` 0.0 "matched" to the red
  circle. Nothing tells the learner that both shapes were in fact localised well (IoU 0.92 and 0.86) and only scored
  under the 0.1 threshold (OVT-M3).

The fine-tuning defects of the sibling owlv2 notebook (row 59: OWD-M2 in-place adaptation, OWD-M3 ranking asserts)
**do not apply**. This notebook performs no adaptation and contains no `assert`. Its over-prediction note (OWD-S1)
is inverted here: the problem is under-detection at the default threshold, not duplicate boxes.

Findings: 0 Blocker, 3 Major, 6 Minor, 4 Suggestion. Prefix `OVT`.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Repository | `kurtvalcorza/owlvit-detection-pipeline` |
| Notebook | `tutorials/owlvit_detection_colab.ipynb` (19 cells: 8 code, 11 markdown; no outputs saved) |
| Reviewed revision | `origin/main` = `48731001c6e628e2e5194addef125a6891c4b73d` (GitHub API `commits/main`, 2026-10-04) |
| Notebook blob | `fff9ff981ddcb6a5f6b6f02588281491ce99e8fb`, identical to the blob recorded at `14be73f` in `docs/release-verification.md` |
| Spec baseline | NOTEBOOK_SPEC **2.2** (ml-worker `origin/main` `b1cfe13`). The notebook declares spec `2.1` in `metadata.dimer` |
| Profile / mode | `TASK-INFERENCE` / `GUIDED`, standalone carrier (`pipeline.py` carried verbatim apart from the one `DEFAULT_WEIGHTS_DIR` rewrite; generator `tools/build_notebook.py` + `tools/notebook_template.py`; `--check` passes) |
| Audience / prerequisites | Basic Python and PIL; xyxy boxes, IoU, and why a sigmoid score is not a probability (cell 1) |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU default, CUDA automatic (cell 1) |
| Promised outcomes | Pinned install; digest-verified 8-file snapshot at `cbc355f`; drawn 640×480 scene with three reference boxes; input manifest with an over-long-phrase rejection; text-prompted detection with a caller-owned threshold; correct reading of sigmoid scores and the lack of NMS; `sample-sanity` (or `not-measurable`) evaluation report; five exports with provenance; optional BYOD (one image through `BYOD_IMAGE_PATH` or a Colab upload, plus own phrases) |
| Status | Candidate (README, STATUS, release-verification): default path recorded, REL12 BYOD pending |
| Open PRs | None. PR #1 (E2E carrier) is merged |

### Evidence actually obtained

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Precise contract and score semantics (uncalibrated sigmoid, caller-owned threshold, no NMS, 768×768 squash) and honest limits. Stale "no run recorded / runtimes not measured" text (OVT-m1). No guidance for what the learner actually sees on the default scene, which is two misses (OVT-M3). Guided layer partial (OVT-m5) |
| Clean default | Documented execution evidence + direct execution | Kaggle Tesla T4, 2026-09-25, exact blob `fff9ff98`: PASSED 314.8 s, 8/8 post-restart code cells, **pass 1 stopped after the install cell** with `Core dependencies changed while older modules were loaded` (cuda-bindings, numpy) and the kernel was restarted (OVT-M1). Direct CPU run of the notebook's own cells 3–17 (cell 7 replaced, see limitations): 1 detection, `a red circle` 0.4487, box [378.1, 137.5, 560.0, 321.7], identical to the T4 record; `box_iou` 0.9671 / 0.0 / 0.0; five outputs written. No Colab record (OVT-m3) |
| Active learning | Direct execution, CPU | Threshold rerun from cell 9 (0.05) reaches detection, report and CSV (2 boxes, `report.threshold` 0.05, 2 CSV rows): the control works and outputs are not stale. The documented experiments produce: 0.05 → 2 boxes, 0.02 → 3, **no duplicates** at either; `a photo of …` phrasing → 2 boxes at 0.1 (circle 0.5226, triangle 0.1278); adding `a green star` → nothing new. At threshold 0 the top-scoring box per phrase has IoU 0.9235 (rectangle, score 0.0373), 0.9671 (circle, 0.4487), 0.8641 (triangle, 0.0985) |
| Reuse and recovery | Direct execution, CPU, stand-in images (labelled); Colab upload widget not exercised | BYOD through `BYOD_IMAGE_PATH` with an 800×600 RGBA PNG stand-in reaches detection, a `not-measurable` report and all five exports with `sample.kind` `BYOD`. Rejections: 17 phrases, 0 phrases, duplicates after normalisation, and a 10 px image are refused in cell 11 with messages that name the failed ceiling. Missing path → raw `FileNotFoundError`; non-image → `UnidentifiedImageError` (both name the file). Empty location field outside Colab → `ModuleNotFoundError: google.colab` (OVT-m2). **Phrases of 42–43 characters with 17–18 CLIP tokens are accepted by validation and then crash `detect`** (OVT-M2) |

Limitations: no GPU and no Colab. The local env differs from the pins: torch 2.13.0+cpu against 2.14.0, and pillow
12.3.0 against 11.3.0. transformers is 4.57.6, as pinned, so the tokenizer behaviour behind OVT-M2 matches the
pin. CUDA_VISIBLE_DEVICES=-1 and 4 threads were used. Cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, so no pip
install ran. Cell 7 (manifest write plus Hub staging) was replaced by `verify_snapshot` + `from_pretrained` on the
local clone's pre-staged `weights/owlvit-base-patch32`, and all 8 files re-hashed OK. No photograph was used for BYOD.
Probe P0 confirmed that the carried cell equals `pipeline.py` except for the one documented rewrite line.

## 2. Separate judgments

- **Technical correctness:** the default path is sound. The snapshot is verified before load, with no remote code
  and `local_files_only`. Validation and detection share `_check_inputs`. Outputs carry model, revision, notebook
  source and runtime. CPU output matches the T4 record. Defects:
  - The restart-forcing install (OVT-M1).
  - A validation gap: the 16-token CLIP limit is not checked, and the claimed truncation does not happen (OVT-M2).
  - `matched_label` is reported for zero-overlap references (OVT-m6).
  - A hard-coded weight "source" (OVT-m4).
- **Promise fulfilment:** the default promises are met, except that "Run all … no configuration edit" holds only
  after a restart (OVT-M1). These promises fail:
  - "a long phrase is silently truncated" (cells 0 and 18) is false for the pinned transformers (OVT-M2).
  - BYOD "passes through the same notebook-local validation" does not hold. Validation accepts inputs that detection
    rejects (OVT-M2).
  - "read … the absence of non-maximum suppression correctly" cannot be practised on the default scene, which yields
    no duplicate at 0.1, 0.05 or 0.02 (OVT-M3).
- **Learner experience:** the scientific framing is careful and honest about what a drawn scene proves. But the one
  observable result, a sanity report with two zeros, comes with no expected-result note. The text even says no run
  exists yet (OVT-m1). The experiments are one sentence with no rerun range (OVT-S1).
- **Spec conformance:** these applicable MUSTs are unresolved:
  - RUN1, RUN10 and ENV6 (restart).
  - DAT19 (BYOD failure inside model execution) and DAT12 (the stated limit is characters, but the binding limit is
    tokens).
  - REL12 (BYOD verification pending). The status is honestly Candidate, so this is a gate, not a mislabel.

  The 2.2 guided layer (GDL, SHOULD) is partial. UNC1–UNC4 and §21.8 (sigmoid not probability, prompt dependence,
  unsupported tasks) are met.

## 3. Findings

### OVT-M1 — Major: Run all needs a manual restart after the in-kernel install

- **Cell/section:** cell 3 (Section 1), generated by `tools/build_notebook.py:47–70` (`_INSTALL_GUARD`). PINS come from `pyproject.toml`.
- **Observed issue:** `pip install` of `torch==2.14.0`, `numpy==2.5.3`, `pillow==11.3.0` and the other pins runs
  into the live kernel. When a pre-imported distribution changes, the guard raises *"Restart the runtime, then rerun
  from the top"*. Cell 2 announces this restart instruction.
- **Consequence:** a learner who presses Run all on a hosted runtime stops at cell 3 and must restart by hand. That
  breaks the one-pass `Run all` contract that cell 0 promises.
- **Evidence:** documented execution evidence. In `docs/release-verification.md` (Recorded executions, row 1),
  pass 1 (252.7 s) stopped after the install cell with `RuntimeError: Core dependencies changed while older modules
  were loaded (cuda-bindings 12.9.4 → 13.4.3, numpy 2.0.2 → 2.5.3)`, and the kernel was restarted. The blob is the
  same as the reviewed one. Colab: **not verified**. Colab also preloads NumPy, so the same mechanism is likely.
- **Recommended correction:** replace the in-kernel install with the fleet's **uv isolated-environment pattern**.
  A carrier cell bootstraps uv, runs `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked
  `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in that env,
  so the kernel's preloaded NumPy/torch are never replaced. The reference is
  `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` on origin/main.
  Make the change in the generator (`tools/build_notebook.py` install block) and update the Section 1 text.
- **Acceptance check:** a fresh Colab T4 (and Kaggle) Run all of the new blob completes every code cell in one pass
  with no restart and no error output. The record in `docs/release-verification.md` says "no restart".

### OVT-M2 — Major: validation accepts BYOD phrases over the 16-token CLIP limit, then `detect` crashes, though the notebook says they are "silently truncated"

- **Cell/section:**
  - Cell 0 says "Prompts are free text limited to 16 CLIP tokens each, so a long phrase is silently truncated"
    (`tools/notebook_template.py:71`).
  - Cell 18 says "Prompts longer than 16 CLIP tokens are truncated silently" (`:275`).
  - Cell 1 states the BYOD limit as "at most 48 characters each".
  - Cell 11 runs `validate_inputs`, and cell 13 runs `detect`.
  - Source: `format_prompts` checks only `MAX_PROMPT_CHARS` (`pipeline.py:160–182`). The runner calls
    `processor(images=…, text=[queries])` with no truncation (`pipeline.py:376–378`). The code comment at
    `pipeline.py:41`/`:376` and `MODEL_CARD.md:69` also claim truncation.
- **Observed issue:** CLIP BPE splits hyphens, digits and parentheses into many tokens, so realistic phrases within
  48 characters exceed 16 tokens. Validation writes `"verdict": "accepted"` to the input manifest, and `detect` then
  fails inside the model:
  - `a photo of a red-and-white 330-ml soda can` (42 chars, 18 tokens) fails with `RuntimeError: The size of tensor
    a (18) must match the size of tensor b (16)`.
  - `a photo of a carabao (bubalus bubalis) calf` (43 chars, 17 tokens) next to a short phrase fails with
    `ValueError: Unable to create tensor, you should probably activate truncation…`.
- **Consequence:** a learner's own phrases can pass the stated contract and the validation stage, then crash
  detection with an error that names no phrase and no limit. The notebook teaches the opposite behaviour. This breaks
  DAT19 (failure must be caught before model execution and name the contract) and makes the DAT12 limit wrong.
- **Evidence:** direct execution, CPU, transformers 4.57.6 = pin (probes P3 `long_prompt`, P5). A 44-char,
  15-token phrase runs. Source inspection for the claims.
- **Recommended correction:** in `format_prompts`/`_check_inputs`, count tokens with the snapshot's tokenizer (or a
  carried CLIP BPE count) and reject any phrase over `MAX_TEXT_TOKENS - 2` content tokens with a message naming the
  phrase, its token count and the limit. Alternatively, pass `truncation=True, max_length=MAX_TEXT_TOKENS,
  padding="max_length"` and record the truncation as a manifest finding. Make the notebook text, `MODEL_CARD.md` and
  the code comments match the chosen behaviour, and add a unit test with a ≤48-char, >16-token phrase.
- **Acceptance check:** with `BYOD_PROMPTS = 'a photo of a red-and-white 330-ml soda can'`, either cell 11 rejects
  the phrase with a message naming it and the 16-token limit (no model call), or cell 13 completes and the manifest
  records the truncation. No path ends in a tensor-shape error.

### OVT-M3 — Major: the default sanity demonstration shows two of three shapes as IoU 0.0, with no guidance, and cannot show the no-NMS behaviour it promises to teach

- **Cell/section:** cells 12–15 (Sections 6–7) and cell 18 *Interpretation*. Cell 12 says "No run with the pinned
  weights has been recorded for this checkpoint yet, so look at the boxes and scores yourself"
  (`tools/notebook_template.py:183`).
- **Observed issue:** at the default threshold 0.1 the scene returns one box (`a red circle`, 0.4487). The report
  lists the black rectangle and the blue triangle at `box_iou` 0.0 with `matched_label: "a red circle"`. Neither the
  notebook nor the report explains why. In fact the model localises all three shapes well: the top-scoring box per
  phrase has IoU 0.9235, 0.9671 and 0.8641. The rectangle and triangle score 0.0373 and 0.0985, just under the
  threshold. The learning objective "read … the absence of non-maximum suppression correctly" and the experiment
  "lower `threshold` to 0.05 and count the duplicate boxes" find **no duplicates** (0.05 → 2 boxes, 0.02 → 3, one
  per shape).
- **Consequence:** the only evaluation output reads as "the detector failed on 2/3 shapes" or "the coordinate
  mapping is wrong". Cells 14 and 18 say a sanity IoU proves the coordinate mapping, and the learner has no way to tell
  which reading is right. The no-NMS lesson is described but never observed. The first-time learner is likely to be
  misled about what the sanity check showed.
- **Evidence:** documented execution evidence (T4 record: 1 detection, same score and box) and direct execution on
  CPU (probes P2, P3, which reproduced the record exactly).
- **Recommended correction:** add an *Expected result / What to notice* note after Sections 6 and 7. It should give
  the recorded outcome as one observation for the named runtime, say that a 0.0 IoU at 0.1 means "no box above the
  threshold", and show the learner how to tell a score miss from a localisation miss. One way is to print the
  top-scoring box per phrase and its IoU (threshold 0 or a per-query max) in the evaluation report or in a small
  extra cell. Either add a scene or prompt form (the `a photo of …` form lifts the triangle to 0.1278) that shows a
  duplicate at a stated threshold, or drop the duplicate-counting claim. Remove the "no run recorded" sentence. Edit
  `tools/notebook_template.py` (cells 12, 14, 18) and, if the report gains a field, `evaluation_report`.
- **Acceptance check:** on the default run, the notebook prints for each reference phrase the best box's score and
  IoU, and states in prose which shapes fall below the threshold. The documented threshold/phrasing experiment yields
  the behaviour its text predicts (or the text no longer predicts duplicates).

### OVT-m1 — Minor: stale "no run recorded / runtimes not measured" text

- **Cell/section:**
  - Cell 1: "Runtimes are not measured in this revision" (`tools/notebook_template.py:75`).
  - Cell 12: "No run with the pinned weights has been recorded for this checkpoint yet" (`:183`).
- **Consequence:** the README, STATUS and release-verification record a 314.8 s T4 run (`detect` 0.53 s), so the
  learner gets conflicting status and no runtime expectation (UX12). The text is a knowingly stale instruction (SRC3
  once released).
- **Evidence:** source inspection (probe P1).
- **Correction / acceptance check:** state the recorded timing as measured for the named environment, and remove
  both sentences. A grep for `not measured in this revision` and `No run with the pinned weights` returns nothing.
  Bundle this with the OVT-M1 regeneration, because any edit needs a new exact-blob run.

### OVT-m2 — Minor: an empty BYOD location field imports `google.colab`, although Jupyter is a stated runtime

- **Cell/section:** cell 9 (`tools/notebook_template.py:114–119`). Cell 1 says "Google Colab or Jupyter".
- **Evidence:** direct execution. `USE_BYOD = True` with `BYOD_IMAGE_PATH = ''` outside Colab gives
  `ModuleNotFoundError: No module named 'google.colab'`. A set location field works (EXE2 met). The Colab upload
  dialog was not exercised.
- **Correction / acceptance check:** guard the upload branch. When `google.colab` is unavailable, raise a message
  that says "set BYOD_IMAGE_PATH to an image file". Outside Colab, the empty-field case prints that message instead
  of a `ModuleNotFoundError`.

### OVT-m3 — Minor: execution-evidence gaps: no Colab record, REL12 pending, step-6 threshold-0.05 count not captured

- **Evidence:** documented execution evidence. `docs/release-verification.md` lists one Kaggle T4 run. The Colab
  badge is the primary entry point. Step 7 (REL12: BYOD photograph plus a 17-phrase rejection) and the step-6 count
  at threshold 0.05 are recorded as not done. The status is honestly Candidate.
- **Correction / acceptance check:** record a Colab Run all of the fixed blob, one BYOD photograph through
  `BYOD_IMAGE_PATH`, and one rejection, each with revision and runtime. The direct CPU results here (2 boxes at 0.05;
  the 17-phrase refusal) can guide what to expect, but they are not that record.

### OVT-m4 — Minor: Section 3 prints a hard-coded weight "source"

- **Cell/section:** cell 7 prints `getattr(pipe, 'source', 'local-snapshot')` (`tools/build_notebook.py:501`).
  `OwlViTDetectionPipeline` has no `source` attribute, so the value printed is always the fallback literal.
- **Evidence:** direct execution (`hasattr(pipe, 'source')` is `False`).
- **Correction / acceptance check:** print `WEIGHTS_DIR` and the verified `model.safetensors` digest from the
  manifest. No `getattr(..., 'local-snapshot')` fallback remains.

### OVT-m5 — Minor: the spec 2.2 guided layer is partial, and the notebook declares spec 2.1

- **Observed issue:** the notebook has:
  - no *How to use this notebook* section (GDL2) and no roadmap (GDL3);
  - an unlabelled, uncollapsed 18.5 KB carried module cell, with no **Infrastructure** label telling learners they
    may skip it (GDL11);
  - no predictions before the experiments (GDL7);
  - no interpretation checkpoints (GDL9);
  - no troubleshooting section, although restart, download, BYOD and token-limit failures are all expected (GDL13);
  - no conclusion template (GDL14).

  "Look for" notes appear in only 2 sections (GDL8, partial).
- **Evidence:** source inspection (probe P1). `metadata.dimer.notebook_spec = '2.1'`.
- **Correction / acceptance check:** add the GDL elements in `tools/notebook_template.py` and declare spec 2.2. A
  review against GDL1–GDL15 finds each element present.

### OVT-m6 — Minor: the evaluation report names a "matched_label" for references with zero overlap

- **Cell/section:** `evaluation_report` (`pipeline.py:264–331`) takes the best-IoU detection even when every IoU is
  0.0, which picks the first detection.
- **Evidence:** direct execution (P2). Both the black rectangle and the blue triangle report `matched_label: "a red
  circle"`, `value: 0.0`.
- **Consequence:** the JSON reads as if the rectangle were mistaken for the circle, which compounds OVT-M3.
- **Correction / acceptance check:** when the best IoU is 0 (or below a stated floor), report `matched_label: null`
  and a `"no overlapping detection"` note. The default report then shows null for the two missed shapes.

### OVT-S1 — Suggestion: turn the *Next experiments* into a Predict → Change → Run → Observe → Explain activity

The paragraph should:

- name the cell to edit (`threshold`, `BYOD_PROMPTS` and `USE_BYOD` live in Section 4, cell 9);
- give the exact rerun range ("run from Section 4 to Section 8");
- ask for a prediction first;
- add a `REFERENCE_BOXES` form or dict for the "hand-label a few objects" step, with keys passed through
  `format_prompts` so `label_matches_reference` is not falsely `False` for capitalised keys.

The direct rerun from cell 9 works and refreshes all outputs (GDL10, UX5).

### OVT-S2 — Suggestion: drop or align the unused `torchaudio==2.11.0` pin

The notebook never imports torchaudio, and its minor version differs from `torch==2.14.0`. Same as owlv2 OWD-S4.

### OVT-S3 — Suggestion: apply EXIF orientation to BYOD photographs

Cell 9 opens BYOD images without `ImageOps.exif_transpose`, so a phone photo can be detected and annotated sideways.
The boxes stay self-consistent with the stored pixels, but the learner's view may not match. Source inspection only;
not executed with an EXIF-rotated file.

### OVT-S4 — Suggestion: show the annotated image inline

Cell 17 saves `owlvit_detection_annotated.png` but does not display it. The prose says to "look at the boxes and
scores yourself", so showing it inline (and, for BYOD, side by side with the input) would make that easy (UX3, UX11).

## 4. Readiness

**Needs revision.** Three Majors are open:

- OVT-M1: the restart.
- OVT-M2: the token-limit validation gap and the false truncation claim.
- OVT-M3: an unexplained two-of-three-miss sanity result, and an NMS lesson that cannot be observed.

These applicable MUSTs are unresolved: RUN1/RUN10/ENV6, DAT12/DAT19, and REL12 (pending). Execution evidence exists
for the default path on Kaggle T4 for this exact blob. Remaining gates after the fixes:

1. A new exact-blob Run all on Colab (and Kaggle) with no restart.
2. A BYOD photograph run and a rejection run (REL12), including one over-16-token phrase.

## 5. Verified versus inferred

- **Verified by direct execution (CPU, labelled):**
  - The default path reproduces the T4 detection exactly (P2).
  - Over-16-token phrases of 42–43 characters pass validation and crash `detect` (P3, P5).
  - The per-phrase top boxes localise all three shapes (P3).
  - No duplicates appear at 0.05 or 0.02 (P3).
  - The threshold rerun refreshes all outputs (P3).
  - BYOD via the location field reaches export, and four invalid inputs are refused with named ceilings (P4).
  - The empty field outside Colab raises `ModuleNotFoundError` (P4).
  - Carried-cell parity and `--check` pass (P0).
- **Verified from documented evidence:** the restart on Kaggle T4 for this exact blob, and its outputs.
- **Inferred, not executed:**
  - The restart on Colab.
  - The Colab upload dialog.
  - EXIF behaviour.
  - Exact CUDA scores. The T4 score equals the CPU score to four decimals, so they likely match.
- **Most likely to be wrong:** the severity of OVT-M3. The notebook already hedges ("one observation on one drawn
  scene … look at the boxes yourself"), so a reviewer could reasonably call the missing guidance Minor. It is rated
  Major because the only evaluation output, as printed, points the learner to the wrong conclusion.

Probe files: `owlvit_detection_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
