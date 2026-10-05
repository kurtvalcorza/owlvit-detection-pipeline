# owlvit_detection_colab — fleet-sweep fixes (2026-10-05)

Targeted fix of the 2026-10-05 fleet sweep findings. There is no full Notebook Review Framework v1 report for this
notebook; each flag was first confirmed in the cell source on `main` (`4873100`). All changes are made in the
generator (`tools/build_notebook.py`, `tools/notebook_template*.py`) and the notebook is regenerated. STATUS and the
release labels are unchanged. **Readiness: Verification pending** (hosted Run all not yet done).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R (restart guard) | Fixed — hosted confirmation pending | Confirmed (the recorded Kaggle T4 run of 2026-09-25 stopped with `RuntimeError: Core dependencies changed while older modules were loaded` and needed a restart): Section 1 ran `pip install` into the kernel and raised "Restart the runtime" on stale modules. Generator upgraded to `build_notebook.py/2.2` (the fleet isolated runtime): one kernel cell downloads the pinned `uv` 0.12.15 wheel (size + SHA-256), builds a managed CPython 3.12.12 environment from the new hash lock `tutorials/requirements-colab.lock.txt` (`--require-hashes --only-binary :all:`), and routes every later cell to one persistent worker. The environment folder is keyed on the lock digest and reused by a re-run or a second Run all; re-running Section 1 keeps the live worker and its variables; the worker gets `MPLBACKEND=Agg` and no `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`. | Section 1 (kernel cell + "Record the runtime"); `tools/build_notebook.py`; `tutorials/requirements-colab.lock.txt`; `tools/validate_release_assets.py` (install markers, bootstrap check, kernel cell excluded from the library-use scan); `docs/release-verification.md` (the line describing that check) | `test_swp_r_no_pip_install_or_restart_in_any_cell`, `test_swp_r_lock_is_carried_hash_locked_and_matches_pins`, `test_swp_r_environment_keyed_on_lock_and_child_env_cleaned`, `test_swp_r_section1_reuses_environment_and_worker_when_rerun` (executes the notebook's own kernel cell with a stand-in IPython shell; the worker runs on the test interpreter) |
| SWP-G (guided layer) | Fixed | Confirmed: GUIDED mode with 0 of 9 guided markers. Added audience and Input → Model → Output table, How to use this notebook, roadmap, a Learner prerequisite, three Predict / Check your reasoning pairs (Sections 5–7) quoting the recorded Kaggle T4 run (the 49-character phrase refusal; one detection, `a red circle` at 0.4487; IoU 0.9671; rectangle and triangle not detected), Troubleshooting, Glossary and a Conclusion template. Sections 1–3 labelled Infrastructure and collapsed. | Template opening, prerequisites, Sections 5–7, closing | `test_swp_g_guided_layer_present`, `test_swp_g_infrastructure_cells_labelled_and_collapsed`, `test_swp_g_no_leftover_placeholders` |
| SWP-A (quality asserts) | Not applicable | The sweep found no quality assert; none present. | — | sweep row |
| SWP-F (frozen re-run) | Not applicable | TASK-INFERENCE: no training. | — | — |
| SWP-B (BYOD) | Fixed (partly present before) | A `BYOD_IMAGE_PATH` field already existed. In the same cell, the Colab fallback was unguarded: off Colab it raised a bare `ImportError`, and a cancelled upload a bare `StopIteration`; a missing path raised Pillow's error. These now stop with messages naming the field or path and the rule. | Section 4 | `test_swp_b_byod_path_works_and_missing_path_is_named`, `test_swp_b_empty_path_off_colab_and_cancelled_upload_are_named` (execute the notebook's Section 4) |

## User-visible changes

- Section 1 no longer installs into the notebook's Python and never asks for a restart; it builds (first run) or reuses `dimer_isolated_env_<lock digest>/` and every later code cell runs there. Linux x86_64 runtimes only.
- BYOD: clear messages for an empty path off Colab, a cancelled upload and a missing path.
- Guided material added; Sections 1–3 collapsed.

## Verification (offline; not clean-runtime evidence)

- Real input: none of the model stages could run here (the Hugging Face Hub is unreachable and torch is not installed).
- Stand-in: the kernel-cell test runs the generated bootstrap against a pre-built environment folder whose `python` is the test interpreter (routing, reuse and idempotence are real; the managed CPython and locked packages are stand-ins).
- `python tools/build_notebook.py --check`: OK. `python tools/validate_release_assets.py`: PASS. `ruff check src tests tools`: clean.
- `pytest` with CI's dependencies except torch (CI installs CPU torch; torch-gated tests skip here): 41 passed, 1 skipped before → 52 passed, 1 skipped after.
- Every code cell of the regenerated notebook parses.

## Remaining gates

- A hosted **Run all in one pass** on a fresh runtime (expected: no restart prompt; Section 1 builds the environment; a second Run all reports `'reused': True`).
- The REL12 BYOD run with the BYOD gates and path fields set.
- A full Notebook Review Framework v1 review has not been done.
