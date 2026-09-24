# Weight provenance and hosting

- Upstream: `google/owlvit-base-patch32`
- Revision: **not yet pinned** (`MODEL_REVISION = "unpinned"`). Run `python tools/pin_snapshot.py` to resolve the Hub's `main` to a 40-hex commit, download every manifest-listed file at that commit, cross-check the LFS file against the SHA-256 the Hub records, and write the commit and digests into the manifest and `src/owlvit_detection_pipeline/pipeline.py`.
- Weight format: SafeTensors (`model.safetensors`, 612,983,940 bytes as the Hub reported for `main` when this repository was built). The Hub repository also holds `pytorch_model.bin` (613,049,157 bytes), a pickle checkpoint with the same weights; it is not in the manifest and is never staged or loaded.
- Manifest: `weights/owlvit-base-patch32/dimer-base-manifest.json` (8 files: `README.md`, `config.json`, `merges.txt`, `model.safetensors`, `preprocessor_config.json`, `special_tokens_map.json`, `tokenizer_config.json`, `vocab.json`; `totalBytes` 614579712). Every `sha256` is `null` until the pin tool runs.
- Committed copy: `preprocessor_config.json` (392 bytes) is committed as the Hub served it, so an offline test can check the 768×768 input size behind `MAX_DETECTIONS`. The pin tool replaces it with the bytes downloaded at the pinned commit before hashing.
- Upstream weight license: Apache-2.0 (the checkpoint's `README.md` front matter and the Hub's licence tag).
- Hosting: Apache-2.0 permits use, modification, distribution and commercial use, subject to keeping the licence and notices. The Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is git-ignored).
- Fresh clone, once pinned: `stage_missing_files(allow_download=True)` fetches only the manifest-listed files that are absent, at the pinned revision; `verify_snapshot()` then checks every file before any load. `weights/**` is marked `-text` in `.gitattributes`, so Windows `core.autocrlf` cannot rewrite the committed files and break their digests.
- Loader trust boundary: Transformers `OwlViTForObjectDetection` and `OwlViTProcessor` (with `CLIPTokenizer`) with `trust_remote_code=False` and `local_files_only=True` from the verified directory.
