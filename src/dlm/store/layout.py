"""Constants naming the subdirectories and files inside a store.

Centralized so every module that touches store paths agrees on the layout
without hardcoding strings.

```
~/.dlm/store/<dlm_id>/
    manifest.json
    .lock                       # exclusive-lock file (PID-based)
    adapter/
        # Flat single-adapter layout (default for documents without
        # `training.adapters`):
        current.txt             # plain-text pointer to versions/vNNNN
        versions/
            v0001/               # PEFT save_pretrained output
            v0002/
        # Multi-adapter layout (documents with `training.adapters`):
        <adapter_name>/
            current.txt
            versions/
                v0001/
                v0002/
    training_state.pt            # paired with adapter/current
    training_state.pt.sha256     # integrity check
    replay/
        corpus.zst               # zstd-framed sections
        index.json
    exports/
        <quant>/
            base.gguf
            adapter.gguf
            Modelfile
    cache/
        <base_model_slug>/       # HF snapshot
    logs/
        train-YYYYMMDD-HHMMSS.log
```

The two adapter layouts are mutually exclusive per store: a document is
either flat or multi-adapter at parse time. The flat layout lives under
`adapter/versions/` directly; the multi-adapter layout nests one level
deeper under `adapter/<name>/versions/`.

Subdirectories are created lazily on first write by the sprint that owns
them — `ensure_layout()` only creates the top-level skeleton.
"""

from __future__ import annotations

from typing import Final

# Top-level filenames
MANIFEST_FILENAME: Final = "manifest.json"
LOCK_FILENAME: Final = ".lock"
TRAINING_STATE_FILENAME: Final = "training_state.pt"
TRAINING_STATE_SHA_FILENAME: Final = "training_state.pt.sha256"

# Top-level subdirs
ADAPTER_DIR: Final = "adapter"
REPLAY_DIR: Final = "replay"
EXPORTS_DIR: Final = "exports"
CACHE_DIR: Final = "cache"
TOKENIZED_CACHE_DIR: Final = "tokenized-cache"
CONTROLS_DIR: Final = "controls"
BLOBS_DIR: Final = "blobs"
VL_CACHE_DIR: Final = "vl-cache"
AUDIO_CACHE_DIR: Final = "audio-cache"
LOGS_DIR: Final = "logs"

# Under adapter/
ADAPTER_CURRENT_POINTER: Final = "current.txt"
ADAPTER_VERSIONS_DIR: Final = "versions"

# Under replay/
REPLAY_CORPUS_FILENAME: Final = "corpus.zst"
REPLAY_INDEX_FILENAME: Final = "index.json"

# Top-level directories created by ensure_layout()
ALWAYS_CREATE_DIRS: Final = (
    ADAPTER_DIR,
    LOGS_DIR,
)
# These are created lazily by their owning sprint:
# - replay/ by Sprint 08
# - exports/<quant>/ by Sprint 11
# - cache/<slug>/ by Sprint 06
