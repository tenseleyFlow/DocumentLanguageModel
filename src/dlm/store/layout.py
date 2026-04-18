"""Constants naming the subdirectories and files inside a store.

Centralized so every module that touches store paths agrees on the layout
without hardcoding strings.

```
~/.dlm/store/<dlm_id>/
    manifest.json
    .lock                       # exclusive-lock file (PID-based)
    adapter/
        current.txt             # plain-text pointer to versions/vNNNN
        versions/
            v0001/               # PEFT save_pretrained output (Sprint 09)
            v0002/
    training_state.pt            # paired with adapter/current  (Sprint 09)
    training_state.pt.sha256     # integrity check              (Sprint 09)
    replay/
        corpus.zst               # zstd-framed sections         (Sprint 08)
        index.json
    exports/
        <quant>/
            base.gguf            # Sprint 11
            adapter.gguf         # Sprint 11
            Modelfile            # Sprint 12
    cache/
        <base_model_slug>/       # HF snapshot                  (Sprint 06)
    logs/
        train-YYYYMMDD-HHMMSS.log
```

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
