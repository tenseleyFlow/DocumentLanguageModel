"""Tiny-model fixture: SmolLM2-135M-Instruct, cached and reused across tests.

Design notes:

- Sprint 06 owns the base-model registry with pinned revisions. Until that
  lands, we accept `main` here and print a warning on first use. The CI
  cache key includes the stored revision, so cache hits are stable.
- `snapshot_download` into the caller's `HF_HOME` (set by CI to a cached
  location). Local runs default to `~/.cache/huggingface`.
- This module imports `huggingface_hub` lazily so test collection stays
  fast for paths that don't need the tiny model.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

_LOG: Final = logging.getLogger(__name__)

TINY_MODEL_HF_ID: Final = "HuggingFaceTB/SmolLM2-135M-Instruct"

# TODO(sprint-06): pin to a 40-char commit SHA via the base-model registry.
TINY_MODEL_REVISION: Final = os.environ.get("DLM_TINY_MODEL_REVISION", "main")


def tiny_model_path() -> Path:
    """Download (if needed) the tiny model; return the cached directory path.

    Raises on network failure in offline mode; callers must mark their tests
    with `online` + `slow`.
    """
    # Lazy import so fast-path tests never pull huggingface_hub.
    from huggingface_hub import snapshot_download

    if TINY_MODEL_REVISION == "main":
        _LOG.warning(
            "TINY_MODEL_REVISION unpinned (using 'main'). Sprint 06 will pin.",
        )

    cache_dir = snapshot_download(
        repo_id=TINY_MODEL_HF_ID,
        revision=TINY_MODEL_REVISION,
        local_files_only=_offline_mode(),
    )
    return Path(cache_dir)


def _offline_mode() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "0") == "1"


# --- pytest fixture wrapper ---------------------------------------------------


@pytest.fixture(scope="session")
def tiny_model_dir() -> Iterator[Path]:
    """Session-scoped fixture — download happens once per test session.

    Tests that use it must carry `@pytest.mark.online` and usually
    `@pytest.mark.slow`.
    """
    # Clear the autouse offline env for this fixture's scope so downloads work
    # in tests that opted in (the `online` marker is the gate).
    original = {
        "HF_HUB_OFFLINE": os.environ.pop("HF_HUB_OFFLINE", None),
        "TRANSFORMERS_OFFLINE": os.environ.pop("TRANSFORMERS_OFFLINE", None),
        "HF_DATASETS_OFFLINE": os.environ.pop("HF_DATASETS_OFFLINE", None),
    }
    try:
        yield tiny_model_path()
    finally:
        for k, v in original.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
