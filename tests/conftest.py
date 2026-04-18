"""Shared pytest fixtures and options.

Fast, local-only: no torch / transformers / huggingface_hub imports at
collection time. Fixtures that need heavy deps import them lazily.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Regenerate golden-output fixtures instead of asserting against them.",
    )


@pytest.fixture
def update_goldens(request: pytest.FixtureRequest) -> bool:
    """Expose --update-goldens to tests (golden.py reads this)."""
    return bool(request.config.getoption("--update-goldens"))


@pytest.fixture
def seeded_rng() -> Iterator[int]:
    """Seed Python's random for tests that need local determinism.

    We don't seed numpy / torch here — the hardware_mocks / tiny_model
    fixtures own that, and not every test imports them.
    """
    seed = 42
    state = random.getstate()
    random.seed(seed)
    try:
        yield seed
    finally:
        random.setstate(state)


@pytest.fixture
def dlm_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate $DLM_HOME under tmp_path for per-test store sandboxing."""
    home = tmp_path / "dlm-home"
    home.mkdir()
    monkeypatch.setenv("DLM_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def _offline_hf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fast-path tests never touch HF. Tests that need the network use the
    `tiny_model` fixture which clears these vars in its own scope.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    # Always reinforce our telemetry-off contract.
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")


@pytest.fixture
def hf_cache_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect HF_HOME to tmp for tests that download models.

    The session-scoped `tiny_model` fixture overrides this to a shared
    cache dir so CI can reuse downloads across tests.
    """
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache))
    return cache


# --- platform env cleanup -----------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_cuda_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove CUDA env vars that could contaminate hardware-mock tests."""
    for var in ("CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING"):
        monkeypatch.delenv(var, raising=False)
