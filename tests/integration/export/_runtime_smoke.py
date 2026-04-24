"""Shared helpers for Sprint 41 live runtime smoke tests."""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest


def require_loopback_bind() -> None:
    """Skip when this host blocks loopback binds in the current sandbox."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
    except PermissionError as exc:
        pytest.skip(f"loopback bind blocked on this host: {exc}")


def vendor_server_built() -> bool:
    """True when the vendored llama.cpp server binary exists."""
    vendor_root = Path(__file__).resolve().parents[3] / "vendor" / "llama.cpp"
    return (vendor_root / "build" / "bin" / "llama-server").is_file()


@contextmanager
def cleared_offline_env() -> Iterator[None]:
    """Temporarily clear the offline HF env so cached snapshots can resolve."""
    offline_vars = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    saved = {key: os.environ.pop(key, None) for key in offline_vars}
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value
