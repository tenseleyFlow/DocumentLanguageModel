"""Strict-mode failures for `MetricsRecorder`."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from dlm.metrics.events import StepEvent, TokenizationEvent
from dlm.metrics.recorder import MetricsRecorder


@contextmanager
def _failing_connect(_store_root: Path) -> Iterator[sqlite3.Connection]:
    raise sqlite3.OperationalError("database is locked")
    yield sqlite3.connect(":memory:")


class TestStrictMode:
    def test_step_write_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import dlm.metrics.recorder as recorder_mod

        monkeypatch.setattr(recorder_mod, "connect", _failing_connect)
        rec = MetricsRecorder(tmp_path, strict=True)

        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            rec.record_step(StepEvent(run_id=1, step=1, loss=1.0))

    def test_tokenization_write_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.metrics.recorder as recorder_mod

        monkeypatch.setattr(recorder_mod, "connect", _failing_connect)
        rec = MetricsRecorder(tmp_path, strict=True)

        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            rec.record_tokenization(
                TokenizationEvent(
                    run_id=1,
                    total_sections=4,
                    cache_hits=2,
                    cache_misses=2,
                    total_tokenize_seconds=0.25,
                    cache_bytes_after=1024,
                )
            )
