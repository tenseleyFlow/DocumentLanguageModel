"""`seed_everything` — env contract + RNG seeding + class classification."""

from __future__ import annotations

import os
import random

from dlm.train.determinism import seed_everything


class TestSeedEverything:
    def test_sets_cublas_workspace_default(self, monkeypatch) -> None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        seed_everything(42)
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"

    def test_preserves_user_cublas_workspace(self, monkeypatch) -> None:
        """`setdefault` must NOT override a user-set value."""
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        seed_everything(42)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"

    def test_same_seed_same_random_stream(self) -> None:
        seed_everything(42)
        a = [random.random() for _ in range(5)]
        seed_everything(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_different_seed_different_stream(self) -> None:
        seed_everything(1)
        a = [random.random() for _ in range(5)]
        seed_everything(2)
        b = [random.random() for _ in range(5)]
        assert a != b

    def test_summary_carries_seed_back(self) -> None:
        summary = seed_everything(123)
        assert summary.seed == 123
        assert summary.class_ in ("strict", "best_effort", "loose")
        assert isinstance(summary.notes, list)

    def test_mps_path_adds_best_effort_note(self, monkeypatch) -> None:
        """On MPS-only hosts (our dev box), the banner notes it."""
        summary = seed_everything(0)
        # Either we're on MPS (note present) or on CUDA (strict).
        if summary.class_ == "best_effort":
            assert any("determinism" in note.lower() for note in summary.notes)
