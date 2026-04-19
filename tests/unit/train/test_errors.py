"""Structured error types carry enough context for CLI messages."""

from __future__ import annotations

from dlm.train.errors import (
    DiskSpaceError,
    OOMError,
    ResumeIntegrityError,
    TrainingError,
)


class TestDiskSpaceError:
    def test_message_includes_gb(self) -> None:
        exc = DiskSpaceError(required_bytes=10_000_000_000, free_bytes=2_000_000_000)
        msg = str(exc)
        assert "10.0 GB" in msg
        assert "2.0 GB" in msg
        assert exc.required_bytes == 10_000_000_000
        assert exc.free_bytes == 2_000_000_000

    def test_is_training_error(self) -> None:
        exc = DiskSpaceError(required_bytes=1, free_bytes=0)
        assert isinstance(exc, TrainingError)


class TestOOMError:
    def test_carries_full_context(self) -> None:
        exc = OOMError(
            step=12,
            peak_bytes=22_000_000_000,
            free_at_start_bytes=22_400_000_000,
            current_grad_accum=2,
            recommended_grad_accum=4,
        )
        assert exc.step == 12
        assert exc.peak_bytes == 22_000_000_000
        assert exc.free_at_start_bytes == 22_400_000_000
        assert exc.current_grad_accum == 2
        assert exc.recommended_grad_accum == 4
        assert "step 12" in str(exc)


class TestResumeIntegrityError:
    def test_plain_message_propagates(self) -> None:
        exc = ResumeIntegrityError("sha mismatch foo bar")
        assert "sha mismatch" in str(exc)
        assert isinstance(exc, TrainingError)
