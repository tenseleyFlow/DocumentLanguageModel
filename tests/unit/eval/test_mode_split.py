"""Post-training val-loss split helper (audit-08 N9)."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from dlm.eval.mode_split import _safe_eval_loss, compute_val_loss_by_mode


class _FakeDataset:
    """Minimal Dataset-shaped stub: list iteration + `.select(indices)`."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._rows)

    def select(self, indices: list[int]) -> _FakeDataset:
        return _FakeDataset([self._rows[i] for i in indices])


def _trainer_with_fixed_losses(
    cpt_loss: float | None = 0.5,
    sft_loss: float | None = 0.3,
) -> MagicMock:
    """Mock: .evaluate() returns different eval_loss per subset size.

    We use subset length as the key so we can identify which mode was
    queried without inspecting row contents. The test asserts both
    arms fire on the right subsets.
    """
    trainer = MagicMock()
    call_log: list[int] = []

    def _evaluate(*, eval_dataset: _FakeDataset) -> dict[str, float]:
        n = len(eval_dataset)
        call_log.append(n)
        # Small trick: losses parameterized by call order so both
        # get exercised; we don't care which is which as long as the
        # helper returns them in (cpt, sft) order.
        if cpt_loss is not None and len(call_log) == 1:
            return {"eval_loss": cpt_loss}
        if sft_loss is not None:
            return {"eval_loss": sft_loss}
        return {}

    trainer.evaluate.side_effect = _evaluate
    trainer._call_log = call_log  # noqa: SLF001
    return trainer


class TestEmptyOrMissing:
    def test_none_val_ds_returns_both_none(self) -> None:
        trainer = MagicMock()
        assert compute_val_loss_by_mode(trainer, None) == (None, None)
        trainer.evaluate.assert_not_called()

    def test_empty_val_ds_returns_both_none(self) -> None:
        trainer = MagicMock()
        assert compute_val_loss_by_mode(trainer, _FakeDataset([])) == (None, None)
        trainer.evaluate.assert_not_called()

    def test_non_sized_dataset_returns_both_none(self) -> None:
        trainer = MagicMock()
        assert compute_val_loss_by_mode(trainer, _NonSizedDataset([{"text": "prose"}])) == (
            None,
            None,
        )
        trainer.evaluate.assert_not_called()


class TestModeClassification:
    def test_only_cpt_rows(self) -> None:
        trainer = _trainer_with_fixed_losses(cpt_loss=0.7, sft_loss=None)
        val = _FakeDataset([{"text": "prose a"}, {"text": "prose b"}, {"text": "prose c"}])
        cpt, sft = compute_val_loss_by_mode(trainer, val)
        assert cpt == 0.7
        assert sft is None
        # Only CPT subset was evaluated.
        assert trainer._call_log == [3]

    def test_only_sft_rows(self) -> None:
        trainer = _trainer_with_fixed_losses(cpt_loss=None, sft_loss=0.4)
        val = _FakeDataset(
            [
                {"messages": [{"role": "user", "content": "hi"}]},
                {"messages": [{"role": "user", "content": "hi"}]},
            ]
        )
        cpt, sft = compute_val_loss_by_mode(trainer, val)
        assert cpt is None
        assert sft == 0.4

    def test_mixed_rows(self) -> None:
        trainer = _trainer_with_fixed_losses(cpt_loss=0.9, sft_loss=0.5)
        val = _FakeDataset(
            [
                {"text": "prose"},
                {"messages": []},
                {"text": "more prose"},
                {"messages": []},
            ]
        )
        cpt, sft = compute_val_loss_by_mode(trainer, val)
        assert cpt == 0.9
        assert sft == 0.5
        # Both subsets evaluated; sizes 2 each.
        assert sorted(trainer._call_log) == [2, 2]

    def test_preference_rows_skipped(self) -> None:
        """Preference triples aren't part of CPT or SFT — they shouldn't
        inflate either subset."""
        trainer = _trainer_with_fixed_losses(cpt_loss=0.1, sft_loss=None)
        val = _FakeDataset(
            [
                {"text": "prose"},
                {"prompt": "q", "chosen": "c", "rejected": "r"},
                {"prompt": "q", "chosen": "c", "rejected": "r"},
            ]
        )
        cpt, sft = compute_val_loss_by_mode(trainer, val)
        assert cpt == 0.1
        assert sft is None
        assert trainer._call_log == [1]  # only the one CPT row


class TestEvalFailures:
    def test_evaluate_exception_yields_none(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A stack-version skew that makes evaluate() raise shouldn't
        crash training — the affected mode just stays None."""
        caplog.set_level(logging.WARNING, logger="dlm.eval.mode_split")
        trainer = MagicMock()
        trainer.evaluate.side_effect = RuntimeError("TRL drift")
        val = _FakeDataset([{"text": "a"}, {"messages": []}])
        cpt, sft = compute_val_loss_by_mode(trainer, val)
        assert cpt is None
        assert sft is None
        assert "val-loss split skipped cpt evaluation" in caplog.text
        assert "val-loss split skipped sft evaluation" in caplog.text

    def test_missing_eval_loss_key_yields_none(self) -> None:
        trainer = MagicMock()
        trainer.evaluate.return_value = {"other_metric": 1.0}
        val = _FakeDataset([{"text": "a"}])
        cpt, sft = compute_val_loss_by_mode(trainer, val)
        assert cpt is None
        assert sft is None

    def test_non_numeric_eval_loss_yields_none(self) -> None:
        trainer = MagicMock()
        trainer.evaluate.return_value = {"eval_loss": object()}
        val = _FakeDataset([{"text": "a"}])
        cpt, sft = compute_val_loss_by_mode(trainer, val)
        assert cpt is None
        assert sft is None

    def test_select_failure_yields_none(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING, logger="dlm.eval.mode_split")
        trainer = MagicMock()
        trainer.evaluate.return_value = {"eval_loss": 0.0}
        # Dataset iteration works, but subset selection does not.
        bad_val = _NoSelectDataset([{"text": "a"}])
        cpt, sft = compute_val_loss_by_mode(trainer, bad_val)
        # Both None — the helper couldn't build subsets.
        assert cpt is None
        assert sft is None
        assert "val-loss split skipped cpt subset selection" in caplog.text


class _NoSelectDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._rows)


class _NonSizedDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._rows)


def test_safe_eval_loss_value_error_yields_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="dlm.eval.mode_split")
    trainer = MagicMock()
    trainer.evaluate.side_effect = ValueError("bad eval")
    val = _FakeDataset([{"text": "a"}])

    assert _safe_eval_loss(trainer, val, [0], mode="cpt") is None
    assert "val-loss split skipped cpt evaluation" in caplog.text
