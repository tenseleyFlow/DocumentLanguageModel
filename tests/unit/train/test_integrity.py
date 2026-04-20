"""Post-training finite-weights + finite-eval gates.

Both gates were added after a MPS + tiny-data + no-warmup `dlm train`
persisted a fully NaN adapter and silently promoted it to `current.txt`.
Downstream consumers then produced NaN logits; the user got exit 0 and
no signal. These tests guard against that regression.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from dlm.train.checkpoint_commit import commit_version
from dlm.train.integrity import (
    NaNEvalError,
    NaNWeightsError,
    assert_eval_finite,
    assert_finite_adapter,
    audit_trainable_finite,
)


class _TinyLoRAModel(nn.Module):
    """Minimal module with trainable lora_A/lora_B tensors."""

    def __init__(self, *, nan: bool = False, inf: bool = False) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(4, 8))
        self.lora_B = nn.Parameter(torch.zeros(8, 4))
        # A frozen "base" param — must be ignored by the audit even if non-finite.
        self.base = nn.Parameter(torch.full((4, 4), float("nan")), requires_grad=False)
        if nan:
            with torch.no_grad():
                self.lora_A[0, 0] = float("nan")
        if inf:
            with torch.no_grad():
                self.lora_B[1, 1] = float("inf")


class TestAuditTrainableFinite:
    def test_clean_model_reports_ok(self) -> None:
        result = audit_trainable_finite(_TinyLoRAModel())
        assert result.ok
        assert result.checked == 2  # lora_A + lora_B
        assert result.offending == ()

    def test_nan_trainable_param_is_flagged(self) -> None:
        result = audit_trainable_finite(_TinyLoRAModel(nan=True))
        assert not result.ok
        assert "lora_A" in result.offending[0]

    def test_inf_trainable_param_is_flagged(self) -> None:
        result = audit_trainable_finite(_TinyLoRAModel(inf=True))
        assert not result.ok
        assert "lora_B" in result.offending[0]

    def test_frozen_nan_base_is_ignored(self) -> None:
        # The frozen `base` tensor is NaN-filled but requires_grad=False;
        # audit must skip it (we only check what training updates).
        result = audit_trainable_finite(_TinyLoRAModel())
        assert result.ok
        assert all("base" not in name for name in result.offending)


class TestAssertFiniteAdapter:
    def test_clean_model_does_not_raise(self) -> None:
        assert_finite_adapter(_TinyLoRAModel())  # no raise

    def test_nan_model_raises(self) -> None:
        with pytest.raises(NaNWeightsError) as exc:
            assert_finite_adapter(_TinyLoRAModel(nan=True))
        assert "NaN/inf" in str(exc.value)
        assert len(exc.value.full_offending) == 1


class TestAssertEvalFinite:
    def test_empty_log_history_no_raise(self) -> None:
        assert_eval_finite([])

    def test_no_eval_entries_no_raise(self) -> None:
        # log_history may contain train-only entries — the contract is
        # "check iff eval ran", so no eval entries means nothing to check.
        assert_eval_finite([{"loss": 2.0, "step": 1}, {"loss": 1.5, "step": 2}])

    def test_finite_eval_does_not_raise(self) -> None:
        assert_eval_finite([{"eval_loss": 1.8, "step": 10}])

    def test_nan_eval_raises(self) -> None:
        with pytest.raises(NaNEvalError) as exc:
            assert_eval_finite([{"eval_loss": float("nan"), "step": 10}])
        assert math.isnan(exc.value.value)

    def test_inf_eval_raises(self) -> None:
        with pytest.raises(NaNEvalError):
            assert_eval_finite([{"eval_loss": float("inf"), "step": 10}])

    def test_last_eval_is_authoritative(self) -> None:
        # Walks from the tail — the last eval entry (NaN here) wins even
        # if an earlier one was finite.
        with pytest.raises(NaNEvalError):
            assert_eval_finite(
                [
                    {"eval_loss": 1.5, "step": 10},
                    {"eval_loss": float("nan"), "step": 20},
                ]
            )

    def test_only_walks_until_first_eval_entry(self) -> None:
        # If the final eval is finite, the gate is satisfied — earlier
        # NaN eval entries don't matter (they were historical).
        assert_eval_finite(
            [
                {"eval_loss": float("nan"), "step": 10},
                {"eval_loss": 1.3, "step": 20},
            ]
        )


class _FakeStore:
    """Minimal StorePath stand-in for commit_version's integration test."""

    def __init__(self, root: Path) -> None:
        self.adapter_versions = root / "versions"
        self.adapter_versions.mkdir(parents=True)
        self._current: Path | None = None

    def adapter_version(self, n: int) -> Path:
        return self.adapter_versions / f"v{n:04d}"

    def set_current_adapter(self, path: Path) -> None:
        self._current = path


class TestCommitVersionRenamesOnNaN:
    def test_rejected_dir_created_and_current_not_flipped(self, tmp_path: Path) -> None:
        store = _FakeStore(tmp_path)

        def writer(pending: Path) -> None:
            # Simulate: save weights (touch a file), then gate fails.
            (pending / "adapter_model.safetensors").write_bytes(b"bogus")
            raise NaNWeightsError(["base_model.lora_A.weight"])

        with pytest.raises(NaNWeightsError):
            commit_version(store, writer)  # type: ignore[arg-type]

        # The rejected dir exists with the saved (bad) weights preserved.
        rejected = tmp_path / "versions" / "v0001-rejected"
        assert rejected.exists()
        assert (rejected / "adapter_model.safetensors").exists()
        # The plain v0001 dir no longer exists (was renamed).
        assert not (tmp_path / "versions" / "v0001").exists()
        # current.txt was never flipped.
        assert store._current is None

    def test_next_version_skips_rejected_name(self, tmp_path: Path) -> None:
        # After a rejected commit, the next allocate_next_version should
        # still pick v0001 (the `-rejected` suffix makes the old one
        # unparseable as a version number).
        from dlm.train.checkpoint_commit import allocate_next_version

        store = _FakeStore(tmp_path)
        (store.adapter_versions / "v0001-rejected").mkdir()
        next_dir = allocate_next_version(store)  # type: ignore[arg-type]
        assert next_dir.name == "v0001"
