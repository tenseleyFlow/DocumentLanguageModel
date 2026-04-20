"""Preference-phase error hierarchy tests."""

from __future__ import annotations

import pytest

from dlm.train.preference import (
    DpoPhaseError,
    DpoReferenceLoadError,
    NoPreferenceContentError,
    PriorAdapterRequiredError,
)


class TestHierarchy:
    @pytest.mark.parametrize(
        "subclass",
        [NoPreferenceContentError, PriorAdapterRequiredError, DpoReferenceLoadError],
    )
    def test_each_subclass_inherits_base(self, subclass: type[Exception]) -> None:
        assert issubclass(subclass, DpoPhaseError)
        assert issubclass(subclass, Exception)


class TestDpoReferenceLoadError:
    def test_message_includes_path_and_cause(self) -> None:
        err = DpoReferenceLoadError(
            adapter_path="/tmp/adapter_v3",
            cause="missing adapter_config.json",
        )
        msg = str(err)
        assert "/tmp/adapter_v3" in msg
        assert "missing adapter_config.json" in msg
        assert err.adapter_path == "/tmp/adapter_v3"
        assert err.cause == "missing adapter_config.json"
