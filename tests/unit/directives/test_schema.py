"""Schema-level tests for SourceDirective + TrainingConfig fields.

These are pure pydantic round-trips — no filesystem, no expansion.
Validation rules: extra="forbid", frozen=True, ge/min_length bounds.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlm.doc.schema import CURRENT_SCHEMA_VERSION, SourceDirective, TrainingConfig


def test_current_schema_is_v6() -> None:
    assert CURRENT_SCHEMA_VERSION == 6


def test_source_directive_defaults() -> None:
    d = SourceDirective(path="~/code/foo")
    assert d.path == "~/code/foo"
    assert d.include == ("**/*",)
    assert d.exclude == ()
    assert d.max_bytes_per_file is None
    assert d.max_files is None


def test_source_directive_frozen() -> None:
    d = SourceDirective(path="foo")
    with pytest.raises(ValidationError):
        d.path = "bar"  # type: ignore[misc]


def test_source_directive_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        SourceDirective(path="foo", unknown_key="x")  # type: ignore[call-arg]


def test_source_directive_rejects_empty_path() -> None:
    with pytest.raises(ValidationError):
        SourceDirective(path="")


def test_source_directive_rejects_nonpositive_caps() -> None:
    with pytest.raises(ValidationError):
        SourceDirective(path="foo", max_files=0)
    with pytest.raises(ValidationError):
        SourceDirective(path="foo", max_bytes_per_file=0)


def test_training_config_sources_defaults_none() -> None:
    cfg = TrainingConfig()
    assert cfg.sources is None
    assert cfg.sources_policy == "permissive"


def test_training_config_accepts_sources_list() -> None:
    cfg = TrainingConfig(
        sources=(
            SourceDirective(path="src"),
            SourceDirective(path="docs", include=("**/*.md",)),
        ),
        sources_policy="strict",
    )
    assert cfg.sources is not None
    assert len(cfg.sources) == 2
    assert cfg.sources_policy == "strict"


def test_training_config_rejects_bad_policy() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(sources_policy="lenient")  # type: ignore[arg-type]
