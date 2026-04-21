"""Schema-level tests for SourceDirective + TrainingConfig fields.

These are pure pydantic round-trips — no filesystem, no expansion.
Validation rules: extra="forbid", frozen=True, ge/min_length bounds.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlm.directives.schema import DlmTrainingConfig
from dlm.doc.schema import CURRENT_SCHEMA_VERSION, SourceDirective, TrainingConfig


def test_current_schema_at_least_v8() -> None:
    # Directive schema landed at v6; this file pins that the CURRENT
    # version never regresses below it. Strict-equality pinning per
    # schema bump would make every future sprint touch this file; the
    # lower-bound assertion catches regressions without the churn.
    assert CURRENT_SCHEMA_VERSION >= 8


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


# ---- DlmTrainingConfig -----------------------------------------------------


def test_dlm_training_config_defaults() -> None:
    cfg = DlmTrainingConfig()
    assert cfg.dlm_training_version == 1
    assert cfg.include == ()
    assert cfg.exclude == ()
    assert cfg.exclude_defaults is True
    assert cfg.metadata == {}


def test_dlm_training_config_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        DlmTrainingConfig(unknown_key="x")  # type: ignore[call-arg]


def test_dlm_training_config_rejects_wrong_version() -> None:
    with pytest.raises(ValidationError):
        DlmTrainingConfig(dlm_training_version=2)  # type: ignore[arg-type]


def test_dlm_training_config_accepts_full_shape() -> None:
    cfg = DlmTrainingConfig.model_validate(
        {
            "dlm_training_version": 1,
            "include": ["src/**/*.py"],
            "exclude": ["**/test_*.py"],
            "exclude_defaults": False,
            "metadata": {"language": "python", "domain": "auth"},
        }
    )
    assert cfg.include == ("src/**/*.py",)
    assert cfg.exclude == ("**/test_*.py",)
    assert cfg.exclude_defaults is False
    assert cfg.metadata == {"language": "python", "domain": "auth"}
