"""Verify the `trl.experimental.orpo` import path we depend on (audit-07 N6).

Sprint 18's status flagged TRL 1.x's demotion of `ORPOTrainer` into
`trl.experimental.orpo` as a fragile surface. This test imports the
symbols we actually need; if TRL re-promotes or drops the path, this
test catches it in unit CI rather than at the first live training run.
"""

from __future__ import annotations

import importlib

import pytest


def test_trl_experimental_orpo_imports() -> None:
    try:
        import trl  # noqa: F401
    except ImportError:
        pytest.skip("trl not installed")

    mod = importlib.import_module("trl.experimental.orpo")
    assert hasattr(mod, "ORPOTrainer"), (
        "trl.experimental.orpo.ORPOTrainer is missing — the import path "
        "has changed. Update src/dlm/train/preference/orpo_trainer.py."
    )
    assert hasattr(mod, "ORPOConfig"), (
        "trl.experimental.orpo.ORPOConfig is missing — the import path "
        "has changed. Update src/dlm/train/preference/orpo_trainer.py."
    )


def test_orpo_trainer_heavy_path_imports() -> None:
    """`orpo_trainer.build_orpo_trainer` defers its TRL imports; make
    sure the surface-level import from our module works so the pragma
    doesn't hide a stale symbol reference."""
    try:
        import trl  # noqa: F401
    except ImportError:
        pytest.skip("trl not installed")

    from dlm.train.preference.orpo_trainer import (
        build_orpo_config_kwargs,
        build_orpo_trainer,
    )

    assert callable(build_orpo_config_kwargs)
    assert callable(build_orpo_trainer)
