"""Shared tiny-model-aware planning helpers for integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from dlm.base_models.schema import BaseModelSpec
    from dlm.doc.parser import ParsedDlm
    from dlm.hardware.capabilities import Capabilities
    from dlm.hardware.plan import TrainingPlan


def resolve_spec_and_plan(
    parsed: ParsedDlm,
    *,
    accept_license: bool = False,
    skip_reason: str = "doctor() returned no viable training plan on this host",
) -> tuple[BaseModelSpec, TrainingPlan, Capabilities]:
    """Resolve the base spec and a host-appropriate plan for `parsed`.

    Slow integration tests should not call bare `doctor()` for tiny-model
    fixtures; that falls back to a generic host heuristic and can reject
    CPU-capable tiny bases that should run fine on CI. Mirror the real
    train path instead: feed the parsed training config, the base-model
    parameter count, and the bounded effective context length.
    """

    from dlm.base_models import resolve as resolve_base_model
    from dlm.hardware import doctor

    spec = resolve_base_model(parsed.frontmatter.base_model, accept_license=accept_license)
    doctor_result = doctor(
        training_config=parsed.frontmatter.training,
        base_params=spec.params,
        seq_len=min(parsed.frontmatter.training.sequence_len, spec.effective_context_length),
    )
    plan = doctor_result.plan
    if plan is None:
        pytest.skip(skip_reason)
    return spec, plan, doctor_result.capabilities
