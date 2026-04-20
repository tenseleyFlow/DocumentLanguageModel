"""TemplateMeta schema — round-trip, strict validation, defaults."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlm.templates.schema import TemplateMeta


def _minimal_payload() -> dict[str, object]:
    return {
        "name": "demo",
        "title": "Demo template",
        "recommended_base": "qwen2.5-1.5b",
        "summary": "A demo.",
    }


def test_minimal_payload_validates() -> None:
    meta = TemplateMeta.model_validate(_minimal_payload())
    assert meta.name == "demo"
    assert meta.domain_tags == ()
    assert meta.expected_steps is None
    assert meta.expected_duration == {}
    assert meta.sample_prompts == ()


def test_all_fields_populated() -> None:
    payload = {
        **_minimal_payload(),
        "domain_tags": ["writing", "dev"],
        "expected_steps": 400,
        "expected_duration": {"cuda-sm80+": "~3 min", "cpu": "~1 hr"},
        "sample_prompts": ["prompt one", "prompt two"],
    }
    meta = TemplateMeta.model_validate(payload)
    assert meta.domain_tags == ("writing", "dev")
    assert meta.expected_steps == 400
    assert meta.expected_duration == {"cuda-sm80+": "~3 min", "cpu": "~1 hr"}
    assert meta.sample_prompts == ("prompt one", "prompt two")


def test_unknown_keys_reject() -> None:
    payload = {**_minimal_payload(), "mystery_key": "value"}
    with pytest.raises(ValidationError):
        TemplateMeta.model_validate(payload)


def test_empty_strings_reject() -> None:
    with pytest.raises(ValidationError):
        TemplateMeta.model_validate({**_minimal_payload(), "title": ""})
    with pytest.raises(ValidationError):
        TemplateMeta.model_validate({**_minimal_payload(), "summary": ""})


def test_expected_steps_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        TemplateMeta.model_validate({**_minimal_payload(), "expected_steps": 0})
    with pytest.raises(ValidationError):
        TemplateMeta.model_validate({**_minimal_payload(), "expected_steps": -5})


def test_frozen_instance() -> None:
    meta = TemplateMeta.model_validate(_minimal_payload())
    with pytest.raises(ValidationError):
        meta.name = "other"  # type: ignore[misc]
