"""Prompt template substrate for Sprint 43 synth generation."""

from __future__ import annotations

import pytest

from dlm.synth import DEFAULT_PROMPT_TEMPLATES, SynthPromptTemplate, get_prompt_template


def test_shipped_prompt_templates_cover_both_strategies() -> None:
    assert set(DEFAULT_PROMPT_TEMPLATES) == {"extraction", "expansion"}


@pytest.mark.parametrize("strategy", ["extraction", "expansion"])
def test_get_prompt_template_returns_shipped_template(strategy: str) -> None:
    template = get_prompt_template(strategy)  # type: ignore[arg-type]
    assert template is DEFAULT_PROMPT_TEMPLATES[strategy]
    assert template.output_parser == "json_list"


def test_render_user_prompt_injects_required_values() -> None:
    template = get_prompt_template("extraction")
    rendered = template.render_user_prompt(prose="alpha beta", n=3)
    assert "alpha beta" in rendered
    assert "3" in rendered


def test_user_template_must_reference_required_variables() -> None:
    with pytest.raises(ValueError, match="required variable"):
        SynthPromptTemplate(
            system_prompt="hi",
            user_template="Missing one variable: {{ prose }}",
        )
