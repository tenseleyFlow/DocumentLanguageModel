"""Prompt template substrate for Sprint 43 synth generation."""

from __future__ import annotations

from typing import Literal, cast

import pytest

from dlm.synth import DEFAULT_PROMPT_TEMPLATES, SynthPromptTemplate, get_prompt_template


def test_shipped_prompt_templates_cover_both_strategies() -> None:
    assert set(DEFAULT_PROMPT_TEMPLATES) == {"extraction", "expansion"}


@pytest.mark.parametrize("strategy", ["extraction", "expansion"])
def test_get_prompt_template_returns_shipped_template(strategy: str) -> None:
    typed_strategy = cast(Literal["extraction", "expansion"], strategy)
    template = get_prompt_template(typed_strategy)
    assert template is DEFAULT_PROMPT_TEMPLATES[typed_strategy]
    assert template.output_parser == "json_list"


def test_render_user_prompt_injects_required_values() -> None:
    template = get_prompt_template("extraction")
    rendered = template.render_user_prompt(prose="alpha beta", n=3)
    assert "alpha beta" in rendered
    assert "3" in rendered


@pytest.mark.parametrize(
    ("template", "missing"),
    [
        ("Missing one variable: {{ prose }}", "['n']"),
        ("Missing one variable: {{ n }}", "['prose']"),
        ("Missing both variables.", "['prose', 'n']"),
    ],
)
def test_user_template_must_reference_required_variables(
    template: str,
    missing: str,
) -> None:
    with pytest.raises(ValueError, match=missing):
        SynthPromptTemplate(
            system_prompt="hi",
            user_template=template,
        )
