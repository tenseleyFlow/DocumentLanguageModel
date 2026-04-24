"""Prompt templates for Sprint 43 synthetic instruction generation."""

from __future__ import annotations

from typing import Final, Literal

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, ConfigDict, Field, model_validator

SynthStrategy = Literal["extraction", "expansion", "both"]
PromptParserKind = Literal["json_list", "numbered_list"]


class SynthPromptTemplate(BaseModel):
    """One shipped or user-supplied synth prompt template."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    system_prompt: str = Field(..., min_length=1)
    user_template: str = Field(..., min_length=1)
    output_parser: PromptParserKind = "json_list"

    @model_validator(mode="after")
    def _template_mentions_required_vars(self) -> SynthPromptTemplate:
        missing: list[str] = []
        if "{{ prose }}" not in self.user_template:
            missing.append("prose")
        if "{{ n }}" not in self.user_template:
            missing.append("n")
        if missing:
            raise ValueError(f"user_template must reference required variable(s) {missing!r}")
        return self

    def render_user_prompt(self, *, prose: str, n: int) -> str:
        """Render the user prompt with strict variable handling."""
        template = Template(self.user_template, undefined=StrictUndefined)
        return template.render(prose=prose, n=n).strip()


DEFAULT_PROMPT_TEMPLATES: Final[dict[Literal["extraction", "expansion"], SynthPromptTemplate]] = {
    "extraction": SynthPromptTemplate(
        system_prompt=(
            "You generate high-quality single-turn instruction data from prose. "
            "Only write questions that the prose directly answers. Return a JSON "
            "list of objects with keys `question` and `answer`."
        ),
        user_template=(
            "Given the prose below, generate {{ n }} factual question/answer pairs "
            "whose answers are explicitly supported by the prose.\n\n"
            "Return only a JSON list.\n\n"
            "Prose:\n{{ prose }}"
        ),
        output_parser="json_list",
    ),
    "expansion": SynthPromptTemplate(
        system_prompt=(
            "You generate high-quality single-turn instruction data from prose. "
            "Write curious but grounded follow-up questions a reader might ask, "
            "and answer them as helpfully as possible. Return a JSON list of "
            "objects with keys `question` and `answer`."
        ),
        user_template=(
            "Given the prose below, generate {{ n }} question/answer pairs that "
            "expand on the material in a useful way without contradicting it.\n\n"
            "Return only a JSON list.\n\n"
            "Prose:\n{{ prose }}"
        ),
        output_parser="json_list",
    ),
}


def get_prompt_template(strategy: Literal["extraction", "expansion"]) -> SynthPromptTemplate:
    """Return the shipped prompt template for one synth strategy."""
    return DEFAULT_PROMPT_TEMPLATES[strategy]
