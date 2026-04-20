"""Pydantic model for a template's `meta.yaml` sidecar.

One sidecar per template. Kept alongside the `.dlm` body so the pair is
a self-contained unit that external repos can copy verbatim.

Example:

    name: coding-tutor
    title: Coding tutor (Python, curated)
    domain_tags: [code, python, tutor]
    recommended_base: qwen2.5-coder-1.5b
    expected_steps: 800
    expected_duration:
      "cuda-sm80+": "~5 min"
      "mps":        "~15 min"
      "cpu":        "~2 hr"
    summary: |
      A compact Python-focused Q&A tutor.
    sample_prompts:
      - "What are Python decorators?"
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TemplateMeta(BaseModel):
    """Metadata sidecar for a gallery template.

    Strict validation: unknown keys reject at load time so drift between
    the bundled gallery and this schema surfaces loudly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: str = Field(
        min_length=1,
        description="Stable identifier. Matches the .dlm / meta.yaml filename stem.",
    )
    title: str = Field(min_length=1, description="Short human-readable title.")
    domain_tags: tuple[str, ...] = Field(
        default=(),
        description="Freeform tag list for filtering (`code`, `writing`, etc.).",
    )
    recommended_base: str = Field(
        min_length=1,
        description="Base model key (registry) or `hf:org/name`.",
    )
    expected_steps: int | None = Field(
        default=None,
        ge=1,
        description="Rough step count at the template's defaults, for capacity planning.",
    )
    expected_duration: dict[str, str] = Field(
        default_factory=dict,
        description="Free-form hardware-tier → wall-clock estimate map.",
    )
    summary: str = Field(
        min_length=1,
        description="One-paragraph description; used by `dlm templates list`.",
    )
    sample_prompts: tuple[str, ...] = Field(
        default=(),
        description="Prompts to try after `dlm train`.",
    )
