"""Synthesize `.dlm` documents for tests.

The factory produces text that matches the Sprint 03 schema:

    ---
    dlm_id: <ULID>
    dlm_version: 1
    base_model: <key or hf:org/name>
    training: { ... }
    export: { ... }
    ---

    # body

    prose...
    ::instruction::
    ### Q / ### A pairs
    ::preference::
    ### Prompt / ### Chosen / ### Rejected

Shape is stable; Sprint 03's parser will round-trip these exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import ulid

SectionType = Literal["prose", "instruction", "preference"]


@dataclass(frozen=True)
class ProseSection:
    content: str
    type: SectionType = "prose"


@dataclass(frozen=True)
class QA:
    question: str
    answer: str


@dataclass(frozen=True)
class InstructionSection:
    pairs: list[QA]
    type: SectionType = "instruction"


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str


@dataclass(frozen=True)
class PreferenceSection:
    pairs: list[PreferencePair]
    type: SectionType = "preference"


Section = ProseSection | InstructionSection | PreferenceSection


@dataclass(frozen=True)
class DlmSpec:
    """Builder spec. `make_dlm(**asdict(spec))` is the canonical call form."""

    sections: list[Section] = field(default_factory=list)
    base_model: str = "smollm2-135m"
    dlm_id: str | None = None
    dlm_version: int = 1
    system_prompt: str | None = None
    training_overrides: dict[str, object] = field(default_factory=dict)
    export_overrides: dict[str, object] = field(default_factory=dict)


# --- API ----------------------------------------------------------------------


def make_dlm(
    sections: list[Section] | None = None,
    *,
    base_model: str = "smollm2-135m",
    dlm_id: str | None = None,
    dlm_version: int = 1,
    system_prompt: str | None = None,
    training_overrides: dict[str, object] | None = None,
    export_overrides: dict[str, object] | None = None,
) -> str:
    """Produce a `.dlm` text blob.

    Deterministic when `dlm_id` is supplied. Without it, a fresh ULID is
    generated; tests that need reproducibility across runs must pass one.
    """
    spec = DlmSpec(
        sections=list(sections or _default_sections()),
        base_model=base_model,
        dlm_id=dlm_id or str(ulid.ULID()),
        dlm_version=dlm_version,
        system_prompt=system_prompt,
        training_overrides=dict(training_overrides or {}),
        export_overrides=dict(export_overrides or {}),
    )
    return _render(spec)


# --- Convenience constructors -------------------------------------------------


def prose(content: str) -> ProseSection:
    return ProseSection(content=content)


def instruction(*pairs: tuple[str, str]) -> InstructionSection:
    return InstructionSection(pairs=[QA(q, a) for q, a in pairs])


def preference(*triples: tuple[str, str, str]) -> PreferenceSection:
    return PreferenceSection(
        pairs=[PreferencePair(prompt=p, chosen=c, rejected=r) for p, c, r in triples]
    )


# --- Rendering ----------------------------------------------------------------


def _default_sections() -> list[Section]:
    return [
        prose("# Sample\n\nA minimal starter document for tests.\n"),
        instruction(
            ("What is 2 + 2?", "4."),
            ("What is the capital of France?", "Paris."),
        ),
    ]


def _render(spec: DlmSpec) -> str:
    parts: list[str] = []
    parts.append(_render_frontmatter(spec))
    parts.append("\n")
    for i, section in enumerate(spec.sections):
        if i > 0:
            parts.append("\n")
        parts.append(_render_section(section))
    # Trailing newline to match editor conventions.
    rendered = "".join(parts)
    if not rendered.endswith("\n"):
        rendered += "\n"
    return rendered


def _render_frontmatter(spec: DlmSpec) -> str:
    training = _default_training() | spec.training_overrides
    export = _default_export() | spec.export_overrides
    lines = [
        "---",
        f"dlm_id: {spec.dlm_id}",
        f"dlm_version: {spec.dlm_version}",
        f"base_model: {spec.base_model}",
        "training:",
        *_render_mapping(training, indent=2),
        "export:",
        *_render_mapping(export, indent=2),
    ]
    if spec.system_prompt is not None:
        lines.append("system_prompt: |")
        lines.extend(f"  {line}" for line in spec.system_prompt.splitlines())
    lines.append("---")
    return "\n".join(lines) + "\n"


def _default_training() -> dict[str, object]:
    return {
        "adapter": "lora",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "sequence_len": 512,
        "learning_rate": 2e-4,
        "num_epochs": 1,
        "optimizer": "adamw_torch",
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.1,
        "seed": 42,
    }


def _default_export() -> dict[str, object]:
    return {"default_quant": "Q4_K_M"}


def _render_mapping(mapping: dict[str, object], *, indent: int) -> list[str]:
    pad = " " * indent
    lines: list[str] = []
    for key, value in mapping.items():
        lines.append(f"{pad}{key}: {_scalar(value)}")
    return lines


def _scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        # Conservative: quote if contains whitespace or YAML-special chars.
        if any(ch in value for ch in " :#"):
            escaped = value.replace('"', '\\"')
            return f'"{escaped}"'
        return value
    return str(value)


def _render_section(section: Section) -> str:
    if isinstance(section, ProseSection):
        body = section.content
        if not body.endswith("\n"):
            body += "\n"
        return body
    if isinstance(section, InstructionSection):
        lines = ["::instruction::", ""]
        for i, qa in enumerate(section.pairs):
            if i > 0:
                lines.append("")
            lines.extend(["### Q", qa.question, "", "### A", qa.answer])
        return "\n".join(lines) + "\n"
    if isinstance(section, PreferenceSection):
        lines = ["::preference::", ""]
        for i, triple in enumerate(section.pairs):
            if i > 0:
                lines.append("")
            lines.extend(
                [
                    "### Prompt",
                    triple.prompt,
                    "",
                    "### Chosen",
                    triple.chosen,
                    "",
                    "### Rejected",
                    triple.rejected,
                ]
            )
        return "\n".join(lines) + "\n"
    raise TypeError(f"unknown section type: {type(section)!r}")
