"""Dialect → Go `text/template` registry.

One `.gotmpl` file per chat-template dialect, shipped beside this
module. The registry also records per-dialect:

- **Default stops** — token strings that Ollama must refuse to
  generate past. Missing stops cause runaway generation (findings §9).
- **Default params** — `temperature` + `top_p` tuned per family.

The templates themselves are Go `text/template` syntax: `{{.Role}}`,
`{{range .Messages}}`, `{{if .System}}`, etc. Ollama renders them
at inference time. The round-trip test (`tests/unit/export/ollama/
test_template_registry_roundtrip.py`) verifies each template
produces token-identical output to the base model's Jinja reference
on a fixed message-set matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

from dlm.export.ollama.errors import TemplateRegistryError

Dialect = Literal["chatml", "llama3", "phi3", "mistral"]

_TEMPLATES_DIR: Final[Path] = Path(__file__).resolve().parent / "templates"


@dataclass(frozen=True)
class DialectTemplate:
    """One registry row: a dialect's Go template + param defaults."""

    dialect: Dialect
    template_path: Path
    default_stops: tuple[str, ...]
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    # Added-by-training special tokens go here ONLY if they're
    # dialect-inherent (e.g., chatml's `<|im_end|>`). Per-adapter added
    # tokens (Sprint 07 pad-token fallback) come from the adapter
    # tokenizer at render time.
    extra_stop_hints: tuple[str, ...] = field(default_factory=tuple)

    def read_template(self) -> str:
        """Return the `.gotmpl` file contents verbatim."""
        if not self.template_path.is_file():
            raise TemplateRegistryError(
                f"template file missing for {self.dialect!r}: {self.template_path}"
            )
        return self.template_path.read_text(encoding="utf-8")


_REGISTRY: Final[dict[Dialect, DialectTemplate]] = {
    "chatml": DialectTemplate(
        dialect="chatml",
        template_path=_TEMPLATES_DIR / "chatml.gotmpl",
        default_stops=("<|im_end|>", "<|endoftext|>"),
    ),
    "llama3": DialectTemplate(
        dialect="llama3",
        template_path=_TEMPLATES_DIR / "llama3.gotmpl",
        default_stops=("<|eot_id|>", "<|end_of_text|>"),
    ),
    "phi3": DialectTemplate(
        dialect="phi3",
        template_path=_TEMPLATES_DIR / "phi3.gotmpl",
        default_stops=("<|end|>", "<|endoftext|>"),
    ),
    "mistral": DialectTemplate(
        dialect="mistral",
        template_path=_TEMPLATES_DIR / "mistral.gotmpl",
        # Mistral's stop sequences are v2's; older models use EOS only.
        default_stops=("</s>", "[/INST]"),
    ),
}


def registered_dialects() -> tuple[Dialect, ...]:
    """Tuple of all shipped dialects. Useful for CLI help + parametrize."""
    return tuple(_REGISTRY.keys())


def get_template(dialect: str) -> DialectTemplate:
    """Return the registry row for `dialect` or raise.

    `dialect` is typed as `str` at the edge so callers from pydantic
    `Literal` fields can pass through; we validate against the
    registry keys.
    """
    if dialect not in _REGISTRY:
        raise TemplateRegistryError(
            f"unknown template dialect {dialect!r}; "
            f"registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[dialect]


def load_template_text(dialect: str) -> str:
    """Shortcut: `get_template(dialect).read_template()`."""
    return get_template(dialect).read_template()
