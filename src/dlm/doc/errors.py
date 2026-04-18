"""Typed parser errors with `file:line:col` location reporting.

All subclasses of `DlmParseError` carry `(path, line, col)` so CLI error
reporting (Sprint 13) can format diagnostics uniformly.
"""

from __future__ import annotations

from pathlib import Path


class DlmParseError(ValueError):
    """Base class for all `.dlm` parse errors.

    `line` is 1-indexed. `col` is 1-indexed or 0 when column info isn't
    meaningful (e.g., schema validation errors on whole values).
    """

    def __init__(
        self,
        message: str,
        *,
        path: Path | None = None,
        line: int = 0,
        col: int = 0,
    ) -> None:
        self.message = message
        self.path = path
        self.line = line
        self.col = col
        super().__init__(self._format())

    def _format(self) -> str:
        where = str(self.path) if self.path is not None else "<text>"
        if self.line and self.col:
            return f"{where}:{self.line}:{self.col}: {self.message}"
        if self.line:
            return f"{where}:{self.line}: {self.message}"
        return f"{where}: {self.message}"


class FrontmatterError(DlmParseError):
    """YAML-level errors in the frontmatter block (missing delimiters, bad YAML)."""


class SchemaValidationError(DlmParseError):
    """Pydantic validation failure against the frontmatter schema.

    Unknown keys, wrong types, out-of-range values all land here.
    """


class DlmVersionError(DlmParseError):
    """`dlm_version` is present but isn't a version this parser implements.

    Sprint 12b's migration framework is the home for promoting older
    documents; this error's message points there.
    """


class FenceError(DlmParseError):
    """Malformed or unknown section fence in the body."""


class InstructionGrammarError(DlmParseError):
    """Malformed `### Q` / `### A` structure inside an instruction section."""


class PreferenceGrammarError(DlmParseError):
    """Malformed `### Prompt` / `### Chosen` / `### Rejected` structure."""
