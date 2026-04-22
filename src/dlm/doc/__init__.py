"""`.dlm` document parsing, schema, and serialization.

The public surface is:

- `parse_file(path)` / `parse_text(text, path=None)` → `ParsedDlm`
- `serialize(parsed)` → canonical text form
- `DlmFrontmatter`, `TrainingConfig`, `ExportConfig` — Pydantic schema
- `Section`, `SectionType` — body section model
- `DlmParseError` and its subclasses — typed errors with file:line:col

The schema migration framework lives under `dlm.doc.migrations`; this
module's parser raises `DlmVersionError` when a document can't be
promoted to the current schema.
"""

from __future__ import annotations

from dlm.doc.errors import (
    DlmParseError,
    DlmVersionError,
    FenceError,
    FrontmatterError,
    InstructionGrammarError,
    PreferenceGrammarError,
    SchemaValidationError,
)
from dlm.doc.parser import ParsedDlm, parse_file, parse_text
from dlm.doc.schema import DlmFrontmatter, ExportConfig, TrainingConfig
from dlm.doc.sections import Section, SectionType
from dlm.doc.serializer import serialize

__all__ = [
    "DlmFrontmatter",
    "DlmParseError",
    "DlmVersionError",
    "ExportConfig",
    "FenceError",
    "FrontmatterError",
    "InstructionGrammarError",
    "ParsedDlm",
    "PreferenceGrammarError",
    "SchemaValidationError",
    "Section",
    "SectionType",
    "TrainingConfig",
    "parse_file",
    "parse_text",
    "serialize",
]
