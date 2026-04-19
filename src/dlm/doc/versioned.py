"""Versioned frontmatter validation — run migrations before Pydantic (Sprint 12b).

The plain `DlmFrontmatter.model_validate(raw)` path refuses unknown
keys (`extra="forbid"`) and would therefore reject *any* v2+ document
with a field added after v1. That turns "forgot to run `dlm migrate`"
into a confusing `SchemaValidationError` instead of an actionable
version-drift message.

This module threads the raw dict through the migration registry first
(bringing a v1 dict up to `CURRENT_SCHEMA_VERSION`) and only then hands
it to Pydantic. Older-but-runnable documents parse cleanly; newer
documents raise a typed `DlmVersionError` pointing at the dlm upgrade.

One entry point:

    validate_versioned(raw: dict, *, path: Path | None) -> DlmFrontmatter

Callers: `dlm.doc.parser._validate_frontmatter` (post-YAML, pre-Pydantic).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from dlm.doc.errors import DlmVersionError, SchemaValidationError
from dlm.doc.migrations.dispatch import apply_pending
from dlm.doc.schema import CURRENT_SCHEMA_VERSION, DlmFrontmatter


def validate_versioned(raw: dict[str, object], *, path: Path | None = None) -> DlmFrontmatter:
    """Dispatch: migrate (if needed) then Pydantic-validate.

    Raises:
        DlmVersionError: `raw["dlm_version"]` is newer than this parser
            supports, or a required intermediate migrator is missing.
        SchemaValidationError: Pydantic validation failed after any
            applicable migrations.
    """
    version = raw.get("dlm_version", 1)
    if isinstance(version, int) and version > CURRENT_SCHEMA_VERSION:
        raise DlmVersionError(
            f"dlm_version {version} is newer than this parser "
            f"({CURRENT_SCHEMA_VERSION}); upgrade dlm or check the source's schema",
            path=path,
            line=2,
        )

    migrated, applied = apply_pending(raw, target_version=CURRENT_SCHEMA_VERSION)
    if applied:
        # The parser's own error path logs migrations as an info line; the
        # migration framework's data-rewrite happens here without
        # side-effects on the source file. `dlm migrate <path>` is the
        # write-path — parsing never mutates the on-disk document.
        pass

    try:
        return DlmFrontmatter.model_validate(migrated)
    except ValidationError as exc:
        raise SchemaValidationError(
            _format_pydantic_error(exc),
            path=path,
            line=2,
        ) from exc


def _format_pydantic_error(exc: ValidationError) -> str:
    """Collapse Pydantic's error-tree into a single-line message."""
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        msg = err.get("msg", "invalid value")
        parts.append(f"{loc}: {msg}")
    return "; ".join(parts) or "validation failed"
