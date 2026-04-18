"""Typed-error formatting: file:line:col must render in __str__."""

from __future__ import annotations

from pathlib import Path

from dlm.doc.errors import (
    DlmParseError,
    DlmVersionError,
    FenceError,
    FrontmatterError,
    SchemaValidationError,
)


class TestDlmParseErrorFormatting:
    def test_includes_path_line_col(self) -> None:
        err = DlmParseError("boom", path=Path("a.dlm"), line=3, col=5)
        assert str(err) == "a.dlm:3:5: boom"

    def test_includes_line_only(self) -> None:
        err = DlmParseError("boom", path=Path("a.dlm"), line=3)
        assert str(err) == "a.dlm:3: boom"

    def test_no_path_uses_placeholder(self) -> None:
        err = DlmParseError("boom")
        assert str(err) == "<text>: boom"

    def test_preserves_structured_fields(self) -> None:
        err = DlmParseError("boom", path=Path("/a/b.dlm"), line=7, col=2)
        assert err.message == "boom"
        assert err.path == Path("/a/b.dlm")
        assert err.line == 7
        assert err.col == 2


class TestErrorHierarchy:
    def test_subclasses_are_dlm_parse_errors(self) -> None:
        assert issubclass(FrontmatterError, DlmParseError)
        assert issubclass(SchemaValidationError, DlmParseError)
        assert issubclass(FenceError, DlmParseError)
        assert issubclass(DlmVersionError, DlmParseError)

    def test_all_subclasses_raise_as_value_error(self) -> None:
        # ValueError base makes try/except ValueError work for callers that
        # don't want to discriminate further.
        assert issubclass(DlmParseError, ValueError)
