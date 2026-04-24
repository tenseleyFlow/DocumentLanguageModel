"""`migrate_file` + `dlm migrate` CLI (Sprint 12b)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dlm.cli.app import app
from dlm.doc import versioned as versioned_module
from dlm.doc.errors import FrontmatterError
from dlm.doc.migrate import _rejoin, migrate_file
from dlm.doc.migrations import MIGRATORS, register
from dlm.doc.schema import DlmFrontmatter

_VALID_ULID = "01JQ7Z0000000000000000000A"

_V1_DOC = f"""---
dlm_id: {_VALID_ULID}
base_model: smollm2-135m
dlm_version: {versioned_module.CURRENT_SCHEMA_VERSION}
---

prose body

::instruction::
### Q
what?
### A
this.
"""


@pytest.fixture
def scratch_registry() -> Iterator[None]:
    saved = dict(MIGRATORS)
    try:
        MIGRATORS.clear()
        yield
    finally:
        MIGRATORS.clear()
        MIGRATORS.update(saved)


@pytest.fixture
def bumped_current(scratch_registry: None) -> Iterator[int]:
    """Pretend CURRENT_SCHEMA_VERSION is one higher than shipped."""
    original = versioned_module.CURRENT_SCHEMA_VERSION
    bumped = original + 1
    # Also patch the migrate + schema module constants. Schema is where
    # the pydantic field validator reads from (audit-07 M6 landed
    # defense-in-depth there); without this patch the validator
    # rejects the bumped version.
    import dlm.doc.migrate as migrate_module
    import dlm.doc.schema as schema_module

    versioned_module.CURRENT_SCHEMA_VERSION = bumped
    migrate_module.CURRENT_SCHEMA_VERSION = bumped
    schema_module.CURRENT_SCHEMA_VERSION = bumped
    try:
        yield bumped
    finally:
        versioned_module.CURRENT_SCHEMA_VERSION = original
        migrate_module.CURRENT_SCHEMA_VERSION = original
        schema_module.CURRENT_SCHEMA_VERSION = original


class TestIdempotent:
    def test_already_current_is_noop(self, tmp_path: Path) -> None:
        doc = tmp_path / "mydoc.dlm"
        doc.write_text(_V1_DOC, encoding="utf-8")
        before = doc.read_text(encoding="utf-8")

        result = migrate_file(doc)
        assert result.applied == []
        assert result.wrote is False
        assert result.backup_path is None
        assert doc.read_text(encoding="utf-8") == before


class TestMigratedWrite:
    def test_write_path_updates_version_and_body(self, tmp_path: Path, bumped_current: int) -> None:
        current = bumped_current - 1  # the version the doc lives at

        @register(from_version=current)
        def _pre_current(raw: dict[str, object]) -> dict[str, object]:
            # Drop an obsolete field that would fail extra="forbid" on the next version.
            return {k: v for k, v in raw.items() if k != "legacy"}

        src = f"""---
dlm_id: {_VALID_ULID}
base_model: smollm2-135m
dlm_version: {current}
legacy: gone
---

## heading

::instruction::
### Q
x
### A
y
"""
        doc = tmp_path / "mydoc.dlm"
        doc.write_text(src, encoding="utf-8")

        result = migrate_file(doc)
        assert result.applied == [current]
        assert result.wrote is True
        assert result.backup_path == tmp_path / "mydoc.dlm.bak"
        assert result.backup_path.read_text(encoding="utf-8") == src

        rewritten = doc.read_text(encoding="utf-8")
        assert f"dlm_version: {bumped_current}" in rewritten
        assert "legacy" not in rewritten
        # Body content survived.
        assert "::instruction::" in rewritten
        assert "### Q" in rewritten
        assert "### A" in rewritten
        # Trailing newline.
        assert rewritten.endswith("\n")

    def test_no_backup_skips_bak_write(self, tmp_path: Path, bumped_current: int) -> None:
        current = bumped_current - 1

        @register(from_version=current)
        def _pre_current(raw: dict[str, object]) -> dict[str, object]:
            return dict(raw)

        doc = tmp_path / "mydoc.dlm"
        doc.write_text(_V1_DOC, encoding="utf-8")

        result = migrate_file(doc, no_backup=True)
        assert result.wrote is True
        assert result.backup_path is None
        assert not (tmp_path / "mydoc.dlm.bak").exists()

    def test_dry_run_reports_without_writing(self, tmp_path: Path, bumped_current: int) -> None:
        current = bumped_current - 1

        @register(from_version=current)
        def _pre_current(raw: dict[str, object]) -> dict[str, object]:
            return dict(raw)

        doc = tmp_path / "mydoc.dlm"
        doc.write_text(_V1_DOC, encoding="utf-8")
        before = doc.read_text(encoding="utf-8")

        result = migrate_file(doc, dry_run=True)
        assert result.applied == [current]
        assert result.wrote is False
        assert result.backup_path is None
        assert doc.read_text(encoding="utf-8") == before


class TestInvalidInputs:
    def test_missing_frontmatter_raises(self, tmp_path: Path) -> None:
        doc = tmp_path / "junk.dlm"
        doc.write_text("no frontmatter here\n", encoding="utf-8")
        with pytest.raises(FrontmatterError):
            migrate_file(doc)

    def test_unclosed_frontmatter_raises(self, tmp_path: Path) -> None:
        doc = tmp_path / "open.dlm"
        doc.write_text("---\ndlm_id: x\n", encoding="utf-8")
        with pytest.raises(FrontmatterError):
            migrate_file(doc)

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        doc = tmp_path / "bad-yaml.dlm"
        doc.write_text("---\ndlm_id: [unclosed\n---\n", encoding="utf-8")
        with pytest.raises(FrontmatterError, match="invalid YAML"):
            migrate_file(doc)

    def test_non_mapping_frontmatter_raises(self, tmp_path: Path) -> None:
        doc = tmp_path / "list-frontmatter.dlm"
        doc.write_text("---\n- just\n- a\n- list\n---\n", encoding="utf-8")
        with pytest.raises(FrontmatterError, match="must be a mapping"):
            migrate_file(doc)


class TestInternals:
    def test_rejoin_without_body_returns_header_only(self) -> None:
        fm = DlmFrontmatter(dlm_id=_VALID_ULID, base_model="smollm2-135m")
        rendered = _rejoin(fm, "\n\n")
        assert rendered.endswith("\n\n")
        assert "::instruction::" not in rendered


# --- CLI surface ---------------------------------------------------------


def _joined_output(result: object) -> str:
    """Normalize Rich's terminal-width wrapping before substring assertions.

    `CliRunner` captures stdout + stderr separately; Rich may also break
    a single logical line across `\\n` depending on console width (on
    narrow CI terminals with long tmp paths, this is routine).
    """
    combined = getattr(result, "output", "") + getattr(result, "stderr", "")
    # Collapse all whitespace (including Rich-inserted line breaks) into
    # single spaces so multi-word substrings match regardless of wrap.
    return " ".join(combined.split())


class TestCli:
    def test_noop_prints_already_current(self, tmp_path: Path) -> None:
        runner = CliRunner()
        doc = tmp_path / "mydoc.dlm"
        doc.write_text(_V1_DOC, encoding="utf-8")

        result = runner.invoke(app, ["migrate", str(doc)])
        assert result.exit_code == 0, result.output
        assert "no migrations needed" in _joined_output(result)

    def test_dry_run_prints_plan_without_writing(self, tmp_path: Path, bumped_current: int) -> None:
        current = bumped_current - 1

        @register(from_version=current)
        def _pre_current(raw: dict[str, object]) -> dict[str, object]:
            return dict(raw)

        runner = CliRunner()
        doc = tmp_path / "mydoc.dlm"
        doc.write_text(_V1_DOC, encoding="utf-8")
        before = doc.read_text(encoding="utf-8")

        result = runner.invoke(app, ["migrate", str(doc), "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "dry-run" in _joined_output(result)
        assert doc.read_text(encoding="utf-8") == before

    def test_bad_file_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        bogus = tmp_path / "bogus.dlm"
        bogus.write_text("no frontmatter\n", encoding="utf-8")
        result = runner.invoke(app, ["migrate", str(bogus)])
        assert result.exit_code != 0
