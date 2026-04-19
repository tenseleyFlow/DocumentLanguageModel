"""`dlm init` CLI — ULID scaffold + license gate + overwrite refusal."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app


def _joined_output(result: object) -> str:
    """Collapse Rich line-wrapping for stable substring assertions."""
    text = getattr(result, "output", "") + getattr(result, "stderr", "")
    return " ".join(text.split())


class TestHappyPath:
    def test_scaffold_parses_back(self, tmp_path: Path) -> None:
        from dlm.doc.parser import parse_file

        runner = CliRunner()
        out = tmp_path / "doc.dlm"
        result = runner.invoke(app, ["init", str(out), "--base", "smollm2-135m"])
        assert result.exit_code == 0, result.output

        parsed = parse_file(out)
        assert parsed.frontmatter.base_model == "smollm2-135m"
        assert parsed.frontmatter.dlm_id
        # Body has prose + instruction scaffolds.
        assert any(s.type.value == "instruction" for s in parsed.sections)

    def test_default_base_is_qwen(self, tmp_path: Path) -> None:
        """Default --base matches the documented Sprint-13 spec."""
        from dlm.doc.parser import parse_file

        runner = CliRunner()
        out = tmp_path / "doc.dlm"
        result = runner.invoke(app, ["init", str(out)])
        assert result.exit_code == 0, result.output
        parsed = parse_file(out)
        assert parsed.frontmatter.base_model == "qwen2.5-1.5b"


class TestOverwriteRefusal:
    def test_existing_file_refused(self, tmp_path: Path) -> None:
        runner = CliRunner()
        out = tmp_path / "doc.dlm"
        out.write_text("prior content\n", encoding="utf-8")

        result = runner.invoke(app, ["init", str(out)])
        assert result.exit_code == 1
        assert "already exists" in _joined_output(result)
        # Content must be untouched.
        assert out.read_text(encoding="utf-8") == "prior content\n"

    def test_force_overwrites(self, tmp_path: Path) -> None:
        runner = CliRunner()
        out = tmp_path / "doc.dlm"
        out.write_text("prior content\n", encoding="utf-8")

        result = runner.invoke(app, ["init", str(out), "--base", "smollm2-135m", "--force"])
        assert result.exit_code == 0, result.output
        # Now looks like a valid scaffold.
        assert out.read_text(encoding="utf-8").startswith("---\n")


class TestLicenseGate:
    def test_gated_base_without_flag_refuses_non_interactive(self, tmp_path: Path) -> None:
        """CliRunner has no TTY; gated resolve should refuse and exit 1."""
        runner = CliRunner()
        out = tmp_path / "doc.dlm"
        result = runner.invoke(app, ["init", str(out), "--base", "llama-3.2-1b"])
        assert result.exit_code == 1
        joined = _joined_output(result)
        assert "--i-accept-license" in joined
        assert not out.exists()

    def test_gated_base_with_flag_succeeds(self, tmp_path: Path) -> None:
        runner = CliRunner()
        out = tmp_path / "doc.dlm"
        result = runner.invoke(
            app,
            ["init", str(out), "--base", "llama-3.2-1b", "--i-accept-license"],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_unknown_base_surfaces_typed_error(self, tmp_path: Path) -> None:
        runner = CliRunner()
        out = tmp_path / "doc.dlm"
        result = runner.invoke(app, ["init", str(out), "--base", "not-a-real-model"])
        assert result.exit_code == 1
        assert "unknown" in _joined_output(result).lower()
        assert not out.exists()


class TestTemplateReserved:
    def test_template_flag_noop_with_note(self, tmp_path: Path) -> None:
        runner = CliRunner()
        out = tmp_path / "doc.dlm"
        result = runner.invoke(
            app,
            ["init", str(out), "--base", "smollm2-135m", "--template", "starter"],
        )
        assert result.exit_code == 0, result.output
        assert "reserved" in _joined_output(result).lower()
