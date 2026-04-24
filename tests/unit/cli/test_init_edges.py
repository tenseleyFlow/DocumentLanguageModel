"""Edge coverage for `dlm init` helper paths near the top of cli/commands."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

import dlm.base_models as base_models
import dlm.templates as templates
from dlm.base_models.errors import GatedModelError
from dlm.cli import commands
from dlm.cli.app import app
from dlm.templates.errors import TemplateError


def test_stub_mentions_sprint_and_subject() -> None:
    with pytest.raises(NotImplementedError, match="owned by Sprint 43"):
        commands._stub("43", "dlm synth")


class TestPromptAcceptLicense:
    def test_non_tty_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console = Console(record=True)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        assert commands._prompt_accept_license(console, "llama-3.2-1b", None) is False

    def test_yes_accepts_and_prints_license_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console = Console(record=True)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda: "Yes")

        assert (
            commands._prompt_accept_license(
                console,
                "llama-3.2-1b",
                "https://example.test/license",
            )
            is True
        )
        text = console.export_text()
        assert "requires accepting the upstream license" in text
        assert "https://example.test/license" in text

    def test_eof_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console = Console(record=True)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        def _raise_eof() -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", _raise_eof)

        assert commands._prompt_accept_license(console, "llama-3.2-1b", None) is False


class TestInitTemplateEdges:
    def test_explicit_base_warning_when_template_overrides(self, tmp_path: Path) -> None:
        runner = CliRunner()
        out = tmp_path / "doc.dlm"
        home = tmp_path / "home"

        result = runner.invoke(
            app,
            [
                "--home",
                str(home),
                "init",
                str(out),
                "--base",
                "smollm2-135m",
                "--template",
                "changelog",
            ],
        )

        assert result.exit_code == 0, result.output
        joined = " ".join((result.output + result.stderr).split())
        assert "--base smollm2-135m ignored" in joined
        assert "uses smollm2-360m" in joined

    def test_interactive_acceptance_retries_resolution(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = CliRunner()
        out = tmp_path / "doc.dlm"
        home = tmp_path / "home"
        calls: list[tuple[str, bool, bool]] = []
        spec = SimpleNamespace(key="llama-3.2-1b", revision="rev-1", modality="text")

        def _fake_resolve(
            base: str,
            *,
            accept_license: bool = False,
            skip_export_probes: bool = False,
        ) -> object:
            calls.append((base, accept_license, skip_export_probes))
            if len(calls) == 1:
                raise GatedModelError(base, "https://example.test/license")
            return spec

        monkeypatch.setattr(base_models, "resolve", _fake_resolve)
        monkeypatch.setattr(base_models, "is_gated", lambda spec: False)
        monkeypatch.setattr(commands, "_prompt_accept_license", lambda console, base, url: True)

        result = runner.invoke(
            app,
            ["--home", str(home), "init", str(out), "--base", "llama-3.2-1b"],
        )

        assert result.exit_code == 0, result.output
        assert calls == [
            ("llama-3.2-1b", False, False),
            ("llama-3.2-1b", True, False),
        ]
        assert out.exists()

    def test_template_apply_error_exits_cleanly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = CliRunner()
        out = tmp_path / "doc.dlm"
        home = tmp_path / "home"

        monkeypatch.setattr(
            templates,
            "load_template",
            lambda name: SimpleNamespace(meta=SimpleNamespace(recommended_base="smollm2-135m")),
        )

        def _fake_apply_template(
            name: str,
            target: Path,
            *,
            force: bool = False,
            accept_license: bool = False,
        ) -> object:
            raise TemplateError("template exploded")

        monkeypatch.setattr(templates, "apply_template", _fake_apply_template)

        result = runner.invoke(
            app,
            ["--home", str(home), "init", str(out), "--template", "custom"],
        )

        assert result.exit_code == 1
        assert "template exploded" in result.output
        assert not out.exists()
