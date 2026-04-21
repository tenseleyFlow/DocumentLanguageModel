"""CLI-level validation of `dlm train --listen-rpc` (Sprint 33.8).

The probe-queue RPC server needs a drain cadence (--watch or
--max-cycles), a bearer token (DLM_PROBE_TOKEN), and a well-formed
host:port. Each of those four preconditions refuses cleanly before
any HF weight download — this test pins the refusal behavior so a
future refactor can't silently remove a gate.

The happy path (starting the actual RPC server) remains a slow
integration target: it needs tiny-model weights + an aiohttp-style
async stack to exercise the drain loop. That's tracked separately.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dlm.cli.app import app


def _joined(result: object) -> str:
    text = getattr(result, "output", "") + getattr(result, "stderr", "")
    return " ".join(text.split())


def _scaffold_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "t.dlm"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_path / "home"),
            "init",
            str(doc),
            "--base",
            "smollm2-135m",
        ],
    )
    assert result.exit_code == 0, result.output
    return doc


class TestListenRpcValidation:
    def test_requires_watch_or_max_cycles(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--listen-rpc without --watch or --max-cycles N refuses with exit 2."""
        doc = _scaffold_doc(tmp_path)
        # Set a token so we don't trip that gate first — we want to
        # hit the drain-cadence refusal specifically.
        monkeypatch.setenv("DLM_PROBE_TOKEN", "test-secret")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "train",
                str(doc),
                "--listen-rpc",
                "127.0.0.1:7429",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "requires --watch or --max-cycles" in _joined(result)

    def test_requires_bearer_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing DLM_PROBE_TOKEN refuses with exit 2 (server can't open unauth)."""
        doc = _scaffold_doc(tmp_path)
        monkeypatch.delenv("DLM_PROBE_TOKEN", raising=False)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "train",
                str(doc),
                "--listen-rpc",
                "127.0.0.1:7429",
                "--max-cycles",
                "1",
            ],
        )
        assert result.exit_code == 2, result.output
        text = _joined(result)
        assert "needs a bearer token" in text
        assert "DLM_PROBE_TOKEN" in text

    def test_empty_bearer_token_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whitespace-only DLM_PROBE_TOKEN is treated the same as unset.

        Users who accidentally export `DLM_PROBE_TOKEN=""` shouldn't get
        a silently-unauth'd server; the check strips before the truthy
        test so a single space also refuses.
        """
        doc = _scaffold_doc(tmp_path)
        monkeypatch.setenv("DLM_PROBE_TOKEN", "   ")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "train",
                str(doc),
                "--listen-rpc",
                "127.0.0.1:7429",
                "--max-cycles",
                "1",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "needs a bearer token" in _joined(result)

    def test_malformed_host_port_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing colon in --listen-rpc value → exit 2 with host:port hint."""
        doc = _scaffold_doc(tmp_path)
        monkeypatch.setenv("DLM_PROBE_TOKEN", "test-secret")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "train",
                str(doc),
                "--listen-rpc",
                "no-colon-here",
                "--max-cycles",
                "1",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "expects host:port" in _joined(result)

    def test_non_integer_port_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-numeric port → exit 2 with a clear error."""
        doc = _scaffold_doc(tmp_path)
        monkeypatch.setenv("DLM_PROBE_TOKEN", "test-secret")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "train",
                str(doc),
                "--listen-rpc",
                "127.0.0.1:not-a-port",
                "--max-cycles",
                "1",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "port must be an integer" in _joined(result)
