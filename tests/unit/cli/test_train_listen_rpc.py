"""Flag-validation tests for `dlm train --listen-rpc`.

These exercise the early-refusal gates only — the full RPC lifecycle
inside a training loop is covered by live-socket integration. Here we
confirm the flag combinations that should refuse before any model
download happens do so with the right exit codes and explanatory
messages.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app

_FRONTMATTER = (
    "---\ndlm_id: 01KPQRPCTEST00000000000000\ndlm_version: 7\nbase_model: smollm2-135m\n---\n"
)


def _write_dlm(path: Path) -> None:
    path.write_text(_FRONTMATTER + "prose\n", encoding="utf-8")


class TestListenRpcGates:
    def test_missing_drain_cadence_refused(self, tmp_path: Path, monkeypatch) -> None:
        doc = tmp_path / "doc.dlm"
        _write_dlm(doc)
        monkeypatch.setenv("DLM_PROBE_TOKEN", "t")

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--home", str(tmp_path), "train", str(doc), "--listen-rpc", "127.0.0.1:7429"],
        )
        assert result.exit_code == 2, result.output
        assert "--watch or --max-cycles" in result.output

    def test_missing_token_refused(self, tmp_path: Path, monkeypatch) -> None:
        doc = tmp_path / "doc.dlm"
        _write_dlm(doc)
        monkeypatch.delenv("DLM_PROBE_TOKEN", raising=False)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path),
                "train",
                str(doc),
                "--listen-rpc",
                "127.0.0.1:7429",
                "--max-cycles",
                "3",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "DLM_PROBE_TOKEN" in result.output

    def test_malformed_host_port_refused(self, tmp_path: Path, monkeypatch) -> None:
        doc = tmp_path / "doc.dlm"
        _write_dlm(doc)
        monkeypatch.setenv("DLM_PROBE_TOKEN", "t")

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path),
                "train",
                str(doc),
                "--listen-rpc",
                "noport",
                "--max-cycles",
                "3",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "host:port" in result.output

    def test_non_int_port_refused(self, tmp_path: Path, monkeypatch) -> None:
        doc = tmp_path / "doc.dlm"
        _write_dlm(doc)
        monkeypatch.setenv("DLM_PROBE_TOKEN", "t")

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path),
                "train",
                str(doc),
                "--listen-rpc",
                "127.0.0.1:not-a-port",
                "--max-cycles",
                "3",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "integer" in result.output
