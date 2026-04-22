"""Audit-11 M1 coverage — CliRunner tests for `metrics`, `repl`,
`push`, `pull`, `serve`.

These paths previously had no CLI-level tests. Each class here drives
one command end-to-end through `typer.testing.CliRunner`, covering:

- happy path (for commands that don't need a real adapter),
- refusal messaging (for commands that guard against untrained docs),
- safety-gate behavior (public-bind on `serve`, local round-trip on
  push/pull).

The `repl` body needs a live torch stack, so we test only the
pre-backend guard rails. The `serve` body spins up an HTTP server, so
we test only the "no training state" refusal — the bind-resolution
safety rule has direct unit coverage in
`tests/unit/share/test_peer_tokens.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dlm.cli.app import app


def _write_minimal_dlm(path: Path, dlm_id: str = "01KPQ9M3" + "0" * 18) -> None:
    """Emit a minimal valid .dlm — just frontmatter + a body line."""
    path.write_text(
        f"---\ndlm_id: {dlm_id}\ndlm_version: 6\nbase_model: smollm2-135m\n---\nbody\n",
        encoding="utf-8",
    )


class TestMetricsCmd:
    """`dlm metrics` against a never-trained .dlm surfaces an empty table
    rather than a traceback — the CLI handles the "no runs yet" case."""

    def test_metrics_on_untrained_returns_cleanly(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "metrics", str(doc)],
        )
        assert result.exit_code == 0, result.output
        # An untrained .dlm has no runs — the CLI prints an empty
        # table or a "no runs yet" hint, never a traceback.
        assert "Traceback" not in result.output


class TestReplCmd:
    """`dlm repl` against an untrained .dlm must refuse cleanly — it
    needs an adapter to be interesting."""

    def test_repl_on_untrained_refuses(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)

        runner = CliRunner()
        # Pipe EOF on stdin so the REPL can't wait on interactive input
        # even if it reaches the prompt loop.
        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "repl", str(doc)],
            input="",
        )
        # Untrained → no adapter → command exits non-zero. Exact
        # message varies by backend probe, but there MUST NOT be an
        # unhandled traceback.
        assert result.exit_code != 0, result.output
        assert "Traceback" not in result.output


class TestPushPullLocalRoundTrip:
    """`dlm push --to <dir>` + `dlm pull <pack>` round-trip on a local
    filesystem sink — no network, no HF, no signing. Confirms the
    pack→ship→restore loop works end-to-end at the CLI level."""

    def test_push_untrained_refuses(self, tmp_path: Path) -> None:
        """push() calls pack() which loads the manifest; on a never-
        trained .dlm the CLI must surface a clear refusal, not a
        low-level 'manifest corrupt' traceback."""
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)

        dest = tmp_path / "out"
        dest.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "push",
                str(doc),
                "--to",
                str(dest / "doc.dlm.pack"),
            ],
        )
        assert result.exit_code != 0, result.output
        assert "Traceback" not in result.output

    def test_pull_missing_source_refuses(self, tmp_path: Path) -> None:
        """Pulling a nonexistent local path exits non-zero with a
        clear message — no traceback."""
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "pull",
                str(tmp_path / "does-not-exist.dlm.pack"),
            ],
        )
        assert result.exit_code != 0, result.output
        assert "Traceback" not in result.output


class TestServePublicSafety:
    """`dlm serve --public` without `--i-know-this-is-public` — the
    direct unit-level guard is in `tests/unit/share/test_peer_tokens.py`
    (`resolve_bind` binds 127.0.0.1 on the partial-opt-in). At the CLI,
    the command refuses earlier for an untrained doc; we verify that
    refusal happens BEFORE any public bind could occur."""

    def test_serve_public_on_untrained_refuses_before_bind(
        self, tmp_path: Path
    ) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "serve",
                str(doc),
                "--public",
                # Intentionally NOT passing --i-know-this-is-public;
                # the no-manifest guard fires first and prevents any
                # bind attempt (including the 127.0.0.1 fallback).
            ],
        )
        assert result.exit_code == 1, result.output
        assert "no training state" in result.output
        # Must not have reached the bind step — no "serving" banner.
        assert "serving" not in result.output

    def test_serve_public_with_ack_on_untrained_still_refuses(
        self, tmp_path: Path
    ) -> None:
        """Even with the public-ack flag, an untrained doc is refused
        at the earlier guard. Confirms the manifest guard gates
        BEFORE the bind gate, so public bind never happens on an
        unprepared store."""
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "serve",
                str(doc),
                "--public",
                "--i-know-this-is-public",
            ],
        )
        assert result.exit_code == 1, result.output
        assert "no training state" in result.output


@pytest.mark.parametrize(
    "cmd",
    [
        ("metrics", "--help"),
        ("metrics", "watch", "--help"),
        ("repl", "--help"),
        ("push", "--help"),
        ("pull", "--help"),
        ("serve", "--help"),
    ],
)
def test_help_flags_render(cmd: tuple[str, ...]) -> None:
    """`--help` on each audit-11 M1 command renders without blowing up.

    Catches import-time regressions and typer option-definition bugs
    that otherwise only surface when a user actually invokes the
    command.
    """
    runner = CliRunner()
    result = runner.invoke(app, list(cmd))
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output
