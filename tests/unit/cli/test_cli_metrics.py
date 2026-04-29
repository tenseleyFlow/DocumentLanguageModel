"""CLI shape tests for ``dlm metrics`` (M13.3).

The audit hit "Missing argument 'PATH'" on
``dlm metrics PATH --run-id 1``. Click's group dispatch couldn't
disambiguate a positional-then-option from a subcommand-then-args
when the group had both a callback positional AND a registered
subcommand. The fix made ``show`` explicit. These tests assert the
new shape and that both arg orders parse without "Missing argument".
"""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_app() -> Any:
    from dlm.cli.app import app

    return app


def test_metrics_help_lists_show_and_watch_subcommands(cli_app: Any) -> None:
    result = CliRunner().invoke(cli_app, ["metrics", "--help"])
    assert result.exit_code == 0
    out = (result.stdout or "") + (result.stderr or "")
    import re

    plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", out)
    assert "show" in plain
    assert "watch" in plain


def test_metrics_show_with_option_after_positional_parses(
    cli_app: Any,
) -> None:
    """Audit 13 M13.3 regression: ``--run-id`` after the positional now
    parses (it errored before with 'Missing argument PATH'). The actual
    file doesn't exist so we expect a downstream error, but we must NOT
    see the old parser error."""
    result = CliRunner().invoke(
        cli_app, ["metrics", "show", "/nonexistent/path.dlm", "--run-id", "1"]
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "Missing argument" not in combined
    # Allow any non-zero exit (file not found / parse error / etc.) but
    # surface a useful message — not a typer Usage block.
    assert result.exit_code != 0


def test_metrics_show_with_option_before_positional_parses(
    cli_app: Any,
) -> None:
    result = CliRunner().invoke(
        cli_app, ["metrics", "show", "--run-id", "1", "/nonexistent/path.dlm"]
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "Missing argument" not in combined
    assert result.exit_code != 0


def test_metrics_watch_subcommand_unchanged(cli_app: Any) -> None:
    """``dlm metrics watch <path>`` was already unambiguous and still
    parses — the restructure didn't break it."""
    result = CliRunner().invoke(cli_app, ["metrics", "watch", "--help"])
    assert result.exit_code == 0
