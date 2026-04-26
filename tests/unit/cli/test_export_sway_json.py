"""Sprint 26 X1 — CLI-flag wiring for `dlm export --emit-sway-json`.

The helper-module unit tests (``write_sway_json`` round-trip + every
error path) live at ``tests/unit/export/test_sway_json.py`` so the
``Coverage gate — src/dlm/export = 100%`` job (which runs only
``tests/unit/export``) sees them. This file owns the CLI-surface
half: the flag is registered, shows up in ``--help``, and carries
the sprint-specified text.
"""

from __future__ import annotations

import re

from typer.testing import CliRunner

from dlm.cli.app import app


class TestExportCliFlagWiring:
    def test_flag_present_in_export_help(self) -> None:
        """``--emit-sway-json`` flag appears in ``dlm export --help``."""
        # Force a wide terminal so typer/Rich don't wrap the long
        # ``--emit-sway-json`` flag across lines (CI's runner has a
        # narrow default that breaks substring asserts).
        runner = CliRunner(env={"COLUMNS": "200", "TERM": "dumb"})
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0, result.output

        # Strip ANSI escapes + collapse whitespace so substring asserts
        # are robust to color codes and any wrap that COLUMNS=200 still
        # leaves in place.
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        plain = re.sub(r"\s+", " ", plain)

        assert "--emit-sway-json" in plain, plain
        assert "sway.yaml" in plain, plain
