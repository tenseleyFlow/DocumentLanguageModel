"""Direct coverage for top-level CLI app wiring."""

from __future__ import annotations

import logging
import runpy
from unittest.mock import patch

import pytest
import typer

import dlm.cli.app as cli_app


def test_disable_third_party_telemetry_sets_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("TRANSFORMERS_NO_ADVISORY_WARNINGS", raising=False)

    cli_app._disable_third_party_telemetry()

    assert cli_app.os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert cli_app.os.environ["DO_NOT_TRACK"] == "1"
    assert cli_app.os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] == "1"


def test_version_callback_exits_when_requested(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(typer.Exit) as excinfo:
        cli_app._version_callback(True)

    assert excinfo.value.exit_code == 0
    assert "dlm " in capsys.readouterr().out


def test_version_callback_is_noop_when_flag_is_false(capsys: pytest.CaptureFixture[str]) -> None:
    cli_app._version_callback(False)
    assert capsys.readouterr().out == ""


def test_root_sets_home_and_debug_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DLM_HOME", raising=False)

    with patch("logging.basicConfig") as basic_config:
        cli_app._root(version=False, home="/tmp/dlm-home", verbose=True, quiet=False)

    assert cli_app.os.environ["DLM_HOME"] == "/tmp/dlm-home"
    basic_config.assert_called_once_with(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        force=True,
    )


def test_root_sets_warning_logging_for_quiet_mode() -> None:
    with patch("logging.basicConfig") as basic_config:
        cli_app._root(version=False, home=None, verbose=False, quiet=True)

    basic_config.assert_called_once_with(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        force=True,
    )


def test_root_rejects_verbose_and_quiet_together() -> None:
    with pytest.raises(typer.BadParameter, match="mutually exclusive"):
        cli_app._root(version=False, home=None, verbose=True, quiet=True)


def test_main_routes_through_reporter_and_exits() -> None:
    with (
        patch("dlm.cli.reporter.run_with_reporter", return_value=7) as run_with_reporter,
        patch("sys.exit", side_effect=SystemExit(7)) as sys_exit,
        pytest.raises(SystemExit) as excinfo,
    ):
        cli_app.main()

    assert excinfo.value.code == 7
    run_with_reporter.assert_called_once_with(cli_app.app)
    sys_exit.assert_called_once_with(7)


def test_module_main_guard_invokes_main() -> None:
    with (
        patch("dlm.cli.reporter.run_with_reporter", return_value=3),
        patch("sys.exit", side_effect=SystemExit(3)),
        pytest.raises(SystemExit) as excinfo,
    ):
        runpy.run_module("dlm.cli.app", run_name="__main__")

    assert excinfo.value.code == 3
