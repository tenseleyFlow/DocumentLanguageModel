"""Direct coverage for the `python -m dlm` entrypoint."""

from __future__ import annotations

import runpy


def test_module_entrypoint_invokes_cli_main(monkeypatch) -> None:
    called: list[bool] = []

    monkeypatch.setattr("dlm.cli.app.main", lambda: called.append(True))

    runpy.run_module("dlm", run_name="__main__")

    assert called == [True]
