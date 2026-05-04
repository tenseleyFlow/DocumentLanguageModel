"""CLI tests for ``dlm unpack`` argument forwarding (M13.2).

The audit caught that ``CLAUDE.md`` documents ``dlm unpack --home DIR``
but the CLI didn't expose the flag — only ``DLM_HOME=`` env var worked.
These tests assert the flag is wired and forwards correctly to
``dlm.pack.unpacker.unpack``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_app() -> Any:
    from dlm.cli.app import app

    return app


def test_unpack_help_lists_home_flag(cli_app: Any) -> None:
    result = CliRunner().invoke(cli_app, ["unpack", "--help"])
    assert result.exit_code == 0
    out = (result.stdout or "") + (result.stderr or "")
    # Strip ANSI to handle typer/rich coloured rendering on CI.
    import re

    plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", out)
    assert "--home" in plain
    assert "DLM_HOME" in plain


def test_unpack_home_flag_forwards_to_unpacker(
    cli_app: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Asserts ``--home`` propagates as the ``home`` kwarg to
    ``dlm.pack.unpacker.unpack``. Mocks the underlying call so we don't
    need a synthesized pack — the wiring is what M13.2 is about."""
    captured: dict[str, Any] = {}

    def _fake_unpack(path: Path, **kwargs: Any) -> Any:
        captured["path"] = path
        captured.update(kwargs)

        class _Result:
            dlm_id = "01TEST"
            dlm_path = tmp_path / "doc.dlm"
            store_path = tmp_path / "store" / "01TEST"
            applied_migrations: tuple[int, ...] = ()
            header = type("h", (), {"pack_format_version": 1})()

        # Touch the dlm_path so any subsequent inspection doesn't blow up.
        (tmp_path / "doc.dlm").write_text("---\ndlm_id: 01TEST\n---\n")
        return _Result()

    import dlm.pack.unpacker as unpacker_mod

    monkeypatch.setattr(unpacker_mod, "unpack", _fake_unpack)

    pack = tmp_path / "fake.dlm.pack"
    pack.write_bytes(b"")  # CLI doesn't read this directly — _fake_unpack short-circuits.

    home = tmp_path / "alt-home"
    result = CliRunner().invoke(cli_app, ["unpack", str(pack), "--home", str(home)])

    assert result.exit_code == 0, (result.stdout or "") + (result.stderr or "")
    assert captured["home"] == home
    assert captured["force"] is False
