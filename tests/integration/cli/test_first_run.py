"""First-run DoD (audit F22) — init/show/doctor succeed without network or ollama.

Sprint 13 spec §DoD: "a regression test installs a fresh venv with no
`ollama` on PATH and no network reachability and asserts `dlm init
mydoc.dlm`, `dlm show mydoc.dlm`, and `dlm doctor` all succeed; no
`ollama`/HF calls are attempted".

Enforced here without a fresh venv — by running the three commands
under a scrubbed env:

- `PATH` stripped of any `ollama` binary (via monkeypatch).
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` forced (already autouse
  in `tests/conftest.py`).
- A tmp `DLM_HOME` to keep state isolated.

The real iptables-based sandbox belongs to the CI `no-network-sandbox`
job; this test gates regression at the command level.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dlm.cli.app import app


@pytest.fixture
def _scrubbed_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ollama from PATH + point DLM_HOME at a tmp sandbox."""
    home = tmp_path / "dlm-home"
    monkeypatch.setenv("DLM_HOME", str(home))
    # Put a PATH that explicitly doesn't have /usr/local/bin or
    # /opt/homebrew/bin where ollama commonly lives.
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))


@pytest.mark.usefixtures("_scrubbed_env")
class TestFirstRun:
    def test_init_then_show_then_doctor(self, tmp_path: Path) -> None:
        runner = CliRunner()
        doc = tmp_path / "mydoc.dlm"

        # init
        r = runner.invoke(app, ["init", str(doc), "--base", "smollm2-135m"])
        assert r.exit_code == 0, r.output
        assert doc.exists()

        # show (uninitialized store path — no `dlm train` run)
        r = runner.invoke(app, ["show", str(doc)])
        assert r.exit_code == 0, r.output

        r = runner.invoke(app, ["show", str(doc), "--json"])
        assert r.exit_code == 0, r.output
        payload = json.loads(r.output)
        assert payload["store_initialized"] is False

        # doctor
        r = runner.invoke(app, ["doctor", "--json"])
        assert r.exit_code == 0, r.output
        doctor_payload = json.loads(r.output)
        # Whatever fields the doctor emits, at least one MUST be present.
        assert doctor_payload  # non-empty
