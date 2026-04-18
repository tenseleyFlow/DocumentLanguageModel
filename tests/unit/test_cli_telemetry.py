"""Audit F13: the CLI entry point must set telemetry-off env vars before any
downstream imports. We test this by spawning a fresh subprocess (so we get a
clean env) and asserting the vars are set after `dlm --version` returns.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_cli_entry_forces_telemetry_off_env_vars() -> None:
    probe = textwrap.dedent(
        """\
        import os
        # Intentionally unset — we want to see if `import dlm.cli.app` sets them.
        for v in ("HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK",
                  "TRANSFORMERS_NO_ADVISORY_WARNINGS"):
            os.environ.pop(v, None)
        import dlm.cli.app  # noqa: F401
        assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
        assert os.environ["DO_NOT_TRACK"] == "1"
        assert os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] == "1"
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_user_preset_telemetry_vars_are_respected() -> None:
    """If a user has explicitly set one of these vars to "0", we must NOT
    overwrite them — `setdefault` semantics.
    """
    probe = textwrap.dedent(
        """\
        import os
        os.environ["DO_NOT_TRACK"] = "0"
        import dlm.cli.app  # noqa: F401
        assert os.environ["DO_NOT_TRACK"] == "0"
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
