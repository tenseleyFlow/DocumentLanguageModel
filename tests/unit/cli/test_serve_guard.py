"""Audit-09 M3: `dlm serve` guards against untrained `.dlm`.

`serve_cmd` used to call `pack()` which loaded the store manifest; on
a never-trained `.dlm`, the manifest file doesn't exist and the error
surfaced as "store manifest corrupt" rather than "train first". The
M3 guard detects the missing manifest early and prints a useful
message.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app


def _write_minimal_dlm(path: Path, dlm_id: str = "01KPQ9M3" + "0" * 18) -> None:
    path.write_text(
        f"---\ndlm_id: {dlm_id}\ndlm_version: 6\nbase_model: smollm2-135m\n---\nbody\n",
        encoding="utf-8",
    )


class TestServeUntrainedGuard:
    def test_serve_on_untrained_dlm_refuses_cleanly(self, tmp_path: Path) -> None:
        """No prior `dlm train` → no manifest → `dlm serve` must exit
        non-zero with a "train first" message, not a manifest-corrupt
        traceback."""
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
            ],
        )
        assert result.exit_code == 1, result.output
        assert "no training state" in result.output
        assert "dlm train" in result.output
        # Must NOT surface the low-level manifest-corrupt error, which
        # is the pre-guard failure mode.
        assert "manifest is corrupt" not in result.output
