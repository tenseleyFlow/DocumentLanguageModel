"""B12.1 regression: `dlm train` on a hand-authored `.dlm` (no prior `dlm init`).

The original bug surfaced via Audit 12 E2E-1: an authored `.dlm` with a
fresh ULID frontmatter passed straight to `dlm train` crashes with
`manifest is corrupt: read failed: No such file or directory` after the
trainer creates `<store>/{adapter,logs}/` but before any code writes
the manifest.

The fix in `src/dlm/cli/commands.py:train_cmd` bootstraps a manifest
whenever the store layout exists but `manifest.json` does not (covers
both the auto-scaffold path and this hand-authored path).

This test reproduces the original failure mode end-to-end via
`CliRunner` so the bootstrap can't silently regress.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

pytestmark = [pytest.mark.slow, pytest.mark.online]


def test_fresh_train_without_init_writes_manifest_and_advances(
    tmp_path: Path,
    tiny_model_dir: Path,  # noqa: ARG001 — session-cached download
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DLM_HOME", str(tmp_path / "dlm-home"))

    doc = tmp_path / "fresh.dlm"
    doc.write_text(
        "---\n"
        "dlm_id: 01KQB000FRESHB12B12B12B12B\n"
        "dlm_version: 14\n"
        "base_model: smollm2-135m\n"
        "training:\n"
        "  adapter: lora\n"
        "  lora_r: 4\n"
        "  sequence_len: 256\n"
        "  micro_batch_size: 1\n"
        "  grad_accum: 1\n"
        "  num_epochs: 1\n"
        "---\n"
        "# Fresh\n"
        "\n"
        "::instruction::\n"
        "### Q\n"
        "What is two plus two?\n"
        "\n"
        "### A\n"
        "Four.\n"
        "\n"
        "::instruction::\n"
        "### Q\n"
        "What is the capital of France?\n"
        "\n"
        "### A\n"
        "Paris.\n",
        encoding="utf-8",
    )

    from dlm.cli.app import app
    from dlm.store.paths import for_dlm

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["train", str(doc), "--max-steps", "1", "--fresh"],
        env={**os.environ, "DLM_HOME": str(tmp_path / "dlm-home")},
        catch_exceptions=False,
    )

    assert result.exit_code == 0, f"train failed:\n{result.output}"

    store = for_dlm("01KQB000FRESHB12B12B12B12B")
    assert store.manifest.exists(), (
        "B12.1 regression: manifest.json was not bootstrapped on first train"
    )
    versions_dir = store.adapter / "versions"
    assert versions_dir.exists(), "adapter/versions dir missing"
    written_versions = sorted(p.name for p in versions_dir.iterdir() if p.is_dir())
    assert "v0001" in written_versions, (
        f"expected v0001 adapter after first train, got {written_versions}"
    )
