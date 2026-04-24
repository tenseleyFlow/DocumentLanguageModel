"""`dlm train --strict-lock` rejects a warn-severity drift (Sprint 15).

Simulates a torch minor-version drift by writing a prior lock with a
different torch pin, then invoking `dlm train --strict-lock`. The
validator upgrades the WARN to ERROR; the CLI exits non-zero.

Unit-test counterpart: `tests/unit/train/test_lock_wiring.py` covers
the trainer.run() call path with a forged lock; this integration
test covers the CLI wiring end-to-end so a future refactor can't
silently drop the flag or mismap its severity.

Marked `@pytest.mark.slow` because resolving the base-model spec and
running doctor.plan() invokes the real hardware probe (but doesn't
actually train — the LockValidationError fires pre-training).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dlm.cli.app import app
from dlm.lock import DlmLock, write_lock
from dlm.lock.schema import CURRENT_LOCK_VERSION

pytestmark = pytest.mark.slow


def _write_minimal_dlm(path: Path) -> None:
    path.write_text(
        "---\n"
        # Crockford base32 (no I/L/O/U); 26 chars total.
        "dlm_id: 01HRSTR1CT" + "0" * 16 + "\n"
        "base_model: smollm2-135m\n"
        "training:\n"
        "  seed: 42\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )


@pytest.mark.slow
def test_strict_lock_rejects_torch_minor_drift(tmp_path: Path) -> None:
    """Plant a lock with torch=2.5.1; current runtime differs; --strict-lock aborts."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    current = torch.__version__.split("+", 1)[0]
    if current == "2.5.1":
        pytest.skip("runtime torch already matches the planted lock; test can't simulate drift.")

    # Bootstrap home + doc.
    home = tmp_path / "dlm-home"
    home.mkdir()
    os.environ["DLM_HOME"] = str(home)

    doc = tmp_path / "strict.dlm"
    _write_minimal_dlm(doc)

    from dlm.doc.parser import parse_file
    from dlm.store.manifest import Manifest, save_manifest
    from dlm.store.paths import for_dlm

    parsed = parse_file(doc)

    # On hosts without a viable training plan (CPU-only CI) the CLI
    # aborts with "no viable training plan" before the lock validator
    # runs — the test can't distinguish lock-drift from plan-miss in
    # that case. Skip there; the unit-test counterpart still covers
    # the lock validator directly.
    from tests.fixtures.planning import resolve_spec_and_plan

    resolve_spec_and_plan(parsed)

    store = for_dlm(parsed.frontmatter.dlm_id)
    store.ensure_layout()
    save_manifest(
        store.manifest,
        Manifest(dlm_id=parsed.frontmatter.dlm_id, base_model="smollm2-135m"),
    )

    # Plant a lock that disagrees with the live torch version.
    forged = DlmLock(
        lock_version=CURRENT_LOCK_VERSION,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        dlm_id=parsed.frontmatter.dlm_id,
        dlm_sha256="0" * 64,
        base_model_revision="abc",
        hardware_tier="cpu",
        seed=42,
        determinism_class="best-effort",
        last_run_id=1,
        pinned_versions={"torch": "2.5.1"},
    )
    write_lock(store.root, forged)

    offline_vars = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    saved = {k: os.environ.pop(k, None) for k in offline_vars}
    try:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(home),
                "train",
                str(doc),
                "--strict-lock",
                "--max-steps",
                "1",
            ],
        )
        # Non-zero exit + mention of lock drift in the message.
        assert result.exit_code != 0, result.output
        assert "lock" in result.output.lower()
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
