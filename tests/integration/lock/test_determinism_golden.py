"""Determinism golden — two runs on the same seed must produce byte-identical weights.

Sprint 15 DoD: on the tiny model, a fresh training run with seed=42 and
strict determinism flags must reproduce the exact same
`adapter_model.safetensors` SHA on a rerun. CPU host is sufficient —
the determinism contract is "as close as the stack allows", and
SmolLM2-135M on CPU exercises the bulk of the kernel surface that can
introduce nondeterminism.

Skipped on MPS in CI (documented weak determinism) and when the tiny
model fixture isn't resolvable offline.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def _adapter_sha(adapter_dir: Path) -> str:
    weights = adapter_dir / "adapter_model.safetensors"
    assert weights.is_file(), f"missing {weights}"
    digest = hashlib.sha256()
    with weights.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _train_once(home: Path) -> Path:
    """Fresh 20-step training run under `home`. Returns adapter dir."""
    os.environ["DLM_HOME"] = str(home)

    from dlm.base_models import resolve as resolve_base_model
    from dlm.doc.parser import parse_file
    from dlm.hardware import doctor
    from dlm.store.paths import for_dlm
    from dlm.train import run as run_training
    from tests.fixtures.dlm_factory import make_dlm

    doc = home / "det.dlm"
    doc.write_text(
        make_dlm(
            base_model="smollm2-135m",
            # Crockford base32 (no I/L/O/U); fixed → same dlm_sha256 across runs.
            dlm_id="01HRDGN1DN" + "0" * 16,
        ),
        encoding="utf-8",
    )
    parsed = parse_file(doc)
    spec = resolve_base_model(parsed.frontmatter.base_model)
    plan = doctor().plan
    assert plan is not None
    store = for_dlm(parsed.frontmatter.dlm_id)
    store.ensure_layout()

    # Audit-08 P1: run_training reads the manifest via load_manifest
    # (for content-delta diff). Audit-05 moved the initial-manifest
    # write to `dlm init`; this test bypasses init, so seed it here
    # the same way trained_store + Sprint 20 fixtures do.
    from dlm.store.manifest import Manifest, save_manifest

    save_manifest(
        store.manifest,
        Manifest(
            dlm_id=parsed.frontmatter.dlm_id,
            base_model=parsed.frontmatter.base_model,
        ),
    )

    run_training(
        store,
        parsed,
        spec,
        plan,
        mode="fresh",
        seed=42,
        max_steps=20,
        lock_mode="update",
    )
    adapter = store.resolve_current_adapter()
    assert adapter is not None
    return adapter


@pytest.mark.slow
def test_two_fresh_runs_produce_identical_adapter(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Seed-locked training is byte-identical across two isolated runs."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"torch/transformers unavailable: {exc}")

    try:
        from tests.fixtures.tiny_model import tiny_model_path

        tiny_model_path()
    except Exception as exc:
        pytest.skip(f"tiny-model fixture unavailable: {exc}")

    offline_vars = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    saved = {k: os.environ.pop(k, None) for k in offline_vars}
    try:
        home_a = tmp_path_factory.mktemp("det-a")
        sha_a = _adapter_sha(_train_once(home_a))
        home_b = tmp_path_factory.mktemp("det-b")
        sha_b = _adapter_sha(_train_once(home_b))
        assert sha_a == sha_b, (
            f"Determinism contract broken: run-A={sha_a}, run-B={sha_b}. "
            "Regenerate with scripts/regen-determinism-golden.py --approve "
            "after reviewing the stack changes that caused the drift."
        )
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
