"""End-to-end: `dlm.train.run()` one full cycle on SmolLM2-135M (Sprint 09).

Replaces the audit-04 M4 scaffold with a real body. Exercises the live
`_build_real_trainer` → SFTTrainer path — the one the unit suite
pragma-skips because it demands torch, transformers, and model weights.

The shared `trained_store` fixture (Sprint 14.5) handles setup; this test
only asserts the aftermath:

- Adapter version directory `v0001/` under the store carries
  `adapter_config.json`, `adapter_model.safetensors`,
  `training_state.pt`, and the matching `.sha256`.
- `load_state` round-trips the sidecar (integrity: hash still matches
  the file on disk after training wrote it).
- `manifest.json` has exactly one `TrainingRunSummary` with populated
  `content_hashes` (audit-04 M2 wired the delta → manifest loop).
- `logs/train-000001-*.jsonl` exists, contains a banner + at least one
  step record.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.slow
def test_one_cycle_produces_adapter_sidecar_manifest_log(trained_store) -> None:
    from dlm.store.manifest import load_manifest
    from dlm.train.state_sidecar import STATE_FILENAME, STATE_SHA_FILENAME, load_state

    store = trained_store.store

    adapter_dir = store.resolve_current_adapter()
    assert adapter_dir is not None, "trained_store fixture didn't set adapter/current.txt"

    # PEFT artifacts.
    assert (adapter_dir / "adapter_config.json").is_file()
    assert (adapter_dir / "adapter_model.safetensors").is_file()

    # Training-state sidecar (Sprint 09 audit F12 two-phase commit).
    assert (adapter_dir / STATE_FILENAME).is_file()
    assert (adapter_dir / STATE_SHA_FILENAME).is_file()
    # `load_state` raises on sha mismatch — proves integrity survived the write.
    from dlm.train.state_sidecar import capture_runtime_versions

    state = load_state(adapter_dir, runtime_versions=capture_runtime_versions())
    assert state["global_step"] > 0

    # Manifest: at least the fixture's initial TrainingRunSummary + populated
    # content_hashes (audit-04 M2). Other session-scoped tests sharing
    # trained_store may append additional runs.
    manifest = load_manifest(store.manifest)
    assert len(manifest.training_runs) >= 1, manifest.training_runs
    run = manifest.training_runs[0]
    assert run.run_id == 1
    assert run.seed == 42
    assert manifest.content_hashes, "content_hashes empty — delta → manifest loop regressed"

    # JSONL log — banner + at least one step record.
    log_files = sorted(store.logs.glob("train-000001-*.jsonl"))
    assert log_files, f"no train-000001 log under {store.logs}"
    rows = [json.loads(line) for line in log_files[-1].read_text().splitlines() if line.strip()]
    assert rows, "log file is empty"
    row_types = {row.get("type") for row in rows}
    assert "banner" in row_types
    assert any(row.get("type") == "step" for row in rows), (
        f"no step records in {log_files[-1]}: {row_types}"
    )
