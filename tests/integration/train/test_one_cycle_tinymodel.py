"""End-to-end: `dlm train` one full cycle on the SmolLM2-135M fixture.

Sprint 09 DoD: write adapter + state sidecar, update manifest, assert
training loss is finite. The mock-factory unit tests in
`tests/unit/train/test_trainer.py` cover orchestration plumbing; this
test exercises the real `_build_real_trainer` → SFTTrainer path that
the unit tests pragma-skip.

Marked `@pytest.mark.slow` so the default `pytest` run skips it; CI
invokes it via `pytest -m slow`.

Tokens of evidence this test should produce:
- `adapter_config.json` + `adapter_model.safetensors` in `v0001/`
- `training_state.pt` + `.sha256` that round-trips via `load_state`
- `manifest.json` with exactly one `TrainingRunSummary` and populated
  `content_hashes`
- `logs/train-000001-*.jsonl` with banner + at least one step record

Runs ≤20 steps with a tiny batch size so CPU/MPS wall-clock stays in
the 1–2 minute range on a reasonable dev box.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.slow
def test_one_cycle_on_smollm2_135m(tmp_path: Path) -> None:
    """Full training cycle on the tiny-model fixture.

    Implementation deferred — this test asserts the plumbing exists
    but depends on the SmolLM2-135M fixture being resolvable offline
    AND the CI runner having ≥4GB of RAM free. Until a dedicated
    slow-job runner is provisioned, `pytest -m slow` will collect
    this test and skip via the xfail below when the fixture isn't
    available.
    """
    from tests.fixtures.tiny_model import tiny_model_path

    try:
        tiny_model_path()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"tiny model fixture unavailable: {exc}")

    # TODO(sprint-09-integration): flesh out the end-to-end run.
    # Blocking:
    #   1. Writing a synthetic `.dlm` via tests.fixtures.dlm_factory
    #   2. `for_dlm(dlm_id, home=tmp_path).ensure_layout()`
    #   3. `save_manifest(Manifest(dlm_id=..., base_model="smollm2-135m"))`
    #   4. `spec = BASE_MODELS["smollm2-135m"]`
    #   5. `plan = doctor().plan` (CI host must support bf16 or fp16)
    #   6. `run(store, parsed, spec, plan, mode="fresh", max_steps=20)`
    #   7. Assertions per docstring.
    pytest.xfail("slow integration test scaffolded; body deferred to first CI slow run")
