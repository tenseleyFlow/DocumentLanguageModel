"""Resume produces bit-identical loss curves under the strict determinism contract.

Sprint 09 DoD: after training N steps and saving state, resuming from that
checkpoint for k more steps must produce the exact same loss at step N+k
as a single run from scratch that never resumed.

CUDA-only. MPS and CPU are best-effort; loss curves are close but not
bit-identical across resume boundaries because some ops don't ship
deterministic kernels for those backends. Enforcing bit-exactness on
those backends would false-flag a passing determinism contract as broken.

The test doesn't share the session-scoped `trained_store` fixture — it
needs two independent training stores so the straight-to-20 baseline
isn't contaminated by the 10+resume=10 interrupt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def _final_loss_at_run(logs_dir: Path, run_id: int) -> float:
    """Return the last-step loss recorded under `train-<run_id>-*.jsonl`."""
    matches = sorted(logs_dir.glob(f"train-{run_id:06d}-*.jsonl"))
    assert matches, f"no log for run_id={run_id} under {logs_dir}"
    rows = [json.loads(line) for line in matches[-1].read_text().splitlines() if line.strip()]
    step_rows = [row for row in rows if row.get("type") == "step"]
    assert step_rows, f"no step records in {matches[-1]}"
    last = max(step_rows, key=lambda row: row.get("step", 0))
    return float(last["loss"])


def _train_once(home: Path, *, max_steps: int, mode: str) -> float:
    """Train a fresh `.dlm` under `home`, return final-step loss."""
    os.environ["DLM_HOME"] = str(home)

    from dlm.base_models import resolve as resolve_base_model
    from dlm.doc.parser import parse_file
    from dlm.hardware import doctor
    from dlm.store.paths import for_dlm
    from dlm.train import run as run_training
    from tests.fixtures.dlm_factory import make_dlm

    doc = home / "resume.dlm"
    if not doc.exists():
        doc.write_text(make_dlm(base_model="smollm2-135m"), encoding="utf-8")

    parsed = parse_file(doc)
    spec = resolve_base_model(parsed.frontmatter.base_model)
    plan = doctor().plan
    if plan is None:
        pytest.skip("no viable plan on this host — determinism test needs a real trainer")

    store = for_dlm(parsed.frontmatter.dlm_id)
    store.ensure_layout()

    run_training(
        store,
        parsed,
        spec,
        plan,
        mode=mode,  # type: ignore[arg-type]
        seed=42,
        max_steps=max_steps,
    )
    # Run IDs start at 1 and increment per call; the fresh run is 1, the
    # post-checkpoint resume is 2.
    run_id = len(list(store.logs.glob("train-*.jsonl")))
    return _final_loss_at_run(store.logs, run_id)


@pytest.mark.slow
def test_resume_reproduces_bit_identical_loss_at_step_20(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Straight 20 vs 10+resume=20: final loss must agree to the last bit.

    Requires CUDA + a determinism-strict plan; skipped otherwise.
    """
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    if not torch.cuda.is_available():
        pytest.skip("strict-determinism contract is CUDA-only")

    # Pop offline env so tokenizer / base model can load.
    offline_vars = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    saved = {k: os.environ.pop(k, None) for k in offline_vars}
    try:
        home_a = tmp_path_factory.mktemp("dlm-resume-baseline")
        loss_a = _train_once(home_a, max_steps=20, mode="fresh")

        home_b = tmp_path_factory.mktemp("dlm-resume-split")
        _train_once(home_b, max_steps=10, mode="fresh")
        loss_b = _train_once(home_b, max_steps=20, mode="resume")

        assert loss_a == loss_b, (
            f"determinism broken across resume boundary: baseline={loss_a!r}, "
            f"resumed={loss_b!r}, delta={loss_a - loss_b!r}"
        )
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
