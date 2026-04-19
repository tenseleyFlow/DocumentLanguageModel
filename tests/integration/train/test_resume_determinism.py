"""Resume produces bit-identical loss curves under the determinism contract.

Sprint 09 DoD: after training N steps + saving state, resuming from
that checkpoint for a few more steps must produce the exact same loss
at step N+k as a single run from scratch that never resumed.

Marked `@pytest.mark.slow`. Runs only on CUDA hosts where the
determinism contract is `strict` (DeterminismSummary.class_). MPS is
`best_effort`; loss curves are close but not bit-identical, so this
test would fail on Apple Silicon and is skipped there.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.slow
def test_resume_bit_identical_loss() -> None:
    """Snapshot → resume → assert loss at step N matches single-run loss.

    Skipped on non-CUDA hosts because MPS / CPU determinism is
    best-effort only. The strict-determinism goal is a CUDA / cuDNN
    contract; Apple Silicon and CPU-only runners can't honor it.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        pytest.skip("torch not installed")

    if not torch.cuda.is_available():
        pytest.skip("strict determinism requires CUDA")

    # TODO(sprint-09-integration): implement when a CUDA slow-runner exists.
    # Shape:
    #   1. Build a tiny .dlm, run 20 steps → record loss_at_step_20_run_a
    #   2. Reset env, build same .dlm, run 10 steps + save checkpoint
    #   3. Resume from checkpoint, run 10 more steps → loss_at_step_20_run_b
    #   4. `assert loss_at_step_20_run_a == loss_at_step_20_run_b` (exact)
    pytest.xfail("CUDA integration scaffolded; body deferred to first CI slow run")
