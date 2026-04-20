"""End-to-end: DPO phase shifts completions on a held-out probe.

Scaffold — the DPO runtime in `dlm.train.preference.dpo_phase.run` is
intentionally a `NotImplementedError` today because the heavy ML path
(policy + frozen reference + DPOTrainer + two-phase adapter commit)
is a substantial piece of plumbing we wanted to separate from the
contract + dispatcher work.

When the runtime lands, flip the `xfail` to a real body that:

1. Uses the shared `trained_store` fixture to get a .dlm with an
   SFT-trained adapter v0001 already on disk.
2. Appends 5 `::preference::` triples that systematically favor
   terse answers over verbose ones.
3. Runs `dlm.train.preference.phase_orchestrator.run_phases(
       ..., phase="dpo")` — expects v0002 adapter, a new training
   summary, and the two manifest entries the sprint's DoD calls for.
4. Generates from both v0001 and v0002 on a held-out probe prompt
   and asserts the v0002 completion is measurably terser
   (character-count delta past a threshold).
5. Confirms determinism: a second identical run produces the same
   adapter file sha256.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.slow
@pytest.mark.xfail(reason="DPO runtime body deferred", strict=True)
def test_dpo_phase_shifts_completions() -> None:
    from dlm.train.preference.dpo_phase import run

    # Calling with stub arguments — any positional works since the
    # body raises before using them.
    run(None, None, None, None, reference_adapter_version=1)  # type: ignore[arg-type]
