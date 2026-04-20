"""Execute one DPO phase end-to-end.

Sibling to `dlm.train.trainer.run` — same result shape, same
checkpoint-commit semantics, different objective. The heavy path lives
behind `# pragma: no cover`; the slow integration test at
`tests/integration/train/preference/` drives this module, and the
orchestrator's unit tests stub it with a mock runner.

Contract with the phase orchestrator:

- Accepts `reference_adapter_version: int` — the SFT-trained version
  to load as DPO's frozen reference (or to continue training from
  when `dpo.reference="base"`).
- Returns a `TrainingRunResult` so the orchestrator can report
  adapter version / steps / loss uniformly across phases.
- Writes a new adapter version (reference_adapter_version + 1) via
  the same two-phase commit (`_write_checkpoint` →
  `commit_checkpoint`) the SFT trainer uses. Guarantees atomicity if
  the process dies mid-train.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.doc.parser import ParsedDlm
    from dlm.hardware.capabilities import Capabilities
    from dlm.hardware.plan import TrainingPlan
    from dlm.lock import LockMode
    from dlm.store.paths import StorePath
    from dlm.train.trainer import TrainingRunResult


def run(  # pragma: no cover
    store: StorePath,
    parsed: ParsedDlm,
    spec: BaseModelSpec,
    plan: TrainingPlan,
    *,
    reference_adapter_version: int,
    seed: int | None = None,
    max_steps: int | None = None,
    lock_mode: LockMode = "default",
    capabilities: Capabilities | None = None,
    trainer_factory: Any = None,
) -> TrainingRunResult:
    """Execute one DPO training cycle.

    Implementation lands alongside the slow integration test that
    covers it; this signature pins the contract for the orchestrator
    and the unit-test mocks.

    High-level steps:

    1. Preflight disk, seed determinism, open log.
    2. Validate dlm.lock against candidate (same rules as SFT).
    3. Build DPO dataset from `::preference::` sections +
       replay-corpus preference samples.
    4. Load policy model: base + SFT adapter v_N as trainable.
    5. Load reference model per `dpo.reference` mode.
    6. Build + run DPOTrainer.
    7. Two-phase adapter commit → v_{N+1}.
    8. Append manifest entry + write training_state sidecar.
    9. Update dlm.lock.
    """
    raise NotImplementedError(
        "DPO phase runtime lands with the slow integration test; the "
        "orchestrator's unit tests cover the dispatcher via mocks"
    )
