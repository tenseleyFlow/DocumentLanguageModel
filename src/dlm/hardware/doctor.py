"""`dlm doctor` backing: produce capabilities + optional training plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dlm.doc.schema import TrainingConfig
from dlm.hardware.capabilities import Capabilities, probe
from dlm.hardware.plan import TrainingPlan, resolve
from dlm.hardware.refusals import ResolutionError

# A sensible reference base for the doctor's suggested-plan display when
# the user hasn't passed a `.dlm`. Picked to match the overview's default
# registry entry.
DEFAULT_REFERENCE_BASE_PARAMS = 1_500_000_000
DEFAULT_REFERENCE_SEQ_LEN = 2048


@dataclass(frozen=True)
class DoctorResult:
    """Combined capabilities probe + reference training plan (if computable)."""

    capabilities: Capabilities
    plan: TrainingPlan | None
    plan_error: str | None  # human-readable reason if refusal prevented plan resolution

    def to_dict(self) -> dict[str, Any]:
        return {
            "capabilities": _capabilities_to_dict(self.capabilities),
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "plan_error": self.plan_error,
        }


def doctor(
    training_config: TrainingConfig | None = None,
    *,
    base_params: int = DEFAULT_REFERENCE_BASE_PARAMS,
    seq_len: int = DEFAULT_REFERENCE_SEQ_LEN,
    force: bool = False,
    world_size: int = 1,
) -> DoctorResult:
    """Probe capabilities, optionally resolve a reference plan.

    The CLI calls this without a `training_config` (`dlm doctor`); it
    can also pass one through (`dlm doctor mydoc.dlm`). Either way, the
    capabilities are always reported; the plan is best-effort.

    `world_size` is threaded into `resolve(...)` so
    `effective_batch_size = micro_batch × grad_accum × world_size`
    reflects the multi-GPU reality in worker ranks. The `dlm train`
    CLI detects the DDP world_size via
    `dlm.train.distributed.detect_world_size()` and passes it here;
    single-process callers default to 1.
    """
    caps = probe()
    config = training_config if training_config is not None else TrainingConfig()
    num_adapters = len(config.adapters) if config.adapters is not None else 1

    try:
        plan = resolve(
            config,
            caps,
            base_params=base_params,
            seq_len=seq_len,
            force=force,
            num_adapters=num_adapters,
            world_size=world_size,
        )
        return DoctorResult(capabilities=caps, plan=plan, plan_error=None)
    except ResolutionError as exc:
        return DoctorResult(capabilities=caps, plan=None, plan_error=str(exc))


# --- helpers -----------------------------------------------------------------


def _capabilities_to_dict(caps: Capabilities) -> dict[str, Any]:
    """Explicit dict conversion so enums + tuples serialize cleanly."""
    data = asdict(caps)
    data["backend"] = caps.backend.value
    data["sm"] = list(caps.sm) if caps.sm is not None else None
    return data
