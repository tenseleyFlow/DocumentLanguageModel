"""Domain dispatcher for `dlm train` (single-shot path).

Lifts the doctor → manifest-provision → run_phases pipeline out of the
CLI. Callers (CLI, LSP "Run Training" command, future automation) build
a `TrainRequest`, call `run_train`, and render the typed `TrainResult`.
The dispatcher does no console I/O; CLI-shaped concerns — multi-GPU
launcher dispatch, license interactive prompt, --watch loop, RPC
probe server, terminal rendering — stay in `dlm.cli.commands.train`.

External-module imports are dotted (e.g. `from dlm import hardware as
_hardware; _hardware.doctor(...)`) so test fixtures that monkeypatch
`dlm.hardware.doctor` and
`dlm.train.preference.phase_orchestrator.run_phases` resolve at call
time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dlm import hardware as _hardware
from dlm.train.preference import phase_orchestrator as _orchestrator
from dlm.train.preference.phase_orchestrator import Phase, PhaseResult

if TYPE_CHECKING:
    from pathlib import Path

    from dlm.base_models.schema import BaseModelSpec
    from dlm.doc.parser import ParsedDlm
    from dlm.hardware.plan import TrainingPlan
    from dlm.lock import LockMode
    from dlm.store.paths import StorePath
    from dlm.train.trainer import Mode


class NoViableTrainingPlanError(RuntimeError):
    """`doctor()` returned no plan for the current host / config."""


@dataclass(frozen=True)
class TrainRequest:
    """Inputs to `run_train`.

    The CLI is responsible for parsing the .dlm, resolving the spec
    (with license acceptance), validating flags, and detecting the
    accelerate world size; the dispatcher receives all of those as
    already-typed objects.
    """

    parsed: ParsedDlm
    target_path: Path
    spec: BaseModelSpec
    store: StorePath
    phase: Phase
    mode: Mode
    seed: int | None
    max_steps: int | None
    lock_mode: LockMode
    world_size: int
    strict_metrics: bool
    include_auto_mined: bool


@dataclass(frozen=True)
class TrainResult:
    """Outcome of `run_train`. `phase_results` is empty when nothing
    matched the requested phase (no SFT content, no preference content,
    etc.); the CLI surfaces a "no-op" message."""

    plan: TrainingPlan
    phase_results: list[PhaseResult]


def run_train(req: TrainRequest) -> TrainResult:
    """Probe hardware, ensure store manifest, run all requested phases."""
    doctor_result = _hardware.doctor(
        training_config=req.parsed.frontmatter.training,
        base_params=req.spec.params,
        seq_len=min(
            req.parsed.frontmatter.training.sequence_len,
            req.spec.effective_context_length,
        ),
        world_size=req.world_size,
    )
    plan = doctor_result.plan
    if plan is None:
        raise NoViableTrainingPlanError(
            "no viable training plan for this host. Run `dlm doctor` for details."
        )

    req.store.ensure_layout()

    # `dlm init` writes a manifest as part of store provisioning. Mirror
    # that here when the layout exists but the manifest doesn't — covers
    # auto-scaffold via `dlm train <dir>` and hand-authored .dlms with
    # fresh ULIDs that never went through `dlm init` (e.g. authored via
    # the LSP). License acceptance has already been validated upstream.
    if not req.store.manifest.exists():
        from dlm.base_models import is_gated
        from dlm.base_models.license import require_acceptance
        from dlm.store.manifest import Manifest, save_manifest

        acceptance = (
            require_acceptance(req.spec, accept_license=True, via="cli_flag")
            if is_gated(req.spec)
            else None
        )
        save_manifest(
            req.store.manifest,
            Manifest(
                dlm_id=req.parsed.frontmatter.dlm_id,
                base_model=req.spec.key,
                base_model_revision=req.spec.revision,
                source_path=req.target_path.resolve(),
                license_acceptance=acceptance,
            ),
        )

    phase_results = _orchestrator.run_phases(
        req.store,
        req.parsed,
        req.spec,
        plan,
        phase=req.phase,
        mode=req.mode,
        seed=req.seed,
        max_steps=req.max_steps,
        lock_mode=req.lock_mode,
        capabilities=doctor_result.capabilities,
        world_size=req.world_size,
        strict_metrics=req.strict_metrics,
        include_auto_mined=req.include_auto_mined,
    )

    return TrainResult(plan=plan, phase_results=phase_results)
