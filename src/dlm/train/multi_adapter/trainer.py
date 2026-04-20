"""Multi-adapter training orchestrator.

`run_all(store, parsed, ...)` loops over the declared named adapters,
calling the single-adapter `trainer.run()` once per adapter with:

- a `ParsedDlm` scoped to that adapter's sections (per the router's
  `build_plan`),
- `adapter_name=<name>` so checkpoint commits land under
  `adapter/<name>/versions/vNNNN/` rather than the flat tree,
- the same base model / hardware plan / seed as the caller specified.

Single-adapter documents (no `training.adapters` block) fall through
to a single `run()` call — the orchestrator is a safe default entry
point regardless of document shape.

Scope note: inference selection, export merge, and doctor memory
refusal layer on in sprint 20b. This module owns only the per-adapter
orchestration and the resulting adapter-versioned store layout.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dlm.doc.parser import ParsedDlm
from dlm.train.multi_adapter.router import build_plan, declared_adapter_names

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.hardware.capabilities import Capabilities
    from dlm.hardware.plan import TrainingPlan
    from dlm.lock import LockMode
    from dlm.store.paths import StorePath
    from dlm.train.trainer import Mode, TrainingRunResult


def run_all(
    store: StorePath,
    parsed: ParsedDlm,
    spec: BaseModelSpec,
    plan: TrainingPlan,
    *,
    mode: Mode = "fresh",
    seed: int | None = None,
    max_steps: int | None = None,
    lock_mode: LockMode = "default",
    capabilities: Capabilities | None = None,
    trainer_factory: Callable[..., Any] | None = None,
) -> list[TrainingRunResult]:
    """Train every declared adapter in declaration order.

    Returns one `TrainingRunResult` per adapter trained. For a
    single-adapter document the list has length 1 and the result
    mirrors `trainer.run()`'s return value.

    Determinism: each adapter uses the same `seed`; repeated runs over
    the same document yield the same per-adapter version histories.
    """
    from dlm.train.trainer import run as run_single

    names = declared_adapter_names(parsed)

    # Single-adapter doc: passthrough, no scoping, no named-layout flip.
    if parsed.frontmatter.training.adapters is None:
        return [
            run_single(
                store,
                parsed,
                spec,
                plan,
                mode=mode,
                seed=seed,
                max_steps=max_steps,
                lock_mode=lock_mode,
                capabilities=capabilities,
                trainer_factory=trainer_factory,
            )
        ]

    # Multi-adapter doc: scope sections per adapter and target
    # `adapter/<name>/` on every call. Lock validation is done once
    # up-front on the first adapter (using the full document); the
    # subsequent adapters skip it because the document state is
    # unchanged within a single `run_all` invocation.
    routing = build_plan(parsed)
    results: list[TrainingRunResult] = []
    per_adapter_lock_mode: LockMode = lock_mode
    for name in names:
        scoped = ParsedDlm(
            frontmatter=parsed.frontmatter,
            sections=list(routing.by_adapter[name]),
            source_path=parsed.source_path,
        )
        result = run_single(
            store,
            scoped,
            spec,
            plan,
            mode=mode,
            seed=seed,
            max_steps=max_steps,
            lock_mode=per_adapter_lock_mode,
            capabilities=capabilities,
            trainer_factory=trainer_factory,
            adapter_name=name,
        )
        results.append(result)
        # After the first adapter, the lock has been written — skip
        # re-validating it for subsequent adapters in the same call.
        per_adapter_lock_mode = "ignore"

    return results
