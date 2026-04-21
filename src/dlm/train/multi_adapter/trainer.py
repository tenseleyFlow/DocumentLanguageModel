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
    gate_embed_factory: Callable[[], tuple[Callable[[str], Any], int]] | None = None,
) -> list[TrainingRunResult]:
    """Train every declared adapter in declaration order.

    Returns one `TrainingRunResult` per adapter trained. For a
    single-adapter document the list has length 1 and the result
    mirrors `trainer.run()`'s return value.

    Determinism: each adapter uses the same `seed`; repeated runs over
    the same document yield the same per-adapter version histories.

    `gate_embed_factory` lets callers plug in a `(prompt -> Tensor,
    input_dim)` pair when `training.gate.enabled` is set. Omitting it
    defaults to the base-model embedder (loads an HF model with
    adapters disabled, mean-pools the last hidden state). Tests pass
    a stub to skip the real HF load.
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
            sections=tuple(routing.by_adapter[name]),
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

    _maybe_run_gate_pass(
        store=store,
        parsed=parsed,
        spec=spec,
        plan=plan,
        seed=seed,
        run_id=results[-1].run_id if results else 0,
        gate_embed_factory=gate_embed_factory,
    )

    return results


def _maybe_run_gate_pass(
    *,
    store: StorePath,
    parsed: ParsedDlm,
    spec: BaseModelSpec,
    plan: TrainingPlan,
    seed: int | None,
    run_id: int,
    gate_embed_factory: Callable[[], tuple[Callable[[str], Any], int]] | None,
) -> None:
    """Run the post-SFT learned-gate training pass when enabled.

    Kept separate so the multi-adapter orchestrator's happy path stays
    short. All errors are swallowed — gate training is best-effort per
    the Sprint 34 risk matrix.
    """
    import logging

    log = logging.getLogger(__name__)
    gate_cfg = parsed.frontmatter.training.gate
    if not gate_cfg.enabled:
        return
    adapters = parsed.frontmatter.training.adapters
    if adapters is None or len(adapters) < 2:
        return

    from dlm.metrics.recorder import MetricsRecorder
    from dlm.train.gate.orchestrator import run_post_sft_gate

    try:
        embed, input_dim = _resolve_gate_embedder(spec, plan, gate_embed_factory)
    except Exception as exc:  # noqa: BLE001 — best-effort path
        log.warning("gate: embedder setup failed, skipping gate pass: %s", exc)
        return

    try:
        run_post_sft_gate(
            store,
            parsed,
            run_id=run_id,
            recorder=MetricsRecorder(store.root),
            embed=embed,
            input_dim=input_dim,
            seed=seed,
        )
    except Exception as exc:  # noqa: BLE001 — best-effort path
        log.warning("gate: post-SFT pass failed, leaving store gate-less: %s", exc)


def _resolve_gate_embedder(
    spec: BaseModelSpec,
    plan: TrainingPlan,
    factory: Callable[[], tuple[Callable[[str], Any], int]] | None,
) -> tuple[Callable[[str], Any], int]:
    """Return a `(embed, input_dim)` pair for the gate training pass.

    Test seam: callers can pass an embed factory that returns a stub
    closure + the synthetic input_dim. Production callers omit it, and
    this helper loads the base model + tokenizer and returns a real
    mean-pool embedder.
    """
    if factory is not None:
        return factory()

    return _default_embedder(spec, plan)


def _default_embedder(
    spec: BaseModelSpec,
    plan: TrainingPlan,
) -> tuple[Callable[[str], Any], int]:  # pragma: no cover — heavy HF path
    """Default embedder — loads the HF base model + tokenizer.

    Covered by the Sprint 34 slow integration test; unit tests pass a
    stub via `gate_embed_factory`.
    """
    from transformers import AutoTokenizer

    from dlm.inference.gate import embed_prompt
    from dlm.train.loader import load_base_model

    base_model = load_base_model(spec, plan)
    base_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision)
    hidden_size = int(base_model.config.hidden_size)

    def _embed(prompt: str) -> Any:
        return embed_prompt(
            prompt=prompt,
            tokenizer=tokenizer,
            base_model=base_model,
        )

    return _embed, hidden_size
