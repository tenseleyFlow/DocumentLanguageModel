"""Execute one DPO phase end-to-end.

Sibling to `dlm.train.trainer.run` — same result shape, same
checkpoint-commit semantics, different objective. The heavy ML path
(`_build_real_dpo_trainer`) is `# pragma: no cover`; unit tests drive
the dispatcher via a `trainer_factory` seam; the slow integration
test drives the real path end-to-end.

Contract with the phase orchestrator:

- Accepts `reference_adapter_version: int` — the SFT-trained version
  to load as DPO's frozen reference (or to continue training from
  when `dpo.reference="base"`).
- Returns a `TrainingRunResult` so the orchestrator can report
  adapter version / steps / loss uniformly across phases.
- Writes a new adapter version via the same two-phase commit
  (`commit_version`) the SFT trainer uses. Guarantees atomicity if
  the process dies mid-train.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from dlm.replay import ReplayStore, diff_against_manifest
from dlm.store.manifest import load_manifest
from dlm.train.checkpoint_commit import commit_version
from dlm.train.determinism import seed_everything
from dlm.train.disk_preflight import preflight_disk
from dlm.train.logger import Banner, StepLogger, log_path_for
from dlm.train.preference.dpo_dataset import build_dpo_dataset
from dlm.train.preference.dpo_trainer import build_dpo_trainer, load_reference_model
from dlm.train.state_sidecar import PinnedVersions, capture_runtime_versions, save_state
from dlm.train.trainer import (
    TrainingRunResult,
    _append_change_set_to_replay,
    _append_training_run,
    _maybe_float,
    _next_run_id,
    _persist_lock,
    _snapshot_training_state,
    _validate_or_abort_lock,
    _write_training_summary,
)

if TYPE_CHECKING:
    from pathlib import Path

    from dlm.base_models import BaseModelSpec
    from dlm.doc.parser import ParsedDlm
    from dlm.hardware.capabilities import Capabilities
    from dlm.hardware.plan import TrainingPlan
    from dlm.lock import LockMode
    from dlm.replay import ChangeSet
    from dlm.store.paths import StorePath

_LOG = logging.getLogger(__name__)

TrainerFactory = Callable[..., Any]


def run(
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
    trainer_factory: TrainerFactory | None = None,
) -> TrainingRunResult:
    """Execute one DPO training cycle.

    Structure mirrors `dlm.train.trainer.run` — same preflight, lock,
    log, replay, commit, manifest, lock-persist lifecycle. The only
    per-phase differences live in dataset assembly (build_dpo_dataset)
    and trainer construction (DPOTrainer + frozen reference).

    `trainer_factory` is the unit-test seam: pass a callable whose
    return value exposes `.train()`, `.state.global_step`, and
    `.save_model(path)`.
    """
    if seed is None:
        seed = parsed.frontmatter.training.seed

    preflight_disk(store.root, spec, plan)

    determinism = seed_everything(seed)

    store.logs.mkdir(parents=True, exist_ok=True)
    run_id = _next_run_id(store)
    log_path = log_path_for(store.logs, run_id)
    versions = capture_runtime_versions()

    prior_manifest = load_manifest(store.manifest)
    change_set = diff_against_manifest(list(parsed.sections), prior_manifest)

    lock_decision = (
        _validate_or_abort_lock(
            store=store,
            parsed=parsed,
            spec=spec,
            seed=seed,
            run_id=run_id,
            versions=versions,
            determinism_class=determinism.class_,
            capabilities=capabilities,
            lock_mode=lock_mode,
            license_acceptance=prior_manifest.license_acceptance,
        )
        if parsed.source_path is not None
        else None
    )
    replay = ReplayStore.at(store.replay_corpus, store.replay_index)

    with StepLogger(log_path) as log:
        log.write_banner(
            Banner(
                run_id=run_id,
                seed=seed,
                determinism_class=determinism.class_,
                determinism_notes=tuple(determinism.notes),
                pinned_versions=tuple(
                    (k, v) for k, v in sorted(versions.items()) if isinstance(v, str)
                ),
                plan=plan.to_dict(),
            )
        )
        log.log_event(
            "dpo_phase_start",
            reference_adapter_version=reference_adapter_version,
            dpo_reference_mode=parsed.frontmatter.training.preference.reference,
        )
        log.log_event(
            "delta",
            new=[s.section_id for s in change_set.new],
            unchanged=[s.section_id for s in change_set.unchanged],
            removed=list(change_set.removed),
        )

        dpo = _build_trainer(
            store=store,
            parsed=parsed,
            spec=spec,
            plan=plan,
            seed=seed,
            max_steps=max_steps,
            reference_adapter_version=reference_adapter_version,
            factory=trainer_factory,
            change_set=change_set,
            replay=replay,
        )

        start = time.perf_counter()
        train_result = dpo.train()
        elapsed = time.perf_counter() - start
        _LOG.info("DPO training finished in %.1fs", elapsed)

        steps = int(getattr(dpo.state, "global_step", 0))
        final_train_loss = _maybe_float(getattr(train_result, "training_loss", None))

        # DPO typically runs to epoch-end with small preference sets;
        # no early-stop callback is wired, and DPO loss curves aren't
        # val-loss/perplexity in the SFT sense. Surface `None` for the
        # eval metrics so the summary stays schema-consistent.
        final_val_loss: float | None = None
        final_val_perplexity: float | None = None
        early_stopped = False

        adapter_path = commit_version(
            store,
            lambda pending: _write_dpo_checkpoint(
                pending_dir=pending,
                dpo=dpo,
                spec=spec,
                versions=versions,
                use_qlora=plan.use_qlora,
            ),
        )
        adapter_version = int(adapter_path.name.lstrip("v"))

        _append_change_set_to_replay(replay, change_set, run_id=run_id)

        summary_path = _write_training_summary(
            store=store,
            log_path=log_path,
            run_id=run_id,
            adapter_version=adapter_version,
            seed=seed,
            steps=steps,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            final_val_perplexity=final_val_perplexity,
            early_stopped=early_stopped,
            duration_seconds=elapsed,
            determinism=determinism,
        )

        _append_training_run(
            store=store,
            run_id=run_id,
            adapter_version=adapter_version,
            seed=seed,
            steps=steps,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            base_model_revision=spec.revision,
            versions=versions,
            current_sections=list(parsed.sections),
            summary_path=summary_path,
        )

        if lock_decision is not None and lock_decision.should_write_lock:
            _persist_lock(
                store=store,
                parsed=parsed,
                spec=spec,
                seed=seed,
                run_id=run_id,
                versions=versions,
                determinism_class=determinism.class_,
                capabilities=capabilities,
                license_acceptance=prior_manifest.license_acceptance,
            )

        log.log_event(
            "run_complete",
            run_id=run_id,
            adapter_version=adapter_version,
            steps=steps,
            elapsed_seconds=elapsed,
            early_stopped=early_stopped,
            summary_path=str(summary_path),
        )

    return TrainingRunResult(
        run_id=run_id,
        adapter_version=adapter_version,
        adapter_path=adapter_path,
        log_path=log_path,
        summary_path=summary_path,
        seed=seed,
        steps=steps,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        final_val_perplexity=final_val_perplexity,
        early_stopped=early_stopped,
        determinism=determinism,
    )


def _write_dpo_checkpoint(
    *,
    pending_dir: Path,
    dpo: Any,
    spec: BaseModelSpec,
    versions: PinnedVersions,
    use_qlora: bool,
) -> None:
    """Drop the DPO-trained policy adapter + state sidecar into `pending_dir`.

    Mirrors SFT's `_write_checkpoint`: `save_model` emits the adapter
    weights + config; `save_state` captures the optimizer/scheduler/RNGs.
    Both paths depend only on the trainer object's public surface
    (`save_model`, `optimizer`, `lr_scheduler`, `state`), so DPOTrainer
    plugs in without adaptation.
    """
    dpo.save_model(str(pending_dir))
    state = _snapshot_training_state(dpo, spec=spec, versions=versions, use_qlora=use_qlora)
    save_state(pending_dir, state)


def _build_trainer(
    *,
    store: StorePath,
    parsed: ParsedDlm,
    spec: BaseModelSpec,
    plan: TrainingPlan,
    seed: int,
    max_steps: int | None,
    reference_adapter_version: int,
    factory: TrainerFactory | None,
    change_set: ChangeSet,
    replay: ReplayStore,
) -> Any:
    """Assemble the DPO trainer via the configured factory or the real
    HF path.

    `factory` is the unit-test seam — when provided, it's called with
    a fully-built kwargs dict and expected to return something with
    `.train() -> TrainOutput`, `.state.global_step`, and
    `.save_model(path)`.
    """
    if factory is not None:
        return factory(
            store=store,
            parsed=parsed,
            spec=spec,
            plan=plan,
            seed=seed,
            max_steps=max_steps,
            reference_adapter_version=reference_adapter_version,
            change_set=change_set,
            replay=replay,
        )

    return _build_real_dpo_trainer(  # pragma: no cover
        store=store,
        parsed=parsed,
        spec=spec,
        plan=plan,
        seed=seed,
        max_steps=max_steps,
        reference_adapter_version=reference_adapter_version,
        replay=replay,
    )


def _build_real_dpo_trainer(  # pragma: no cover
    *,
    store: StorePath,
    parsed: ParsedDlm,
    spec: BaseModelSpec,
    plan: TrainingPlan,
    seed: int,
    max_steps: int | None,
    reference_adapter_version: int,
    replay: ReplayStore,
) -> Any:
    """Heavy path: load policy + reference, build dataset, instantiate
    DPOTrainer. Covered by the slow integration suite."""
    import random as _random

    from peft import PeftModel

    from dlm.data import prepare_tokenizer
    from dlm.train.loader import load_base_model

    # Policy: base + the SFT-trained adapter as trainable.
    base_model = load_base_model(spec, plan)
    adapter_dir = store.adapter_version(reference_adapter_version)
    policy_model = PeftModel.from_pretrained(
        base_model, str(adapter_dir), is_trainable=True
    )

    # Reference: frozen per preference.reference mode. We reload a
    # clean base for the reference rather than sharing `base_model` so
    # policy gradient flow doesn't touch the reference weights.
    pref_cfg = parsed.frontmatter.training.preference
    ref_adapter_path = adapter_dir if pref_cfg.reference == "pre_adapter" else None
    ref_model = load_reference_model(
        spec,
        plan,
        adapter_path=ref_adapter_path,
        mode=pref_cfg.reference,
    )

    tok_bringup = prepare_tokenizer(spec.hf_id, spec.revision)

    # Dataset: current doc's preferences + recency-weighted replay
    # preferences (if any prior preference snapshots are in the corpus).
    from datasets import Dataset, concatenate_datasets

    doc_ds = build_dpo_dataset(list(parsed.sections))
    rng = _random.Random(seed + reference_adapter_version)
    now = datetime.now(UTC).replace(tzinfo=None, microsecond=0)
    replay_rows = replay.sample_preference_rows(
        k=max(8, 2 * len(doc_ds)), now=now, rng=rng
    )
    if replay_rows:
        replay_ds = Dataset.from_list(replay_rows)
        train_ds = concatenate_datasets([doc_ds, replay_ds])
    else:
        train_ds = doc_ds

    output_dir = store.logs / f"dpo-run-{seed}"
    return build_dpo_trainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tok_bringup.tokenizer,
        train_dataset=train_ds,
        pref_cfg=pref_cfg,
        plan=plan,
        output_dir=output_dir,
        max_length=parsed.frontmatter.training.sequence_len,
        seed=seed,
        max_steps=max_steps,
    )
