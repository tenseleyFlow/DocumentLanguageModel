"""Execute one ORPO phase end-to-end.

Sibling to `dpo_phase.run`. Differences:

- No reference model — ORPO's objective wraps SFT + preference into
  one loss, so the policy carries both.
- `reference_adapter_version` still required: ORPO trains on top of
  the SFT-committed adapter, same two-phase commit semantics as DPO
  (v_N in, v_{N+1} out).

Same lifecycle helpers (preflight, lock, log, commit, manifest,
sidecar, lock-persist) as `dpo_phase` — imported from
`dlm.train.trainer`'s shared internals.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable

from dlm.replay import ReplayStore, diff_against_manifest
from dlm.store.manifest import load_manifest
from dlm.train.checkpoint_commit import commit_version
from dlm.train.determinism import seed_everything
from dlm.train.disk_preflight import preflight_disk
from dlm.train.logger import Banner, StepLogger, log_path_for
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
from dlm.train.preference.dpo_phase import _write_dpo_checkpoint
from dlm.train.preference.orpo_trainer import build_orpo_trainer

if TYPE_CHECKING:
    from pathlib import Path

    from dlm.base_models import BaseModelSpec
    from dlm.doc.parser import ParsedDlm
    from dlm.hardware.capabilities import Capabilities
    from dlm.hardware.plan import TrainingPlan
    from dlm.lock import LockMode
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
    """Execute one ORPO training cycle.

    Contract identical to `dpo_phase.run` — same kwargs, same return
    shape. `reference_adapter_version` is the SFT adapter we load as
    the policy starting point; no frozen reference is needed.
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
            "orpo_phase_start",
            reference_adapter_version=reference_adapter_version,
        )
        log.log_event(
            "delta",
            new=[s.section_id for s in change_set.new],
            unchanged=[s.section_id for s in change_set.unchanged],
            removed=list(change_set.removed),
        )

        orpo = _build_trainer(
            store=store,
            parsed=parsed,
            spec=spec,
            plan=plan,
            seed=seed,
            max_steps=max_steps,
            reference_adapter_version=reference_adapter_version,
            factory=trainer_factory,
            replay=replay,
        )

        start = time.perf_counter()
        train_result = orpo.train()
        elapsed = time.perf_counter() - start
        _LOG.info("ORPO training finished in %.1fs", elapsed)

        steps = int(getattr(orpo.state, "global_step", 0))
        final_train_loss = _maybe_float(getattr(train_result, "training_loss", None))

        final_val_loss: float | None = None
        final_val_perplexity: float | None = None
        early_stopped = False

        adapter_path = commit_version(
            store,
            lambda pending: _write_dpo_checkpoint(
                pending_dir=pending,
                dpo=orpo,
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
    replay: ReplayStore,
) -> Any:
    if factory is not None:
        return factory(
            store=store,
            parsed=parsed,
            spec=spec,
            plan=plan,
            seed=seed,
            max_steps=max_steps,
            reference_adapter_version=reference_adapter_version,
            replay=replay,
        )
    return _build_real_orpo_trainer(  # pragma: no cover
        store=store,
        parsed=parsed,
        spec=spec,
        plan=plan,
        seed=seed,
        max_steps=max_steps,
        reference_adapter_version=reference_adapter_version,
        replay=replay,
    )


def _build_real_orpo_trainer(  # pragma: no cover
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
    """Heavy path. Load base + SFT adapter as trainable policy, build
    preference dataset (doc + replay), instantiate TRL ORPOTrainer."""
    import random as _random

    from peft import PeftModel

    from dlm.data import prepare_tokenizer
    from dlm.train.loader import load_base_model
    from dlm.train.preference.dpo_dataset import build_dpo_dataset

    base_model = load_base_model(spec, plan)
    adapter_dir = store.adapter_version(reference_adapter_version)
    policy_model = PeftModel.from_pretrained(
        base_model, str(adapter_dir), is_trainable=True
    )

    tok_bringup = prepare_tokenizer(spec.hf_id, spec.revision)

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

    output_dir = store.logs / f"orpo-run-{seed}"
    return build_orpo_trainer(
        policy_model=policy_model,
        tokenizer=tok_bringup.tokenizer,
        train_dataset=train_ds,
        pref_cfg=parsed.frontmatter.training.preference,
        plan=plan,
        output_dir=output_dir,
        max_length=parsed.frontmatter.training.sequence_len,
        seed=seed,
        max_steps=max_steps,
    )
