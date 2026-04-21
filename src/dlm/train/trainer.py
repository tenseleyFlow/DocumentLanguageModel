"""End-to-end orchestrator: `run(store, parsed, spec, plan, ...)`.

This is the single entry point the CLI (`dlm train`) calls. It wires:

    preflight_disk
    → seed_everything (+ determinism banner)
    → load_base_model + prepare_tokenizer
    → dlm.data.build_dataset (current-doc sections + replay rows)
    → build_or_resume_adapter
    → SFTTrainer.train  (wrapped in OOM guard on CUDA)
    → state_sidecar.save_state  (training_state.pt + sha256)
    → checkpoint_commit.commit_version  (two-phase atomic flip)
    → manifest.training_runs.append

The orchestrator is deliberately thin — each step lives in its own
module and is individually unit-testable. The heavy path (actual HF
training loop) is exercised by the slow-marked integration test
(`tests/integration/train/test_one_cycle_tinymodel.py`).

The `_make_sft_trainer` factory is a seam: unit tests substitute a
stub that captures args + returns a mock with a `.train()` method, so
the orchestration logic is testable without touching HF.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dlm.lock import (
    DlmLock,
    LockDecision,
    LockMode,
    LockValidationError,
    build_lock,
    hardware_tier_from_backend,
    hash_dlm_file,
    load_lock,
    validate_lock,
    write_lock,
)
from dlm.lock import (
    lock_path as _lock_file_path,
)
from dlm.replay import ChangeSet, ReplayStore, SectionSnapshot, diff_against_manifest
from dlm.train.checkpoint_commit import commit_version
from dlm.train.determinism import DeterminismSummary, seed_everything
from dlm.train.disk_preflight import preflight_disk
from dlm.train.logger import Banner, StepLogger, log_path_for
from dlm.train.state_sidecar import (
    STATE_FILENAME,
    STATE_SHA_FILENAME,
    VERSIONS_FILENAME,
    PinnedVersions,
    TrainingState,
    capture_runtime_versions,
    save_state,
)

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.doc.parser import ParsedDlm
    from dlm.hardware.capabilities import Capabilities
    from dlm.hardware.plan import TrainingPlan
    from dlm.store.paths import StorePath

_LOG = logging.getLogger(__name__)

Mode = Literal["fresh", "resume"]


@dataclass(frozen=True)
class TrainingRunResult:
    """Return value of `run()` — what the CLI prints on success."""

    run_id: int
    adapter_version: int
    adapter_path: Path
    log_path: Path
    summary_path: Path
    seed: int
    steps: int
    final_train_loss: float | None
    final_val_loss: float | None
    final_val_perplexity: float | None
    early_stopped: bool
    determinism: DeterminismSummary


def run(
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
    adapter_name: str | None = None,
    world_size: int | None = None,
) -> TrainingRunResult:
    """Execute one training cycle end-to-end.

    `seed` defaults to `parsed.frontmatter.training.seed`. `max_steps`
    caps the run below the num-epochs * dataset-size count; `None`
    lets the full schedule run.

    `lock_mode` controls `dlm.lock` validation (Sprint 15):

    - `"default"` — validate, abort on ERROR, warn on WARN, write on success
    - `"strict"` — upgrade WARN → ERROR (`--strict-lock`)
    - `"update"` — skip validation, always overwrite (`--update-lock`)
    - `"ignore"` — skip validation and don't write (`--ignore-lock`)

    `capabilities` is optional; when passed, the recorded hardware tier
    reflects the real backend + SM. Callers without a doctor report
    (unit tests with mocked trainers) can omit it — the tier falls
    back to `"cpu"`.

    `trainer_factory` is a test seam — pass a callable that returns
    an object with `.train()` + `.state` + `.save_model(dir)` to
    bypass HF loading.

    `adapter_name`, when provided, targets the named multi-adapter
    layout: reads/writes `adapter/<name>/versions/vNNNN/` and
    `adapter/<name>/current.txt` rather than the flat paths. When
    `None`, uses the flat single-adapter layout (backward-compatible
    with `.dlm` files that don't declare `training.adapters`).
    """
    if seed is None:
        seed = parsed.frontmatter.training.seed

    # World-size resolution (Sprint 23 / audit-08 B1). Prefer the
    # caller-passed value (unit tests use this); otherwise read from
    # the DDP env vars `accelerate launch` / `torchrun` set. Default 1
    # in single-process.
    if world_size is None:
        from dlm.train.distributed.rank_env import detect_world_size

        world_size = detect_world_size()

    # 1. Preflight — refuse to start if disk isn't there.
    preflight_disk(store.root, spec, plan)

    # 2. Determinism contract.
    determinism = seed_everything(seed)

    # 3. Open the log + banner.
    store.logs.mkdir(parents=True, exist_ok=True)
    run_id = _next_run_id(store)
    log_path = log_path_for(store.logs, run_id)
    versions = capture_runtime_versions()

    # 3b. Expand frontmatter `training.sources` directives into
    #     synthesized PROSE sections (Sprint 29). The resulting list
    #     merges with `parsed.sections` so every downstream consumer —
    #     diff, dataset builder, retention — treats directive-sourced
    #     and in-body sections identically. Provenance is captured
    #     for the training summary.
    parsed, directive_provenance = _expand_directives(parsed)

    # 4. Content-delta against the previous manifest (audit-04 M1/M2):
    #    feeds replay sampling before training; `change_set.new` is
    #    appended to the corpus + `content_hashes` after training.
    #    Loaded before 3a so the lock record can mirror the
    #    manifest's license_acceptance (audit-05 M1).
    from dlm.store.manifest import load_manifest

    prior_manifest = load_manifest(store.manifest)
    change_set = diff_against_manifest(list(parsed.sections), prior_manifest)

    # 3a. Lock validation (Sprint 15). Build the *candidate* lock for
    #     this run and compare against the prior recorded one. Aborts
    #     via LockValidationError for severity=ERROR mismatches unless
    #     the caller opted into --update-lock / --ignore-lock.
    #
    #     Skipped when `parsed.source_path` is None — unit tests that
    #     synthesize a `ParsedDlm` directly (no on-disk `.dlm`) can't
    #     compute `dlm_sha256`. The CLI path always has a source file,
    #     so production runs always honor the lock.
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
            world_size=world_size,
        )
        if parsed.source_path is not None
        else None
    )
    replay = ReplayStore.at(store.replay_corpus, store.replay_index)

    # Sprint 26: open the metrics recorder early so RunStart lands
    # even if the training loop fails partway. Best-effort — failures
    # here are logged and swallowed to keep training the priority.
    from dlm.metrics import MetricsRecorder, RunEnd, RunStart

    recorder = MetricsRecorder(store.root)
    recorder.record_run_start(
        RunStart(
            run_id=run_id,
            adapter_version=prior_manifest.adapter_version,
            phase="sft",
            seed=seed,
        )
    )

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
            "delta",
            new=[s.section_id for s in change_set.new],
            unchanged=[s.section_id for s in change_set.unchanged],
            removed=list(change_set.removed),
        )

        # 5. Build or resume the SFT trainer.
        sft = _build_trainer(
            store=store,
            parsed=parsed,
            spec=spec,
            plan=plan,
            mode=mode,
            seed=seed,
            max_steps=max_steps,
            factory=trainer_factory,
            change_set=change_set,
            replay=replay,
            adapter_version_for_rng=prior_manifest.adapter_version,
            adapter_name=adapter_name,
        )

        # 6. Run the training loop.
        start = time.perf_counter()
        train_result = sft.train()
        elapsed = time.perf_counter() - start
        _LOG.info("training finished in %.1fs", elapsed)

        steps = int(getattr(sft.state, "global_step", 0))
        final_train_loss = _maybe_float(getattr(train_result, "training_loss", None))

        # 7. Extract final eval metrics from trainer.state.log_history.
        #    Sprint 10: the SFTTrainer calls compute_metrics at each
        #    eval_steps; the last logged `eval_loss` + perplexity are
        #    the authoritative post-run numbers.
        from dlm.eval import summarize_eval_state

        log_history = list(getattr(sft.state, "log_history", []))

        # Audit-08 P2: emit per-step records into the JSONL log from
        # HF's log_history. Sprint 10 shipped `StepLogger.log_step`
        # but never wired a call site — the test_one_cycle test
        # asserts at least one step record lands. We iterate the
        # training-log entries (those with a `loss` field) and emit
        # a step event for each. Live-streaming via a TrainerCallback
        # is a future refinement; post-hoc dump is enough to satisfy
        # the observability contract.
        from dlm.metrics import EvalEvent, StepEvent

        for entry in log_history:
            if not isinstance(entry, dict):
                continue
            step_num = entry.get("step") or entry.get("global_step")
            if step_num is None:
                continue
            step_int = int(step_num)
            # Step rows (training loss + lr + grad_norm).
            if "loss" in entry:
                log.log_step(
                    step=step_int,
                    loss=float(entry["loss"]),
                    lr=float(entry.get("learning_rate", 0.0)),
                    grad_norm=_maybe_float(entry.get("grad_norm")),
                    val_loss=_maybe_float(entry.get("eval_loss")),
                )
                recorder.record_step(
                    StepEvent(
                        run_id=run_id,
                        step=step_int,
                        loss=float(entry["loss"]),
                        lr=_maybe_float(entry.get("learning_rate")),
                        grad_norm=_maybe_float(entry.get("grad_norm")),
                    )
                )
            # Eval rows (separate DB table). HF log_history interleaves
            # train `loss` entries and eval `eval_loss` entries; record
            # whichever is present.
            if "eval_loss" in entry:
                recorder.record_eval(
                    EvalEvent(
                        run_id=run_id,
                        step=step_int,
                        val_loss=_maybe_float(entry.get("eval_loss")),
                        perplexity=_maybe_float(entry.get("eval_perplexity")),
                    )
                )

        # NaN-eval guard (redundant with the weight gate, intentional).
        # Raises `NaNEvalError` so the run fails before we touch disk.
        from dlm.train.integrity import assert_eval_finite

        assert_eval_finite(log_history)

        eval_summary = summarize_eval_state(log_history)
        final_val_loss = eval_summary["final_val_loss"]
        final_val_perplexity = eval_summary["final_val_perplexity"]

        # Per-mode val-loss split (audit-08 N9). Sprint 19 reserved
        # TrainingSummary.val_loss_cpt/val_loss_sft but left them
        # unwired. Run one post-train eval per non-empty mode subset
        # and populate the fields with the resulting eval_loss. Safe:
        # each call is guarded, failures degrade to None.
        from dlm.eval.mode_split import compute_val_loss_by_mode

        val_loss_cpt, val_loss_sft = compute_val_loss_by_mode(
            sft, getattr(sft, "eval_dataset", None)
        )

        # Early-stop detection (audit-05 M2). Prefer HF's real signal
        # (the callback sets `control.should_training_stop`); fall back
        # to the heuristic if the trainer object doesn't expose it
        # (e.g., unit-test mock).
        from dlm.eval import was_early_stopped

        hf_flag = _hf_early_stop_flag(sft)
        early_stopped = (
            hf_flag
            if hf_flag is not None
            else was_early_stopped(
                max_steps_ran=steps,
                configured_max_steps=max_steps,
                num_epochs_done=float(getattr(sft.state, "epoch", 0.0)),
            )
        )

        # 8. Two-phase commit: save adapter + state into a pending
        #    version dir, then flip the current pointer atomically. If
        #    the writer raises `NaNWeightsError`, `commit_version`
        #    renames the pending dir to `{name}-rejected` so the bad
        #    weights are preserved for postmortem but never promoted
        #    to `current.txt`.
        adapter_path = commit_version(
            store,
            lambda pending: _write_checkpoint(
                pending_dir=pending,
                sft=sft,
                spec=spec,
                versions=versions,
                use_qlora=plan.use_qlora,
            ),
            adapter_name=adapter_name,
        )
        adapter_version = int(adapter_path.name.lstrip("v"))

        # 9. Append NEW section snapshots to the replay corpus so the
        #    next `dlm train` draws from today's training signal.
        #    `changed` is reserved empty under the current content-
        #    addressed design; only `new` needs persisting here.
        _append_change_set_to_replay(replay, change_set, run_id=run_id)

        # 10. Write the human-readable TrainingSummary JSON alongside
        #     the JSONL log (audit-05 M3: written BEFORE the manifest
        #     append so the relative path can be recorded).
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
            val_loss_cpt=val_loss_cpt,
            val_loss_sft=val_loss_sft,
            early_stopped=early_stopped,
            duration_seconds=elapsed,
            determinism=determinism,
            source_directives=directive_provenance,
        )

        # 11. Append the training-run summary + refresh `content_hashes`
        #     on the manifest. The delta for the NEXT run is computed
        #     against the hashes written here. `summary_path` is
        #     stored relative to store root so migrations that move the
        #     store root don't orphan the link.
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
            adapter_name=adapter_name,
        )

        # 12. Persist the lock (Sprint 15). `ignore_lock` mode suppresses
        #     the write; every other mode updates `dlm.lock` after the
        #     manifest append so a partial failure can't leave a lock
        #     newer than its run summary.
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
                world_size=world_size,
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

    recorder.record_run_end(RunEnd(run_id=run_id, status="ok"))

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


def _default_eval_steps(max_steps: int | None) -> int:
    """Pick a reasonable `eval_steps` cadence (audit-05 M2).

    Four eval rounds per run gives enough signal to watch a loss curve
    without dominating wall-clock with eval overhead. When `max_steps`
    isn't set we're running to num_epochs — fall back to 50 steps,
    which is "often enough to see a trend" on tiny-model CI runs.
    """
    if max_steps is not None and max_steps > 0:
        return max(1, max_steps // 4)
    return 50


def _default_early_stop_config() -> Any:
    """Patience + threshold defaults until TrainingConfig extends (Sprint 12b)."""
    from dlm.eval import EarlyStopConfig

    return EarlyStopConfig(patience=3, threshold=0.0, metric="eval_loss")


def _hf_early_stop_flag(sft: Any) -> bool | None:
    """Read HF's real early-stop signal, or return None if unavailable.

    `trainer.control.should_training_stop` is True iff a callback
    (e.g., `EarlyStoppingCallback`) asked the loop to exit. Mock
    trainers in unit tests don't expose this; return None to let the
    caller fall back to the heuristic.
    """
    control = getattr(sft, "control", None)
    if control is None:
        return None
    flag = getattr(control, "should_training_stop", None)
    if flag is None:
        return None
    return bool(flag)


# --- internal seams ----------------------------------------------------------


def _build_trainer(
    *,
    store: StorePath,
    parsed: ParsedDlm,
    spec: BaseModelSpec,
    plan: TrainingPlan,
    mode: Mode,
    seed: int,
    max_steps: int | None,
    factory: Callable[..., Any] | None,
    change_set: ChangeSet,
    replay: ReplayStore,
    adapter_version_for_rng: int,
    adapter_name: str | None = None,
) -> Any:
    """Assemble model + tokenizer + dataset + SFTTrainer.

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
            mode=mode,
            seed=seed,
            max_steps=max_steps,
            change_set=change_set,
            replay=replay,
            adapter_version_for_rng=adapter_version_for_rng,
            adapter_name=adapter_name,
        )

    # Exercised by the slow-marked integration test
    # `tests/integration/train/test_one_cycle_tinymodel.py` rather than
    # unit-tested (instantiating SFTTrainer requires a real HF model).
    return _build_real_trainer(  # pragma: no cover
        store=store,
        parsed=parsed,
        spec=spec,
        plan=plan,
        mode=mode,
        seed=seed,
        max_steps=max_steps,
        change_set=change_set,
        replay=replay,
        adapter_version_for_rng=adapter_version_for_rng,
        adapter_name=adapter_name,
    )


def _build_real_trainer(  # pragma: no cover
    *,
    store: StorePath,
    parsed: ParsedDlm,
    spec: BaseModelSpec,
    plan: TrainingPlan,
    mode: Mode,
    seed: int,
    max_steps: int | None,
    change_set: ChangeSet,
    replay: ReplayStore,
    adapter_version_for_rng: int,
    adapter_name: str | None = None,
) -> Any:
    # Deferred imports — heavy ML stack only touched on the real path.
    from trl import SFTConfig, SFTTrainer  # type: ignore[attr-defined]

    from dlm.data import build_dataset, make_formatting_func, prepare_tokenizer
    from dlm.train.adapter import build_or_resume_adapter
    from dlm.train.loader import load_base_model

    base_model = load_base_model(spec, plan)
    tok_bringup = prepare_tokenizer(spec.hf_id, spec.revision)

    replay_rows = _sample_replay_rows(
        replay,
        change_set=change_set,
        seed=seed,
        adapter_version=adapter_version_for_rng,
    )

    train_ds, val_ds = build_dataset(
        list(parsed.sections),
        seed=seed,
        val_frac=0.1,
        replay_rows=replay_rows or None,
    )

    resume_path = (
        (
            store.resolve_current_adapter_for(adapter_name)
            if adapter_name is not None
            else store.resolve_current_adapter()
        )
        if mode == "resume"
        else None
    )

    # Multi-adapter docs override the flat LoRA knobs with the
    # per-adapter `AdapterConfig`. Single-adapter docs and the
    # trainer_factory test path keep reading the flat fields.
    eff_lora_r, eff_lora_alpha, eff_lora_dropout, eff_lr = _resolve_adapter_hparams(
        parsed, adapter_name
    )

    peft_model = build_or_resume_adapter(
        base_model,
        spec,
        lora_r=eff_lora_r,
        lora_alpha=eff_lora_alpha,
        lora_dropout=eff_lora_dropout,
        tokenizer_grew=tok_bringup.tokenizer_grew,
        mode=mode,
        resume_path=resume_path,
        use_qlora=plan.use_qlora,
        gradient_checkpointing=plan.gradient_checkpointing,
    )

    # Eval cadence (audit-05 M2): without eval_strategy="steps" + eval_steps,
    # `trainer.state.log_history` never gains `eval_loss` entries and
    # `summarize_eval_state` always returns None. Default cadence: four
    # eval rounds per run (or every 50 steps if running to epoch end).
    eval_steps = _default_eval_steps(max_steps)
    early_stop_cfg = _default_early_stop_config()

    # Very small documents (tiny-model CI runs, quickstart docs) can hash
    # all rows to the train side, leaving val_ds empty. TRL's SFTTrainer
    # does `next(iter(eval_dataset))` unconditionally, so an empty
    # eval_dataset crashes construction. When that happens, turn off
    # the eval+early-stop machinery for this run — the training contract
    # still holds, we just skip the val-loss curve.
    has_val = len(val_ds) > 0

    sft_config_kwargs: dict[str, Any] = {
        "output_dir": str(store.logs / f"sft-run-{seed}"),
        "num_train_epochs": parsed.frontmatter.training.num_epochs,
        "per_device_train_batch_size": plan.micro_batch_size,
        "gradient_accumulation_steps": plan.grad_accum,
        "learning_rate": eff_lr,
        "lr_scheduler_type": parsed.frontmatter.training.lr_scheduler,
        "warmup_ratio": parsed.frontmatter.training.warmup_ratio,
        "max_steps": max_steps if max_steps is not None else -1,
        # Honor the frontmatter `sequence_len` (default 2048) instead
        # of TRL's built-in 1024 default. Also the value the tokenized-
        # section cache keys on — a silent mismatch between SFTConfig's
        # effective max and the cache key invariant would silently
        # invalidate or, worse, serve stale tokens.
        "max_length": parsed.frontmatter.training.sequence_len,
        "seed": seed,
        "data_seed": seed,
        "bf16": plan.precision == "bf16",
        "fp16": plan.precision == "fp16",
        "report_to": ["none"],
        "save_strategy": "no",  # we own checkpoint commit
        # Modern transformers refuses load_best_model_at_end=True when
        # save_strategy="no" (it has no checkpoints to reload). We own
        # the checkpoint write lifecycle, so we keep save off and take
        # the last-step weights at commit time. Early stopping still
        # fires via the callback.
        "load_best_model_at_end": False,
    }
    if has_val:
        sft_config_kwargs.update(
            eval_strategy="steps",
            eval_steps=eval_steps,
            metric_for_best_model=early_stop_cfg.metric,
            greater_is_better=early_stop_cfg.greater_is_better,
        )

    # CPT refinements (Phase 4): override schedule when prose rows
    # dominate. Reads `training.cpt.schedule` and the actual row mix;
    # `auto` flips to DAPT above the 70% threshold, otherwise the
    # user's `lr_scheduler` setting stands.
    from dlm.train.cpt.runtime import (
        cpt_row_fraction,
        dapt_sft_config_overrides,
        select_schedule,
    )

    cpt_cfg = parsed.frontmatter.training.cpt
    fraction = cpt_row_fraction(list(train_ds))
    chosen = select_schedule(cpt_cfg.schedule, fraction)
    if chosen == "dapt":
        sft_config_kwargs.update(dapt_sft_config_overrides())
        _LOG.info(
            "CPT schedule: dapt (cpt_fraction=%.2f, setting=%s)",
            fraction,
            cpt_cfg.schedule,
        )

    sft_config = SFTConfig(**sft_config_kwargs)

    from dlm.eval import build_callback

    # Newer TRL releases (>=0.9) renamed `tokenizer` → `processing_class`.
    trainer_kwargs: dict[str, Any] = {
        "model": peft_model,
        "processing_class": tok_bringup.tokenizer,
        "train_dataset": train_ds,
        "formatting_func": make_formatting_func(tok_bringup.tokenizer),
        "args": sft_config,
    }
    callbacks: list[Any] = []
    if has_val:
        trainer_kwargs["eval_dataset"] = val_ds
        callbacks.append(build_callback(early_stop_cfg))

    if cpt_cfg.embed_warmup_steps > 0:
        from dlm.train.cpt.embed_warmup import EmbedWarmupCallback

        callbacks.append(EmbedWarmupCallback(peft_model, n_steps=cpt_cfg.embed_warmup_steps))
        _LOG.warning(
            "embed warm-up enabled for %d steps — adapter size will inflate "
            "by vocab_size × hidden_dim",
            cpt_cfg.embed_warmup_steps,
        )

    # Vocab-gap report: emit once per run so users can spot a
    # mismatched tokenizer before burning compute.
    if train_ds:
        _emit_vocab_gap_report(train_ds, tok_bringup.tokenizer)

    if callbacks:
        trainer_kwargs["callbacks"] = callbacks
    return SFTTrainer(**trainer_kwargs)


def _emit_vocab_gap_report(train_ds: Any, tokenizer: Any) -> None:  # pragma: no cover
    """Run the tokenizer over concatenated prose rows and log the fit report.

    Falls back quietly on any error — a descriptive-only report should
    never break training.
    """
    from dlm.train.cpt.vocab_gap import render_report, report

    try:
        prose = [row["text"] for row in train_ds if row.get("text") is not None]
        if not prose:
            return
        joined = "\n\n".join(prose)
        r = report(joined, tokenizer)
        _LOG.info("\n%s", render_report(r))
    except Exception as exc:  # defensive: report is advisory
        _LOG.debug("vocab-gap report skipped: %s", exc)


def _write_checkpoint(
    *,
    pending_dir: Path,
    sft: Any,
    spec: BaseModelSpec,
    versions: PinnedVersions,
    use_qlora: bool,
) -> None:
    """Populate a pending adapter version dir with everything to resume.

    Called by `commit_version` inside the two-phase lifecycle.
    """
    # 1. Adapter weights + config.
    sft.save_model(str(pending_dir))

    # 2. Integrity gate — refuse to commit NaN/inf weights (discovered on
    #    MPS + tiny-data + no-warmup runs; the bad adapter was otherwise
    #    silently promoted to current.txt and all downstream consumers
    #    produced NaN logits). Checked after save so postmortem has the
    #    bad tensors on disk in `{pending}-rejected/` when the caller
    #    renames it.
    from dlm.train.integrity import assert_finite_adapter

    assert_finite_adapter(sft.model)

    # 3. Training state sidecar (optimizer/scheduler/RNG/step counters).
    state = _snapshot_training_state(sft, spec=spec, versions=versions, use_qlora=use_qlora)
    save_state(pending_dir, state)


def _snapshot_training_state(
    sft: Any,
    *,
    spec: BaseModelSpec,
    versions: PinnedVersions,
    use_qlora: bool,
) -> TrainingState:
    """Gather everything needed for a bit-exact resume.

    Wrapped in a helper so unit tests can verify the field set without
    standing up a real SFTTrainer.
    """
    import random as _random

    import numpy as _np
    import torch as _torch

    cuda_state = _torch.cuda.get_rng_state_all() if _torch.cuda.is_available() else None
    scaler_state = None
    scaler = getattr(sft, "scaler", None)
    if scaler is not None and hasattr(scaler, "state_dict"):
        scaler_state = scaler.state_dict()

    return TrainingState(
        optimizer_state_dict=sft.optimizer.state_dict() if sft.optimizer else {},
        scheduler_state_dict=(
            sft.lr_scheduler.state_dict() if getattr(sft, "lr_scheduler", None) else {}
        ),
        scaler_state_dict=scaler_state,
        torch_rng_state=_torch.get_rng_state(),
        cuda_rng_state=cuda_state,
        numpy_rng_state=_np.random.get_state(),
        python_random_state=_random.getstate(),
        global_step=int(getattr(sft.state, "global_step", 0)),
        epoch=float(getattr(sft.state, "epoch", 0.0)),
        best_val_loss=_maybe_float(getattr(sft.state, "best_metric", None)),
        dlm_manifest_hash=None,  # Sprint 13 fills this in
        base_model_revision=spec.revision,
        pinned_versions=versions,
        use_qlora=use_qlora,
    )


def _resolve_adapter_hparams(
    parsed: ParsedDlm, adapter_name: str | None
) -> tuple[int, int, float, float]:
    """Return `(lora_r, lora_alpha, lora_dropout, learning_rate)` for this run.

    When `adapter_name` is set and the doc declares `training.adapters`,
    read from the matching `AdapterConfig`. Otherwise, fall through to
    the flat `TrainingConfig` fields (single-adapter or `None` caller).
    """
    training = parsed.frontmatter.training
    if adapter_name is not None and training.adapters is not None:
        cfg = training.adapters.get(adapter_name)
        if cfg is not None:
            return (cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout, cfg.learning_rate)
    return (
        training.lora_r,
        training.lora_alpha,
        training.lora_dropout,
        training.learning_rate,
    )


def _sample_replay_rows(
    replay: ReplayStore,
    *,
    change_set: ChangeSet,
    seed: int,
    adapter_version: int,
) -> list[dict[str, Any]]:
    """Draw recency-weighted replay rows for this training run.

    `k = max(32, 2 × |new|)` matches the Sprint 08 design note: the
    sample is an anti-forgetting counterweight to the fresh content's
    gradient signal. Returns an empty list when the corpus is cold
    (first `dlm train` on a store).
    """
    import random as _random
    from datetime import UTC, datetime

    entries = replay.load()
    if not entries:
        return []

    k = max(32, 2 * len(change_set.new))
    rng = _random.Random(seed + adapter_version)
    now = datetime.now(UTC).replace(tzinfo=None, microsecond=0)
    return replay.sample_rows(k=k, now=now, rng=rng)


def _write_training_summary(
    *,
    store: StorePath,
    log_path: Path,
    run_id: int,
    adapter_version: int,
    seed: int,
    steps: int,
    final_train_loss: float | None,
    final_val_loss: float | None,
    final_val_perplexity: float | None,
    early_stopped: bool,
    duration_seconds: float,
    determinism: DeterminismSummary,
    val_loss_cpt: float | None = None,
    val_loss_sft: float | None = None,
    source_directives: tuple[Any, ...] = (),
) -> Path:
    """Write `logs/train-*.summary.json` next to the JSONL log."""
    from dlm.eval import (
        SourceProvenanceRecord,
        TrainingSummary,
        save_summary,
        summary_path_for,
    )

    # Re-derive the timestamp portion from the log filename so summary + log share a stem.
    stem = log_path.stem  # "train-000001-2026-04-18T10:15:23"
    ts = stem.split("-", 2)[-1] if "-" in stem else ""
    summary_path = summary_path_for(store.logs, run_id, ts)

    records = [
        SourceProvenanceRecord(
            path=p.path,
            file_count=p.file_count,
            total_bytes=p.total_bytes,
            skipped_binary=p.skipped_binary,
            skipped_encoding=p.skipped_encoding,
            skipped_over_size=p.skipped_over_size,
        )
        for p in source_directives
    ]

    summary = TrainingSummary(
        run_id=run_id,
        adapter_version=adapter_version,
        seed=seed,
        steps=steps,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        final_val_perplexity=final_val_perplexity,
        val_loss_cpt=val_loss_cpt,
        val_loss_sft=val_loss_sft,
        early_stopped=early_stopped,
        duration_seconds=duration_seconds,
        determinism_class=determinism.class_,
        source_directives=records,
    )
    save_summary(summary_path, summary)
    return summary_path


def _append_change_set_to_replay(
    replay: ReplayStore,
    change_set: ChangeSet,
    *,
    run_id: int,
) -> None:
    """Persist `change_set.new` into the replay corpus for future runs.

    Idempotent-ish: a section appended twice produces two frames with
    the same `section_id` but different `last_seen_at`. The sampler
    uses `added_at` for recency weighting so double-appends are a
    harmless no-op beyond mild corpus growth — eviction (Sprint 08)
    reclaims the bytes.
    """
    if not change_set.new:
        return
    now = _utc_naive()
    snapshots = [
        SectionSnapshot(
            section_id=section.section_id,
            section_type=section.type.value,
            content=section.content,
            first_seen_at=now,
            last_seen_at=now,
            training_runs_seen=[run_id],
        )
        for section in change_set.new
    ]
    replay.append_many(snapshots)


def _append_training_run(
    *,
    store: StorePath,
    run_id: int,
    adapter_version: int,
    seed: int,
    steps: int,
    final_train_loss: float | None,
    final_val_loss: float | None,
    base_model_revision: str,
    versions: PinnedVersions,
    current_sections: list[Any],
    summary_path: Path,
    adapter_name: str | None = None,
) -> None:
    """Append a TrainingRunSummary + refresh `content_hashes` (audit-04 M2).

    `content_hashes` is overwritten with the full set of current-document
    section ids. `diff_against_manifest` keys on this dict, so updating
    it here is what makes the NEXT run classify things correctly as
    `new`/`unchanged`/`removed`.

    `summary_path` is stored as a string relative to the store root
    (audit-05 M3) so `dlm show` can load the summary without globbing
    the logs directory.

    `adapter_name` (audit-07 M1): when set, the new summary is tagged
    with the named adapter AND `Manifest.adapter_versions[name]` is
    updated instead of the flat `adapter_version`. Flat-doc runs
    continue to bump the top-level field only.

    Manifest reads/writes go through the Sprint 04 atomic I/O path so
    a concurrent reader never sees a torn file.
    """
    from dlm.store.manifest import TrainingRunSummary, load_manifest, save_manifest

    try:
        summary_rel = str(summary_path.resolve().relative_to(store.root.resolve()))
    except ValueError:
        # Summary path isn't under store root — record absolute. Should
        # not happen in practice (we always write into `store.logs`);
        # the fallback keeps the manifest validator from failing.
        summary_rel = str(summary_path)

    now = _utc_naive()
    summary = TrainingRunSummary(
        run_id=run_id,
        started_at=now,
        ended_at=now,
        adapter_version=adapter_version,
        base_model_revision=base_model_revision,
        seed=seed,
        steps=steps,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        status="completed",
        pinned_versions={k: v for k, v in versions.items() if isinstance(v, str)},
        summary_path=summary_rel,
        adapter_name=adapter_name,
    )

    manifest_path = store.manifest
    manifest = load_manifest(manifest_path)
    new_hashes = {s.section_id: s.section_id for s in current_sections}
    update_fields: dict[str, Any] = {
        "training_runs": [*manifest.training_runs, summary],
        "updated_at": now,
        "content_hashes": new_hashes,
    }
    if adapter_name is None:
        # Flat doc: bump the top-level field as before.
        update_fields["adapter_version"] = adapter_version
    else:
        # Multi-adapter doc: the top-level field has no coherent meaning
        # (audit-07 M1). Record the per-adapter version instead.
        update_fields["adapter_versions"] = {
            **manifest.adapter_versions,
            adapter_name: adapter_version,
        }
    updated = manifest.model_copy(update=update_fields)
    save_manifest(manifest_path, updated)


def _next_run_id(store: StorePath) -> int:
    """Smallest available positive run_id.

    Reads the manifest's `training_runs` list and returns `max+1`, or
    `1` on a fresh store. `TrainingRunSummary.run_id` has `ge=1` so we
    start at 1.
    """
    from dlm.store.manifest import load_manifest

    if not store.manifest.exists():
        return 1
    manifest = load_manifest(store.manifest)
    if not manifest.training_runs:
        return 1
    return max(r.run_id for r in manifest.training_runs) + 1


_TRAIN_TO_LOCK_DETERMINISM: dict[str, str] = {
    # `DeterminismSummary.class_` (training vocabulary) → DlmLock vocabulary.
    # Trainer uses underscores + "loose"; the lock schema matches the
    # hardware module's hyphenated naming.
    "strict": "strong",
    "best_effort": "best-effort",
    "loose": "advisory",
}


def _build_candidate_lock(
    *,
    parsed: ParsedDlm,
    spec: BaseModelSpec,
    seed: int,
    run_id: int,
    versions: PinnedVersions,
    determinism_class: str,
    capabilities: Capabilities | None,
    license_acceptance: object | None = None,
    world_size: int = 1,
) -> DlmLock:
    """Assemble the `DlmLock` describing this run.

    `license_acceptance` is the record (if any) mirrored from
    `manifest.license_acceptance`. Audit-05 M1 wired this through so
    the lock's reproducibility contract actually carries the gated-base
    acceptance fingerprint.

    `world_size` (Sprint 23 / audit-08 B1) records the DDP rank count
    so a resume with a different world_size triggers the policy WARN
    at `dlm.lock.policy._rule_world_size`. Default 1 for
    single-process runs.
    """
    if parsed.source_path is None:
        raise ValueError("parsed.source_path is required to build a dlm.lock record")

    sm = capabilities.sm if capabilities is not None else None
    backend_value = capabilities.backend.value if capabilities is not None else None
    tier = hardware_tier_from_backend(backend_value, sm=sm)

    # `versions` is a TypedDict(total=False) with mixed str | None.
    pinned: dict[str, str] = {k: v for k, v in versions.items() if isinstance(v, str)}

    mapped_class = _TRAIN_TO_LOCK_DETERMINISM.get(determinism_class, determinism_class)

    return build_lock(
        dlm_id=parsed.frontmatter.dlm_id,
        dlm_sha256=hash_dlm_file(parsed.source_path),
        base_model_revision=spec.revision,
        hardware_tier=tier,
        seed=seed,
        determinism_class=mapped_class,  # type: ignore[arg-type]
        run_id=run_id,
        pinned_versions=pinned,
        cuda_version=capabilities.cuda_version if capabilities is not None else None,
        rocm_version=capabilities.rocm_version if capabilities is not None else None,
        license_acceptance=license_acceptance,  # type: ignore[arg-type]
        world_size=world_size,
    )


def _validate_or_abort_lock(
    *,
    store: StorePath,
    parsed: ParsedDlm,
    spec: BaseModelSpec,
    seed: int,
    run_id: int,
    versions: PinnedVersions,
    determinism_class: str,
    capabilities: Capabilities | None,
    lock_mode: LockMode,
    license_acceptance: object | None = None,
    world_size: int = 1,
) -> LockDecision:
    """Compare this run's candidate lock against the prior recorded one.

    Raises `LockValidationError` on `abort`; otherwise returns the
    `LockDecision` so the caller knows whether to write post-success.
    """
    candidate = _build_candidate_lock(
        parsed=parsed,
        spec=spec,
        seed=seed,
        run_id=run_id,
        versions=versions,
        determinism_class=determinism_class,
        capabilities=capabilities,
        license_acceptance=license_acceptance,
        world_size=world_size,
    )
    try:
        prior = load_lock(store.root)
    except Exception:
        # Audit-05 N5: a corrupt `dlm.lock` on disk would normally kill
        # the run at load time. Under `--update-lock` the operator has
        # explicitly opted to overwrite the file; treat the parse
        # failure as "prior is unusable → treat as missing" so the
        # update mode can actually rescue a broken lock. Any other mode
        # re-raises (including --ignore-lock, which explicitly says
        # "don't touch the file").
        if lock_mode != "update":
            raise
        prior = None
    decision = validate_lock(prior, candidate, mode=lock_mode)

    if decision.action == "abort":
        reasons = [msg for _sev, msg in decision.mismatches]
        raise LockValidationError(path=_lock_file_path(store.root), reasons=reasons)
    if decision.action == "proceed_with_warnings":
        for _sev, msg in decision.mismatches:
            _LOG.warning("dlm.lock drift: %s", msg)
    return decision


def _persist_lock(
    *,
    store: StorePath,
    parsed: ParsedDlm,
    spec: BaseModelSpec,
    seed: int,
    run_id: int,
    versions: PinnedVersions,
    determinism_class: str,
    capabilities: Capabilities | None,
    license_acceptance: object | None = None,
    world_size: int = 1,
) -> None:
    """Write the post-run lock. Separate from validation so a failed
    training doesn't leave a fresh lock behind.
    """
    candidate = _build_candidate_lock(
        parsed=parsed,
        spec=spec,
        seed=seed,
        run_id=run_id,
        versions=versions,
        determinism_class=determinism_class,
        capabilities=capabilities,
        license_acceptance=license_acceptance,
        world_size=world_size,
    )
    write_lock(store.root, candidate)


def _utc_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _expand_directives(
    parsed: ParsedDlm,
) -> tuple[ParsedDlm, tuple[Any, ...]]:
    """Resolve `training.sources` directives into synthesized sections.

    Returns a (possibly-new) `ParsedDlm` whose `sections` include both
    the original in-body sections and the directive-sourced ones, plus
    the per-directive provenance tuple for the training summary.

    When the frontmatter declares no `training.sources`, returns the
    input parsed unchanged and an empty provenance tuple — zero-cost
    fast path for the common case.

    Directive expansion needs a filesystem anchor. We use
    `parsed.source_path.parent` when the parser saw an on-disk file;
    for synthesized `ParsedDlm` values (unit tests, in-memory
    pipelines) with `source_path=None` we fall back to the current
    working directory. That keeps tests that don't touch directives
    unaffected.
    """
    import dataclasses

    from dlm.directives import expand_sources

    if parsed.frontmatter.training.sources is None:
        return parsed, ()

    base_path = parsed.source_path.parent if parsed.source_path is not None else Path.cwd()
    result = expand_sources(parsed, base_path=base_path)
    if not result.sections:
        return parsed, result.provenance

    merged = tuple(parsed.sections) + result.sections
    new_parsed = dataclasses.replace(parsed, sections=merged)
    _LOG.info(
        "directives: expanded %d file(s) across %d source(s)",
        len(result.sections),
        len(result.provenance),
    )
    return new_parsed, result.provenance


# Re-export the state-sidecar filenames so callers (e.g., pack tooling)
# can reach them via `dlm.train.trainer.STATE_FILENAME` rather than
# reaching into the private module directly.
__all__ = [
    "STATE_FILENAME",
    "STATE_SHA_FILENAME",
    "VERSIONS_FILENAME",
    "TrainingRunResult",
    "run",
]
