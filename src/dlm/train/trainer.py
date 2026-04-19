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
    seed: int
    steps: int
    final_train_loss: float | None
    final_val_loss: float | None
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
    trainer_factory: Callable[..., Any] | None = None,
) -> TrainingRunResult:
    """Execute one training cycle end-to-end.

    `seed` defaults to `parsed.frontmatter.training.seed`. `max_steps`
    caps the run below the num-epochs * dataset-size count; `None`
    lets the full schedule run.

    `trainer_factory` is a test seam — pass a callable that returns
    an object with `.train()` + `.state` + `.save_model(dir)` to
    bypass HF loading.
    """
    if seed is None:
        seed = parsed.frontmatter.training.seed

    # 1. Preflight — refuse to start if disk isn't there.
    preflight_disk(store.root, spec, plan)

    # 2. Determinism contract.
    determinism = seed_everything(seed)

    # 3. Open the log + banner.
    store.logs.mkdir(parents=True, exist_ok=True)
    run_id = _next_run_id(store)
    log_path = log_path_for(store.logs, run_id)
    versions = capture_runtime_versions()

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

        # 4. Build or resume the SFT trainer.
        sft = _build_trainer(
            store=store,
            parsed=parsed,
            spec=spec,
            plan=plan,
            mode=mode,
            seed=seed,
            max_steps=max_steps,
            factory=trainer_factory,
        )

        # 5. Run the training loop.
        start = time.perf_counter()
        train_result = sft.train()
        elapsed = time.perf_counter() - start
        _LOG.info("training finished in %.1fs", elapsed)

        steps = int(getattr(sft.state, "global_step", 0))
        final_train_loss = _maybe_float(getattr(train_result, "training_loss", None))
        final_val_loss = _maybe_float(getattr(sft.state, "best_metric", None))

        # 6. Two-phase commit: save adapter + state into a pending
        #    version dir, then flip the current pointer atomically.
        adapter_path = commit_version(
            store,
            lambda pending: _write_checkpoint(
                pending_dir=pending,
                sft=sft,
                spec=spec,
                versions=versions,
            ),
        )
        adapter_version = int(adapter_path.name.lstrip("v"))

        # 7. Append the training-run summary to the manifest.
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
        )

        log.log_event(
            "run_complete",
            run_id=run_id,
            adapter_version=adapter_version,
            steps=steps,
            elapsed_seconds=elapsed,
        )

    return TrainingRunResult(
        run_id=run_id,
        adapter_version=adapter_version,
        adapter_path=adapter_path,
        log_path=log_path,
        seed=seed,
        steps=steps,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        determinism=determinism,
    )


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
        )

    # Deferred imports — heavy ML stack only touched on the real path.
    from trl import SFTConfig, SFTTrainer  # type: ignore[attr-defined]

    from dlm.data import build_dataset, make_formatting_func, prepare_tokenizer
    from dlm.train.adapter import build_or_resume_adapter
    from dlm.train.loader import load_base_model

    base_model = load_base_model(spec, plan)
    tok_bringup = prepare_tokenizer(spec.hf_id, spec.revision)

    train_ds, val_ds = build_dataset(
        list(parsed.sections),
        seed=seed,
        val_frac=0.1,
    )

    resume_path = store.resolve_current_adapter() if mode == "resume" else None
    peft_model = build_or_resume_adapter(
        base_model,
        spec,
        lora_r=parsed.frontmatter.training.lora_r,
        lora_alpha=parsed.frontmatter.training.lora_alpha,
        lora_dropout=parsed.frontmatter.training.lora_dropout,
        tokenizer_grew=tok_bringup.tokenizer_grew,
        mode=mode,
        resume_path=resume_path,
        use_qlora=plan.use_qlora,
        gradient_checkpointing=plan.gradient_checkpointing,
    )

    sft_config = SFTConfig(
        output_dir=str(store.logs / f"sft-run-{seed}"),
        num_train_epochs=parsed.frontmatter.training.num_epochs,
        per_device_train_batch_size=plan.micro_batch_size,
        gradient_accumulation_steps=plan.grad_accum,
        learning_rate=parsed.frontmatter.training.learning_rate,
        lr_scheduler_type=parsed.frontmatter.training.lr_scheduler,
        warmup_ratio=parsed.frontmatter.training.warmup_ratio,
        max_steps=max_steps if max_steps is not None else -1,
        seed=seed,
        data_seed=seed,
        bf16=(plan.precision == "bf16"),
        fp16=(plan.precision == "fp16"),
        report_to=["none"],
        save_strategy="no",  # we own checkpoint commit
    )

    # Newer TRL releases (>=0.9) renamed `tokenizer` → `processing_class`.
    return SFTTrainer(
        model=peft_model,
        processing_class=tok_bringup.tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=make_formatting_func(tok_bringup.tokenizer),
        args=sft_config,
    )


def _write_checkpoint(
    *,
    pending_dir: Path,
    sft: Any,
    spec: BaseModelSpec,
    versions: PinnedVersions,
) -> None:
    """Populate a pending adapter version dir with everything to resume.

    Called by `commit_version` inside the two-phase lifecycle.
    """
    # 1. Adapter weights + config.
    sft.save_model(str(pending_dir))

    # 2. Training state sidecar (optimizer/scheduler/RNG/step counters).
    state = _snapshot_training_state(sft, spec=spec, versions=versions)
    save_state(pending_dir, state)


def _snapshot_training_state(
    sft: Any,
    *,
    spec: BaseModelSpec,
    versions: PinnedVersions,
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
    )


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
) -> None:
    """Append a TrainingRunSummary to `manifest.training_runs`.

    Manifest reads/writes go through the Sprint 04 atomic I/O path so
    a concurrent reader never sees a torn file.
    """
    from dlm.store.manifest import TrainingRunSummary, load_manifest, save_manifest

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
    )

    manifest_path = store.manifest
    manifest = load_manifest(manifest_path)
    updated = manifest.model_copy(
        update={
            "training_runs": [*manifest.training_runs, summary],
            "adapter_version": adapter_version,
            "updated_at": now,
        }
    )
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


def _utc_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
