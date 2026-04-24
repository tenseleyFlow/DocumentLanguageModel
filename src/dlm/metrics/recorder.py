"""Writer-side API for the metrics store.

`MetricsRecorder` opens per-write connections rather than holding a
long-lived one — SQLite's WAL handles the concurrency, and keeping
connections short avoids lock conflicts with `dlm metrics watch`.

Two entry shapes:

- Direct: trainer code calls `recorder.record_run_start(RunStart(...))`
  explicitly. Used from `trainer.run` where we know the run_id.
- Callback: `DlmTrainerCallback` plugs into HuggingFace's
  `TrainerCallback` protocol so TRL's `on_log` / `on_evaluate` /
  `on_train_end` map onto the event stream. The trainer attaches it
  to the live `SFTTrainer` so step + eval records stream into SQLite
  and the JSONL step log as each log entry is produced — no post-hoc
  `state.log_history` walk after training completes.

Failure policy is intentionally split:

- run-start / run-end writes are anchor records and raise on SQLite
  failure.
- incremental writes (steps / evals / tokenization / gate / export)
  stay best-effort by default, but log at ERROR once per event stream.
- `dlm train --strict-metrics` promotes every write failure to a hard
  error by constructing the recorder in strict mode.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlm.metrics.db import connect
from dlm.metrics.events import (
    EvalEvent,
    ExportEvent,
    GateEvent,
    PreferenceMineEvent,
    RunEnd,
    RunStart,
    StepEvent,
    TokenizationEvent,
)

if TYPE_CHECKING:
    from dlm.train.logger import StepLogger

_LOG = logging.getLogger(__name__)


class MetricsRecorder:
    """Writer API for per-store metrics.

    Held by `trainer.run` for the duration of one run; callers pass
    events via the `record_*` methods. Each record opens its own
    connection (WAL supports this cheaply) so long-lived state is
    minimal.
    """

    def __init__(self, store_root: Path, *, strict: bool = False) -> None:
        self._store_root = store_root
        self._strict = strict
        self._logged_failures: set[str] = set()

    def _with_conn(
        self,
        fn: Callable[[sqlite3.Connection], None],
        *,
        failure_key: str,
        hard_fail: bool,
    ) -> None:
        """Open a connection, run `fn(conn)`, enforce the recorder policy.

        Anchor writes (run start/end) always raise because downstream
        queries key off them. Other event streams are best-effort
        unless the recorder was created with `strict=True`.
        """
        try:
            with connect(self._store_root) as conn:
                fn(conn)
                conn.commit()
        except sqlite3.Error as exc:
            if hard_fail or self._strict:
                raise
            if failure_key in self._logged_failures:
                return
            self._logged_failures.add(failure_key)
            _LOG.error(
                "metrics %s write failed: %s (best-effort; continuing)",
                failure_key,
                exc,
            )

    def record_run_start(self, event: RunStart) -> None:
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(run_id, started_at, ended_at, adapter_version, phase, seed, status) "
                "VALUES (?, ?, NULL, ?, ?, ?, 'running')",
                (
                    event.run_id,
                    event.started_at,
                    event.adapter_version,
                    event.phase,
                    event.seed,
                ),
            )

        self._with_conn(_do, failure_key="run_start", hard_fail=True)

    def record_run_end(self, event: RunEnd) -> None:
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE runs SET ended_at=?, status=? WHERE run_id=?",
                (event.ended_at, event.status, event.run_id),
            )

        self._with_conn(_do, failure_key="run_end", hard_fail=True)

    def record_step(self, event: StepEvent) -> None:
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR REPLACE INTO steps "
                "(run_id, step, loss, lr, grad_norm, tokens_per_sec, peak_vram_mb, at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.run_id,
                    event.step,
                    event.loss,
                    event.lr,
                    event.grad_norm,
                    event.tokens_per_sec,
                    event.peak_vram_mb,
                    event.at,
                ),
            )

        self._with_conn(_do, failure_key="step", hard_fail=False)

    def record_eval(self, event: EvalEvent) -> None:
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR REPLACE INTO evals "
                "(run_id, step, val_loss, perplexity, retention, at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.run_id,
                    event.step,
                    event.val_loss,
                    event.perplexity,
                    event.retention,
                    event.at,
                ),
            )

        self._with_conn(_do, failure_key="eval", hard_fail=False)

    def record_tokenization(self, event: TokenizationEvent) -> None:
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR REPLACE INTO tokenization "
                "(run_id, total_sections, cache_hits, cache_misses, "
                "total_tokenize_seconds, cache_bytes_after, at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    event.run_id,
                    event.total_sections,
                    event.cache_hits,
                    event.cache_misses,
                    event.total_tokenize_seconds,
                    event.cache_bytes_after,
                    event.at,
                ),
            )

        self._with_conn(_do, failure_key="tokenization", hard_fail=False)

    def record_gate(self, event: GateEvent) -> None:
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR REPLACE INTO gate_events "
                "(run_id, adapter_name, mean_weight, sample_count, mode, at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.run_id,
                    event.adapter_name,
                    event.mean_weight,
                    event.sample_count,
                    event.mode,
                    event.at,
                ),
            )

        self._with_conn(_do, failure_key="gate", hard_fail=False)

    def record_preference_mine(self, event: PreferenceMineEvent) -> None:
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO preference_mining "
                "(run_id, judge_name, sample_count, mined_pairs, skipped_prompts, write_mode, at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    event.run_id,
                    event.judge_name,
                    event.sample_count,
                    event.mined_pairs,
                    event.skipped_prompts,
                    event.write_mode,
                    event.at,
                ),
            )

        self._with_conn(_do, failure_key="preference_mine", hard_fail=False)

    def record_export(self, event: ExportEvent) -> None:
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR REPLACE INTO exports "
                "(exported_at, quant, merged, ollama_name, size_bytes, duration_s) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.exported_at,
                    event.quant,
                    1 if event.merged else 0,
                    event.ollama_name,
                    event.size_bytes,
                    event.duration_s,
                ),
            )

        self._with_conn(_do, failure_key="export", hard_fail=False)


class DlmTrainerCallback:  # pragma: no cover - heavy trainer hook
    """TRL/HF `TrainerCallback`-shaped adapter for the metrics recorder.

    `on_log` fires for step-level loss records; `on_evaluate` fires
    with the eval metrics dict. The callback reads `state.global_step`
    and emits both the SQLite events (via `MetricsRecorder`) and the
    JSONL step record (via the optional `StepLogger`) — so nothing
    downstream needs to walk `trainer.state.log_history` after the
    training loop returns.

    The recorder owns the SQLite failure policy. In the default mode,
    per-step/per-eval writes degrade to one ERROR log per stream; in
    strict mode they raise out of the callback and abort training.

    Not a unit-test target — the HF TrainerCallback surface requires
    real trainer state. The `MetricsRecorder.record_*` methods are
    the tested surface.
    """

    def __init__(
        self,
        recorder: MetricsRecorder,
        run_id: int,
        step_logger: StepLogger | None = None,
    ) -> None:
        self._recorder = recorder
        self._run_id = run_id
        self._step_logger = step_logger

    def on_log(
        self,
        _args: Any,
        state: Any,
        _control: Any,
        logs: dict[str, float] | None = None,
        **_kwargs: Any,
    ) -> None:
        if logs is None:
            return
        if "loss" not in logs:
            return
        step = int(state.global_step)
        loss = float(logs["loss"])
        lr = _maybe_float(logs.get("learning_rate"))
        grad_norm = _maybe_float(logs.get("grad_norm"))
        self._recorder.record_step(
            StepEvent(
                run_id=self._run_id,
                step=step,
                loss=loss,
                lr=lr,
                grad_norm=grad_norm,
            )
        )
        if self._step_logger is not None:
            try:
                self._step_logger.log_step(
                    step=step,
                    loss=loss,
                    lr=lr if lr is not None else 0.0,
                    grad_norm=grad_norm,
                    val_loss=_maybe_float(logs.get("eval_loss")),
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                _LOG.warning("step logger log_step failed: %s (swallowed)", exc)

    def on_evaluate(
        self,
        _args: Any,
        state: Any,
        _control: Any,
        metrics: dict[str, float] | None = None,
        **_kwargs: Any,
    ) -> None:
        if metrics is None:
            return
        self._recorder.record_eval(
            EvalEvent(
                run_id=self._run_id,
                step=int(state.global_step),
                val_loss=_maybe_float(metrics.get("eval_loss")),
                perplexity=_maybe_float(metrics.get("eval_perplexity")),
            )
        )


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
