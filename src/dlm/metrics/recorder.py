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

Best-effort throughout: a failed write is logged and swallowed so a
metrics hiccup never kills training.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlm.metrics.db import connect
from dlm.metrics.events import (
    EvalEvent,
    ExportEvent,
    GateEvent,
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

    def __init__(self, store_root: Path) -> None:
        self._store_root = store_root

    def _with_conn(self, fn: Any) -> None:
        """Open a connection, run `fn(conn)`, best-effort swallow errors.

        SQLite errors here should never take down a training run.
        Metrics are a nice-to-have; correctness of the actual
        checkpoint + manifest + lock is the contract that matters.
        """
        try:
            with connect(self._store_root) as conn:
                fn(conn)
                conn.commit()
        except sqlite3.Error as exc:
            _LOG.warning("metrics write failed: %s (swallowed)", exc)

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

        self._with_conn(_do)

    def record_run_end(self, event: RunEnd) -> None:
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE runs SET ended_at=?, status=? WHERE run_id=?",
                (event.ended_at, event.status, event.run_id),
            )

        self._with_conn(_do)

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

        self._with_conn(_do)

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

        self._with_conn(_do)

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

        self._with_conn(_do)

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

        self._with_conn(_do)

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

        self._with_conn(_do)


class DlmTrainerCallback:  # pragma: no cover - heavy trainer hook
    """TRL/HF `TrainerCallback`-shaped adapter for the metrics recorder.

    `on_log` fires for step-level loss records; `on_evaluate` fires
    with the eval metrics dict. The callback reads `state.global_step`
    and emits both the SQLite events (via `MetricsRecorder`) and the
    JSONL step record (via the optional `StepLogger`) — so nothing
    downstream needs to walk `trainer.state.log_history` after the
    training loop returns.

    A best-effort `try/except` wraps each record call: a metrics write
    failing in the middle of training must never abort the run.

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
        try:
            self._recorder.record_step(
                StepEvent(
                    run_id=self._run_id,
                    step=step,
                    loss=loss,
                    lr=lr,
                    grad_norm=grad_norm,
                )
            )
        except Exception as exc:  # noqa: BLE001 - metrics must never kill training
            _LOG.warning("metrics record_step failed: %s (swallowed)", exc)
        if self._step_logger is not None:
            try:
                self._step_logger.log_step(
                    step=step,
                    loss=loss,
                    lr=lr if lr is not None else 0.0,
                    grad_norm=grad_norm,
                    val_loss=_maybe_float(logs.get("eval_loss")),
                )
            except Exception as exc:  # noqa: BLE001 - JSONL must never kill training
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
        try:
            self._recorder.record_eval(
                EvalEvent(
                    run_id=self._run_id,
                    step=int(state.global_step),
                    val_loss=_maybe_float(metrics.get("eval_loss")),
                    perplexity=_maybe_float(metrics.get("eval_perplexity")),
                )
            )
        except Exception as exc:  # noqa: BLE001 - metrics must never kill training
            _LOG.warning("metrics record_eval failed: %s (swallowed)", exc)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
