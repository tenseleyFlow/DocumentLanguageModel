"""TensorBoard sink (Sprint 26).

Writes per-step scalars under `<store>/tensorboard/run_<id>/`. Users
point TensorBoard at the `tensorboard/` dir to see loss curves etc.
Optional dep — the `observability` extra pulls `tensorboard`. When
the package is absent, `TensorBoardSink` refuses construction so the
caller can surface a clean error.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dlm.metrics.events import EvalEvent, StepEvent


def tensorboard_available() -> bool:
    """True iff the `tensorboard` package is importable.

    Probe-only — we don't import tensorboard here because the user
    may have installed it without SummaryWriter working (bad build).
    Runtime `TensorBoardSink.__init__` does the actual import and
    raises cleanly on failure.
    """
    return importlib.util.find_spec("tensorboard") is not None


class TensorBoardSink:
    """SummaryWriter-backed metrics sink.

    Construction imports `torch.utils.tensorboard.SummaryWriter`
    lazily. If tensorboard isn't installed, construction raises —
    the caller should gate on `tensorboard_available()` first and
    emit a helpful message pointing at `uv sync --extra observability`.
    """

    def __init__(self, store_root: Path, run_id: int) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "TensorBoard sink requires `tensorboard`; "
                "run `uv sync --extra observability` to install it."
            ) from exc

        self._log_dir = store_root / "tensorboard" / f"run_{run_id:04d}"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._writer: Any = SummaryWriter(log_dir=str(self._log_dir))

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def record_step(self, event: StepEvent) -> None:
        if event.loss is not None:
            self._writer.add_scalar("train/loss", event.loss, event.step)
        if event.lr is not None:
            self._writer.add_scalar("train/lr", event.lr, event.step)
        if event.grad_norm is not None:
            self._writer.add_scalar("train/grad_norm", event.grad_norm, event.step)

    def record_eval(self, event: EvalEvent) -> None:
        if event.val_loss is not None:
            self._writer.add_scalar("eval/val_loss", event.val_loss, event.step)
        if event.perplexity is not None:
            self._writer.add_scalar("eval/perplexity", event.perplexity, event.step)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()
