"""W&B (Weights & Biases) sink.

Strictly opt-in: `dlm train --wandb <project>` is the only path that
initializes W&B. We never import or initialize at module load, and
we default to **offline mode** so no run leaves the user's machine
without their explicit action.

Uploads happen later via `wandb sync <run-dir>` — that's a manual
step the user takes after reviewing what was logged locally.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dlm.metrics.events import EvalEvent, StepEvent


def wandb_available() -> bool:
    """True iff the `wandb` package is importable.

    Probe-only; no module import. See `WandbSink.__init__` for the
    actual `import wandb` side-effect.
    """
    return importlib.util.find_spec("wandb") is not None


class WandbSink:
    """wandb.Run-backed metrics sink, offline-by-default.

    On construction: `wandb.init(project=..., mode="offline", ...)`.
    The run directory lives under `<store>/wandb/`; users run
    `wandb sync <store>/wandb/offline-run-*` manually to upload.

    Caller is expected to gate on `wandb_available()` and pass the
    explicit `--wandb <project>` flag; we don't provide a "maybe wandb"
    shortcut that could silently initialize.
    """

    def __init__(self, store_root: Path, run_id: int, project: str) -> None:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "W&B sink requires `wandb`; run `uv sync --extra observability` to install it."
            ) from exc

        wandb_dir = store_root / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)

        self._run: Any = wandb.init(
            project=project,
            name=f"run_{run_id:04d}",
            dir=str(wandb_dir),
            mode="offline",
            reinit=True,
        )

    def record_step(self, event: StepEvent) -> None:
        payload: dict[str, float] = {}
        if event.loss is not None:
            payload["train/loss"] = event.loss
        if event.lr is not None:
            payload["train/lr"] = event.lr
        if event.grad_norm is not None:
            payload["train/grad_norm"] = event.grad_norm
        if payload:
            self._run.log(payload, step=event.step)

    def record_eval(self, event: EvalEvent) -> None:
        payload: dict[str, float] = {}
        if event.val_loss is not None:
            payload["eval/val_loss"] = event.val_loss
        if event.perplexity is not None:
            payload["eval/perplexity"] = event.perplexity
        if payload:
            self._run.log(payload, step=event.step)

    def close(self) -> None:
        self._run.finish()
