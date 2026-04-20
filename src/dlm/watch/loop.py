"""Watch-mode retrain loop.

`run_watch(doc_path, *, max_steps, debounce_ms, ...)` blocks on
filesystem events, coalesces them through a `Debouncer`, and drives
one incremental `trainer.run(mode="resume", max_steps=<cap>)` cycle
per settled burst. Each cycle is bounded by `max_steps` so the loop
stays responsive to the next save.

The loop is structured so the cycle driver itself is easy to
unit-test — the real wiring happens in `_do_one_cycle`, which takes
callables as injection points and returns a typed result. The CLI
binds those to `parse_file` + `diff_against_manifest` + `trainer.run`.

Ctrl-C semantics:

- A SIGINT between cycles → `run_watch` returns cleanly.
- A SIGINT mid-cycle → the current cycle finishes (trainer owns the
  atomic commit; we don't abort mid-run), then the loop exits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from dlm.base_models import BaseModelSpec
    from dlm.doc.parser import ParsedDlm
    from dlm.hardware.plan import TrainingPlan
    from dlm.store.paths import StorePath
    from dlm.train.trainer import TrainingRunResult

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class CycleResult:
    """Outcome of one watch-cycle attempt."""

    ran: bool
    """True iff the trainer actually executed (change-set had new rows)."""

    new_sections: int
    """How many new sections triggered this cycle (0 means skipped)."""

    removed_sections: int
    """Sections that disappeared since the last manifest snapshot."""

    run_result: TrainingRunResult | None = None
    """`trainer.run` result when `ran` is True, else None."""


class _DocReloader(Protocol):
    """Re-parses the `.dlm` from disk each cycle."""

    def __call__(self, path: Path) -> ParsedDlm: ...


class _Retrainer(Protocol):
    """Runs one `trainer.run` cycle with `mode="resume"` + step cap."""

    def __call__(
        self,
        store: StorePath,
        parsed: ParsedDlm,
        spec: BaseModelSpec,
        plan: TrainingPlan,
        *,
        max_steps: int,
    ) -> TrainingRunResult: ...


def do_one_cycle(  # noqa: PLR0913 — cycle driver has many deps by design
    *,
    doc_path: Path,
    store: StorePath,
    spec: BaseModelSpec,
    plan: TrainingPlan,
    max_steps: int,
    reload_doc: _DocReloader,
    retrain: _Retrainer,
) -> CycleResult:
    """Execute one watch cycle.

    Returns a `CycleResult` describing whether the trainer ran.
    Raises whatever the trainer raises — the caller decides how to
    surface it on the status line.
    """
    from dlm.replay import diff_against_manifest
    from dlm.store.manifest import load_manifest

    parsed = reload_doc(doc_path)
    manifest = load_manifest(store.manifest)
    change_set = diff_against_manifest(list(parsed.sections), manifest)

    if not change_set.new:
        _LOG.info(
            "watch: no new sections (unchanged=%d, removed=%d); skipping cycle",
            len(change_set.unchanged),
            len(change_set.removed),
        )
        return CycleResult(
            ran=False,
            new_sections=0,
            removed_sections=len(change_set.removed),
        )

    _LOG.info(
        "watch: %d new section(s); retraining (mode=resume, max_steps=%d)",
        len(change_set.new),
        max_steps,
    )
    result = retrain(store, parsed, spec, plan, max_steps=max_steps)
    return CycleResult(
        ran=True,
        new_sections=len(change_set.new),
        removed_sections=len(change_set.removed),
        run_result=result,
    )


def _default_reload(path: Path) -> ParsedDlm:  # pragma: no cover - thin wrapper
    from dlm.doc.parser import parse_file

    return parse_file(path)


def _default_retrain(  # pragma: no cover - thin wrapper
    store: StorePath,
    parsed: ParsedDlm,
    spec: BaseModelSpec,
    plan: TrainingPlan,
    *,
    max_steps: int,
) -> TrainingRunResult:
    from dlm.train.trainer import run as trainer_run

    return trainer_run(
        store,
        parsed,
        spec,
        plan,
        mode="resume",
        max_steps=max_steps,
    )


def run_watch(  # pragma: no cover - interactive path
    *,
    doc_path: Path,
    store: StorePath,
    spec: BaseModelSpec,
    plan: TrainingPlan,
    max_steps: int = 100,
    debounce_ms: int = 400,
    on_cycle: Callable[[CycleResult], None] | None = None,
) -> int:
    """Block and drive watch-mode cycles until interrupted.

    Returns an exit code (0 on clean Ctrl-C). Non-zero on unhandled
    errors inside the cycle — the CLI surfaces them.

    `on_cycle` is an observer hook (status line renderer).
    """
    from dlm.watch.debounce import Debouncer
    from dlm.watch.watcher import watch_for_changes

    debouncer = Debouncer(quiet_seconds=debounce_ms / 1000.0)
    pending = False

    def _on_change() -> None:
        nonlocal pending
        debouncer.record()
        pending = True

    def _drain_pending() -> None:
        nonlocal pending
        if not pending:
            return
        # Let the FS quiet down before firing.
        import time

        while not debouncer.should_fire():
            time.sleep(0.05)
        coalesced = debouncer.pending_count
        debouncer.reset()
        pending = False

        result = do_one_cycle(
            doc_path=doc_path,
            store=store,
            spec=spec,
            plan=plan,
            max_steps=max_steps,
            reload_doc=_default_reload,
            retrain=_default_retrain,
        )
        if on_cycle is not None:
            on_cycle(result)
        _LOG.info(
            "watch: cycle done (ran=%s, new=%d, coalesced=%d)",
            result.ran,
            result.new_sections,
            coalesced,
        )

    try:
        watch_for_changes(doc_path, _on_change)
    except KeyboardInterrupt:
        _LOG.info("watch: Ctrl-C received; exiting")
        _drain_pending()
        return 0
    return 0
