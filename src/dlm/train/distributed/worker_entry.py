"""Per-rank entry point for `accelerate launch -m`.

Accelerate spawns one process per GPU; each invokes this module. The
worker re-parses the subset of CLI args it cares about (path, seed,
max_steps, resume/fresh, phase) and routes into the existing
`dlm.train.trainer.run` — which Sprint 23 still owns the single-GPU
I/O shape for.

Full DDP integration (refactoring `trainer.run` to gate its I/O via
`rank_io.master_only`) is tracked as Sprint 23 follow-up; this entry
makes the launcher path complete end-to-end from the CLI but the
actual multi-GPU training loop remains a scaffold until the
integration test lands on real hardware.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - runtime path
    """Per-rank entry. Returns the exit code.

    The current implementation delegates to the Typer app directly
    with the forwarded argv, so rank subprocesses re-enter the same
    CLI (one process per GPU). Proper DDP participation —
    `Accelerator` wrap, master-only I/O inside `trainer.run`,
    gradient sync — will land alongside the multi-GPU smoke test and
    is intentionally separated from this scaffold.
    """
    from dlm.cli.app import app

    args = list(argv if argv is not None else sys.argv[1:])
    # The CLI args the worker receives should NOT include the original
    # `--gpus` flag — the launcher strips it before invoking us. If a
    # future refactor ever lets it through, strip defensively.
    cleaned = _strip_gpus_flag(args)
    app(cleaned, standalone_mode=False)
    return 0


def _strip_gpus_flag(args: list[str]) -> list[str]:
    """Drop `--gpus <value>` / `--gpus=<value>` from argv (worker side).

    Per-rank invocations must not recurse into the launcher branch of
    `dlm train`. Delegates to the shared `strip_gpus_flag` helper
    (audit-08 N1); the worker passes argv without argv[0].
    """
    from dlm.train.distributed.gpus import strip_gpus_flag

    return strip_gpus_flag(args, skip_argv0=False)


if __name__ == "__main__":  # pragma: no cover - entry
    sys.exit(main())
