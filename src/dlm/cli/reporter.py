"""Consistent CLI error reporter (Sprint 13).

Three tiers of error presentation so users see the right level of
detail without wading through Python tracebacks:

1. **Parse errors** (`DlmParseError` and subclasses) — already carry
   `path:line:col` via `_format()`. The reporter prints them verbatim
   with a red prefix.
2. **Typed domain errors** (`PreflightError`, `GatedModelError`,
   `TrainingError`, etc.) — shown as a single red-prefixed line plus a
   remediation hint when the error class carries one.
3. **Uncaught exceptions** — a single-line class name + message; the
   full traceback is gated behind `--verbose` (or the `DLM_VERBOSE=1`
   env var for crash-dump scenarios).

Color/format output respects `NO_COLOR=1` and TTY detection via Rich's
Console defaults. The `install_excepthook` wrapper routes anything
that escapes a subcommand through this reporter instead of the
default Python traceback.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console


def _is_verbose() -> bool:
    """Return True when the user asked for full tracebacks.

    Two signals: (a) logger level set to DEBUG by the root callback,
    (b) `DLM_VERBOSE=1` env var for detached crash-dump runs where the
    callback didn't fire.
    """
    if os.environ.get("DLM_VERBOSE") == "1":
        return True
    return logging.getLogger().isEnabledFor(logging.DEBUG)


def _stderr_console() -> Console:
    from rich.console import Console

    return Console(stderr=True)


def report_exception(exc: BaseException) -> int:
    """Print `exc` per the tier rules and return the exit code.

    Callers can use this to funnel all uncaught errors into one code
    path, keeping CLI exit-code semantics consistent (2 for typed
    domain errors → CLI usage problems, 1 for unexpected failures).
    """
    console = _stderr_console()

    # Tier 1: parse errors. Their str() already has file:line:col.
    try:
        from dlm.doc.errors import DlmParseError

        if isinstance(exc, DlmParseError):
            console.print(f"[red]parse:[/red] {exc}")
            return 1
    except ImportError:  # pragma: no cover — dlm.doc always importable
        pass

    # Tier 2: typed domain errors. Pick a short prefix per module.
    prefix = _prefix_for(exc)
    if prefix is not None:
        console.print(f"[red]{prefix}:[/red] {exc}")
        if _is_verbose():
            _print_traceback(console, exc)
        return 1

    # Tier 3: unexpected. Compact one-liner; full traceback only when
    # --verbose or DLM_VERBOSE=1.
    console.print(
        f"[red]error:[/red] {type(exc).__name__}: {exc}\n"
        "  re-run with [bold]--verbose[/bold] for the full traceback."
    )
    if _is_verbose():
        _print_traceback(console, exc)
    return 1


def _prefix_for(exc: BaseException) -> str | None:
    """Map a typed domain error to a short display prefix, or None.

    `None` means "let tier 3 handle it" — the class is outside our
    known hierarchy. Keeps this module from importing every sibling
    package eagerly; each check is a narrow import.
    """
    mod = type(exc).__module__
    name = type(exc).__name__

    if mod.startswith("dlm.base_models"):
        if name == "GatedModelError":
            return "license"
        return "base_model"
    if mod.startswith("dlm.doc"):
        return "doc"
    if mod.startswith("dlm.store"):
        return "store"
    if mod.startswith("dlm.train"):
        return "train"
    if mod.startswith("dlm.export"):
        return "export"
    if mod.startswith("dlm.inference"):
        return "inference"
    if mod.startswith("dlm.hardware"):
        return "doctor"
    return None


def _print_traceback(console: Console, exc: BaseException) -> None:
    from rich.traceback import Traceback

    tb = Traceback.from_exception(
        type(exc),
        exc,
        exc.__traceback__,
        show_locals=False,
    )
    console.print(tb)


def run_with_reporter(app: Callable[[], None]) -> int:
    """Invoke a Typer `app` and route any escaping exception through the reporter.

    Returns the exit code so the installed entry point can propagate it
    to the shell. `typer.Exit` and `SystemExit` are re-raised so their
    user-intended exit codes reach the interpreter unchanged.
    """
    try:
        app()
        return 0
    except SystemExit as exc:  # typer.Exit inherits from SystemExit
        code = exc.code
        if isinstance(code, int):
            return code
        if code is None:
            return 0
        # String-valued SystemExit is legacy — print and fail.
        sys.stderr.write(f"{code}\n")
        return 1
    except KeyboardInterrupt:
        _stderr_console().print("[yellow]interrupted[/yellow]")
        return 130
    except BaseException as exc:
        return report_exception(exc)
