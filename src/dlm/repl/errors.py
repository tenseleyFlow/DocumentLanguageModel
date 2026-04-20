"""Typed errors for `dlm repl`.

Narrow set today; the REPL surface mostly wraps `dlm.inference` + CLI
errors and surfaces them via Rich rather than minting new classes.
The few errors that DO originate here correspond to slash-command
user errors (bad key, unknown command) and session-state problems
(adapter switch on a single-adapter doc).
"""

from __future__ import annotations


class ReplError(Exception):
    """Base class for REPL-specific errors."""


class UnknownCommandError(ReplError):
    """Raised when a line starting with `/` doesn't match any known command."""


class BadCommandArgumentError(ReplError):
    """Raised when a known command is invoked with invalid arguments.

    Examples:

    - `/params temperature=foo` — `foo` isn't a float.
    - `/adapter ghost` — name not declared in `training.adapters`.
    - `/save` — missing path argument.
    """
