"""Slash-command parser + handlers for `dlm repl`.

Pure logic: `parse_command` turns a `/foo bar baz` line into a typed
dispatch, and each handler mutates the session state or returns an
enum describing what the REPL loop should do next (exit, print info,
continue). No stdout or prompt_toolkit touches here — the REPL loop
owns I/O.

The intent splits the work so the parser + handlers are trivially
unit-testable without spinning up an inference backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from dlm.repl.errors import BadCommandArgumentError, UnknownCommandError

if TYPE_CHECKING:
    from dlm.repl.session import ReplSession


class Action(Enum):
    """What the REPL loop should do after a command runs.

    Returned from each handler so the loop doesn't grow a big if/elif
    branch around command names.
    """

    CONTINUE = "continue"
    """Loop resumes; the handler already did its work."""

    EXIT = "exit"
    """Loop breaks cleanly."""


@dataclass(frozen=True)
class CommandResult:
    """Handler return type: action + optional message to print."""

    action: Action
    message: str | None = None


# Paired registry of (name, handler). Using a tuple of tuples rather
# than a dict to preserve declaration order for `/help`.
_COMMANDS: tuple[str, ...] = (
    "exit",
    "quit",  # alias for /exit
    "clear",
    "save",
    "adapter",
    "params",
    "model",
    "history",
    "help",
)


def is_command(line: str) -> bool:
    """True iff `line` begins with `/` after trimming leading whitespace."""
    return line.lstrip().startswith("/")


def parse_and_dispatch(line: str, session: ReplSession) -> CommandResult:
    """Parse a `/command args` line and run its handler on `session`.

    Raises `UnknownCommandError` on an unrecognized name.
    `BadCommandArgumentError` propagates from handlers.
    """
    stripped = line.strip()
    if not stripped.startswith("/"):
        raise UnknownCommandError(f"not a command (does not start with `/`): {line!r}")
    body = stripped[1:]
    if not body:
        raise UnknownCommandError("empty command (`/` with no name)")

    name, _, args = body.partition(" ")
    args = args.strip()
    name = name.lower()

    if name not in _COMMANDS:
        raise UnknownCommandError(f"unknown command `/{name}` — try /help")

    handler = _HANDLERS[name]
    return handler(args, session)


def _cmd_exit(_args: str, _session: ReplSession) -> CommandResult:
    return CommandResult(action=Action.EXIT, message="bye")


def _cmd_clear(_args: str, session: ReplSession) -> CommandResult:
    session.clear_history()
    return CommandResult(action=Action.CONTINUE, message="history cleared")


def _cmd_save(args: str, session: ReplSession) -> CommandResult:
    if not args:
        raise BadCommandArgumentError("/save requires a path argument")
    path = Path(args).expanduser()
    session.save_history(path)
    return CommandResult(
        action=Action.CONTINUE,
        message=f"saved {len(session.history)} messages to {path}",
    )


def _cmd_adapter(args: str, session: ReplSession) -> CommandResult:
    if not args:
        raise BadCommandArgumentError("/adapter requires an adapter name")
    if not session.declared_adapters:
        raise BadCommandArgumentError(
            "/adapter is only valid on multi-adapter documents "
            "(this doc does not declare `training.adapters`)"
        )
    name = args.strip()
    if name not in session.declared_adapters:
        raise BadCommandArgumentError(
            f"/adapter {name!r} is not declared (declared: {sorted(session.declared_adapters)!r})"
        )
    session.active_adapter = name
    return CommandResult(
        action=Action.CONTINUE,
        message=f"active adapter → {name} (restart REPL for the swap to take full effect)",
    )


_FLOAT_PARAMS = {"temperature", "top_p", "repetition_penalty"}
_INT_PARAMS = {"top_k", "max_new_tokens"}
_ALL_PARAMS = _FLOAT_PARAMS | _INT_PARAMS


def _cmd_params(args: str, session: ReplSession) -> CommandResult:
    """`/params key=value [key=value ...]` — update generation knobs.

    Accepts multiple assignments in one call. Invalid keys or values
    raise `BadCommandArgumentError` without partial updates (the
    parse happens before any assignment).
    """
    if not args:
        return CommandResult(
            action=Action.CONTINUE,
            message=_render_params(session),
        )

    updates: dict[str, float | int] = {}
    for token in args.split():
        if "=" not in token:
            raise BadCommandArgumentError(f"/params entry {token!r} must be `key=value`")
        key, _, value_str = token.partition("=")
        key = key.strip()
        value_str = value_str.strip()
        if key not in _ALL_PARAMS:
            raise BadCommandArgumentError(
                f"/params: unknown key {key!r}; valid: {sorted(_ALL_PARAMS)!r}"
            )
        if key in _FLOAT_PARAMS:
            try:
                updates[key] = float(value_str)
            except ValueError as exc:
                raise BadCommandArgumentError(f"/params {key}={value_str!r}: not a number") from exc
        else:
            try:
                updates[key] = int(value_str)
            except ValueError as exc:
                raise BadCommandArgumentError(
                    f"/params {key}={value_str!r}: not an integer"
                ) from exc

    for key, value in updates.items():
        setattr(session.gen_params, key, value)

    return CommandResult(
        action=Action.CONTINUE,
        message=_render_params(session),
    )


def _cmd_model(_args: str, session: ReplSession) -> CommandResult:
    backend_name = getattr(session.backend, "name", "unknown")
    adapter = session.active_adapter or "(flat / single)"
    return CommandResult(
        action=Action.CONTINUE,
        message=f"backend={backend_name} adapter={adapter}",
    )


def _cmd_history(_args: str, session: ReplSession) -> CommandResult:
    if not session.history:
        return CommandResult(action=Action.CONTINUE, message="history is empty")
    lines = [
        f"[{i}] {msg['role']}: {_truncate(msg['content'], 100)}"
        for i, msg in enumerate(session.history)
    ]
    return CommandResult(action=Action.CONTINUE, message="\n".join(lines))


def _cmd_help(_args: str, _session: ReplSession) -> CommandResult:
    lines = [
        "Commands:",
        "  /exit | /quit         end the session",
        "  /clear                reset conversation history",
        "  /save <path>          write history as JSON",
        "  /adapter <name>       switch named adapter (multi-adapter docs)",
        "  /params key=value     update generation knobs (e.g. temperature=0.5)",
        "  /params               (no args) print current generation params",
        "  /model                print backend + active adapter",
        "  /history              list the current conversation",
        "  /help                 this message",
        "  Ctrl-D also exits.",
    ]
    return CommandResult(action=Action.CONTINUE, message="\n".join(lines))


_HANDLERS = {
    "exit": _cmd_exit,
    "quit": _cmd_exit,
    "clear": _cmd_clear,
    "save": _cmd_save,
    "adapter": _cmd_adapter,
    "params": _cmd_params,
    "model": _cmd_model,
    "history": _cmd_history,
    "help": _cmd_help,
}


def _render_params(session: ReplSession) -> str:
    p = session.gen_params
    bits = [
        f"temperature={p.temperature}",
        f"top_p={p.top_p}",
        f"top_k={p.top_k}",
        f"max_new_tokens={p.max_new_tokens}",
        f"repetition_penalty={p.repetition_penalty}",
    ]
    return " ".join(bits)


def _truncate(text: str, limit: int) -> str:
    """Clip a history line for `/history` display."""
    flat = text.replace("\n", " ")
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"
