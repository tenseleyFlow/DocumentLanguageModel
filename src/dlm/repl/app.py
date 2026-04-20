"""REPL main loop — `run_repl(session, history_path)`.

Imports `prompt_toolkit` lazily so collecting this module is cheap
(CLI boot shouldn't pay the ~15ms prompt_toolkit import cost just to
resolve `dlm repl`). Ctrl-C behavior:

- **During input**: prompt_toolkit raises `KeyboardInterrupt`. The
  loop catches it and just redraws the prompt — not an exit.
- **During generation**: we wrap `backend.generate` in a signal
  handler. Ctrl-C sets a cancel flag; the HF generate call responds
  to the streamer's stop signal and returns the partial text. We
  append to history with a `[cancelled]` marker so the next turn's
  context is clean.

Ctrl-D at the prompt (EOFError) is a clean exit.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from dlm.repl.commands import Action, is_command, parse_and_dispatch
from dlm.repl.errors import BadCommandArgumentError, UnknownCommandError
from dlm.repl.streaming import build_streamer, should_stream

if TYPE_CHECKING:
    from dlm.repl.session import ReplSession


DEFAULT_HISTORY_PATH = Path.home() / ".dlm" / "history"
"""Persistent readline history file — survives across sessions."""


def _format_prompt(history: list[object]) -> str:
    """Render the prompt. Shows turn count as a lightweight state cue."""
    turns = len(history) // 2
    return f"[{turns}] > " if turns else "> "


def run_repl(  # pragma: no cover - interactive path; covered by integration
    session: ReplSession,
    *,
    history_path: Path | None = None,
    console: object | None = None,
) -> int:
    """Enter the REPL. Returns an exit code (0 on clean quit).

    `history_path` defaults to `~/.dlm/history`. `console` is an
    optional Rich `Console` for styled output; if None, we fall back
    to plain `print`.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from rich.console import Console

    console = console or Console()
    assert isinstance(console, Console)

    history_path = history_path or DEFAULT_HISTORY_PATH
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pt_session: PromptSession[str] = PromptSession(history=FileHistory(str(history_path)))

    console.print(
        f"[dim]dlm repl — /help for commands, /exit to quit (history: {history_path})[/dim]"
    )

    while True:
        try:
            line = pt_session.prompt(_format_prompt(list(session.history)))
        except KeyboardInterrupt:
            # Ctrl-C at the prompt cancels the current input — redraw.
            console.print("[dim](input cancelled; Ctrl-D to exit)[/dim]")
            continue
        except EOFError:
            # Ctrl-D cleanly exits.
            console.print("[dim]bye[/dim]")
            return 0

        if not line.strip():
            continue

        if is_command(line):
            try:
                result = parse_and_dispatch(line, session)
            except (UnknownCommandError, BadCommandArgumentError) as exc:
                console.print(f"[red]error:[/red] {exc}")
                continue
            if result.message:
                console.print(result.message)
            if result.action is Action.EXIT:
                return 0
            continue

        # Plain user message — add to history, generate response.
        session.append_user(line)

        stream_enabled = should_stream()
        streamer = build_streamer(session.tokenizer, stream_to_stdout=stream_enabled)

        try:
            response = session.backend.generate(
                line,
                streamer=streamer,
                **session.gen_params.to_generate_kwargs(),
            )
            cancelled = False
        except KeyboardInterrupt:
            # Ctrl-C during generation: pull whatever the streamer
            # captured (or re-decode) and mark the turn cancelled.
            response = getattr(streamer, "text", "") or "<cancelled>"
            cancelled = True
            console.print("[yellow][cancelled][/yellow]")

        if not stream_enabled:
            # Non-TTY: the streamer was a capture shim, so the
            # response didn't print incrementally. Emit it now.
            sys.stdout.write(response + "\n")

        session.append_assistant(response, cancelled=cancelled)
