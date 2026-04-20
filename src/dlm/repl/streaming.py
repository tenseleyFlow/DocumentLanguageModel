"""Token-level stdout streaming for the REPL.

Two modes:

- **Streaming (TTY).** Wraps transformers' `TextStreamer` so each
  decoded token lands on stdout as the model emits it. Ctrl-C
  interrupts generation cleanly; the tokens seen so far remain on
  screen and in the accumulated buffer.
- **Batched (non-TTY).** Redirecting `dlm repl > out.txt` flushes
  token-by-token poorly. Detect that case up front, skip streaming,
  and let the REPL loop print the final response once.

The streamer captures everything it prints into `.text` so the REPL
can persist the response (even a cancelled partial) into history.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


def should_stream() -> bool:
    """Stream iff stdout is attached to an interactive terminal.

    Pipe / redirect / `less` all flip this to False; the REPL falls
    back to printing the full response at end-of-turn instead.
    """
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):  # pragma: no cover - closed stdout
        return False


class CaptureStreamer:
    """Lightweight duck-typed stand-in when `transformers.TextStreamer`
    isn't available or we don't want streaming.

    Matches the API callers need (`put`, `end`, `text` attribute) so
    the REPL doesn't branch on type. The real HF streamer is loaded
    lazily by `build_streamer` on the live path.
    """

    def __init__(self) -> None:
        self.text: str = ""

    def put(self, _tokens: Any) -> None:  # pragma: no cover - unused on capture path
        pass

    def end(self) -> None:  # pragma: no cover
        pass


def build_streamer(tokenizer: Any, *, stream_to_stdout: bool) -> Any:  # pragma: no cover
    """Return a TextStreamer-shaped object the generate loop can use.

    `stream_to_stdout=True` returns HF's `TextStreamer` wrapping
    sys.stdout; `False` returns `CaptureStreamer` — generation runs
    without live output, REPL prints the final assistant message at
    the end.

    Pragma'd from unit coverage because it imports transformers.
    The `CaptureStreamer` fallback path is exercised by
    `test_capture_streamer.py`.
    """
    if not stream_to_stdout:
        return CaptureStreamer()

    from transformers import TextStreamer

    # `skip_prompt=True` — the prompt text was already on screen
    # (the user typed it); we only want the assistant's response to
    # stream. `skip_special_tokens=True` drops the chat-template
    # scaffolding so the output reads as prose.
    return TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


def concatenate_tokens(tokens: Iterable[str]) -> str:
    """Join token pieces into a contiguous response string.

    Trivial today; kept as a function so future tokenizer quirks
    (leading-space handling, byte-level decode) land in one place.
    """
    return "".join(tokens)
