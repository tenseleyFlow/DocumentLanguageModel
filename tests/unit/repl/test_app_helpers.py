"""Direct coverage for small non-interactive REPL helpers."""

from __future__ import annotations

from dlm.repl.app import _format_prompt


def test_format_prompt_handles_empty_and_existing_history() -> None:
    assert _format_prompt([]) == "> "
    assert _format_prompt([object(), object(), object(), object()]) == "[2] > "
