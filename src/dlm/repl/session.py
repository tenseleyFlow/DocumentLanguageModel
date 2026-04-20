"""Interactive session state for `dlm repl`.

`ReplSession` holds the live backend, conversation history, active
adapter, and generation parameters. Pure state container — the REPL
loop mutates it; streaming + I/O live in other modules.

History shape matches the OpenAI-style chat-messages convention so
it applies cleanly through `tokenizer.apply_chat_template`:

    [{"role": "user", "content": "..."},
     {"role": "assistant", "content": "..."}]

Each turn appends one user message, then one assistant message. A
`[cancelled]` suffix marks an assistant turn where Ctrl-C interrupted
generation (the partial response still lands in history so context
carries forward).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from dlm.inference.backends.base import InferenceBackend


Role = Literal["user", "assistant", "system"]


class ChatMessage(TypedDict):
    """A single turn in the conversation history.

    The `role` literal matches HF `apply_chat_template`'s expectations
    so we don't need to remap when re-rendering the prompt each turn.
    """

    role: Role
    content: str


@dataclass
class GenerationParams:
    """Mutable knobs the REPL lets users tweak via `/params`.

    Defaults mirror `dlm prompt`'s CLI defaults. `/params key=value`
    updates one field at a time; unknown keys raise
    `BadCommandArgumentError` at parse time.
    """

    temperature: float = 0.7
    top_p: float | None = None
    top_k: int | None = None
    max_new_tokens: int = 256
    repetition_penalty: float | None = None

    def to_generate_kwargs(self) -> dict[str, Any]:
        """Build the kwargs dict the inference backend's `generate` accepts.

        Mirrors the shape `dlm.inference.generate.build_generate_kwargs`
        reads, so the REPL path produces the same output as the
        single-shot CLI path for identical params.
        """
        out: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            out["top_p"] = self.top_p
        if self.top_k is not None:
            out["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            out["repetition_penalty"] = self.repetition_penalty
        return out


@dataclass
class ReplSession:
    """Live state of one REPL invocation."""

    backend: InferenceBackend
    tokenizer: Any  # HF PreTrainedTokenizer; Any avoids the heavy type import
    history: list[ChatMessage] = field(default_factory=list)
    active_adapter: str | None = None
    gen_params: GenerationParams = field(default_factory=GenerationParams)
    # Declared adapter names (from `training.adapters`) for validating
    # `/adapter` commands. Empty tuple for single-adapter docs; the
    # REPL refuses `/adapter` on those with a clear message.
    declared_adapters: tuple[str, ...] = ()

    def append_user(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})

    def append_assistant(self, content: str, *, cancelled: bool = False) -> None:
        """Append the model's response. Cancelled runs get a marker.

        The cancel marker keeps the partial response in context so the
        next turn sees what the model was in the middle of saying,
        while also surfacing that generation didn't complete.
        """
        if cancelled:
            content = content + " [cancelled]"
        self.history.append({"role": "assistant", "content": content})

    def clear_history(self) -> None:
        self.history.clear()

    def save_history(self, path: Path) -> None:
        """Write the history as JSON for replay / review.

        Schema is just `list[ChatMessage]` so a trivial
        `json.loads(path.read_text())` reconstructs it.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(list(self.history), indent=2) + "\n", encoding="utf-8")
