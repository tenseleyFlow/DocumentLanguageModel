"""TRL-compatible `formatting_func` builder.

Branches per row shape:

- `"messages"` present → `tokenizer.apply_chat_template(msgs, tokenize=False)`.
  SFTTrainer's completion-only loss masking kicks in automatically when
  the formatted string is a chat transcript.
- `"text"` present → passthrough. SFTTrainer treats it as CPT (loss on
  all tokens).
- neither → `DataFormatError`.

PREFERENCE rows (`prompt`/`chosen`/`rejected`) are NOT formatted here —
they're routed to DPOTrainer by Sprint 17, which has its own formatter.
This function refuses them explicitly so an accidentally-mixed dataset
fails loudly at format time rather than producing silently-wrong data.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dlm.data.errors import DataFormatError

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

Row = dict[str, Any]
FormattingFunc = Callable[[Row], str]


def make_formatting_func(tokenizer: PreTrainedTokenizerBase) -> FormattingFunc:
    """Return a row→str function bound to `tokenizer`'s chat template."""

    def formatting_func(row: Row) -> str:
        # HF `datasets.Dataset` unifies the schema across mixed-shape
        # rows (e.g. PROSE + INSTRUCTION in one doc), filling missing
        # columns with `None`. Dispatch on presence-and-non-None so a
        # prose row with an injected `messages: None` doesn't route
        # into `apply_chat_template` and crash Jinja with a None
        # iterable.
        messages = row.get("messages")
        if messages is not None:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if not isinstance(rendered, str):
                raise DataFormatError(
                    f"apply_chat_template returned non-str ({type(rendered).__name__}); "
                    "ensure tokenize=False path is taken"
                )
            return rendered
        text = row.get("text")
        if text is not None:
            if not isinstance(text, str):
                raise DataFormatError(f"`text` field must be str, got {type(text).__name__}")
            return text
        if row.get("prompt") is not None and row.get("chosen") is not None and row.get("rejected") is not None:
            raise DataFormatError(
                "preference rows (prompt/chosen/rejected) must be routed to DPOTrainer, "
                "not SFTTrainer's formatting_func"
            )
        raise DataFormatError(f"row has neither `messages` nor `text`: keys={sorted(row.keys())}")

    return formatting_func
