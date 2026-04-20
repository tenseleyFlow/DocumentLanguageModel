"""`make_formatting_func` row-shape routing."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dlm.data.errors import DataFormatError
from dlm.data.formatter import make_formatting_func


class TestMessagesShape:
    def test_messages_calls_apply_chat_template(self) -> None:
        tok = MagicMock()
        tok.apply_chat_template.return_value = "<|user|>q<|assistant|>a"
        fn = make_formatting_func(tok)

        row = {
            "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        }
        result = fn(row)
        assert result == "<|user|>q<|assistant|>a"

        tok.apply_chat_template.assert_called_once_with(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    def test_messages_with_non_str_result_raises(self) -> None:
        tok = MagicMock()
        tok.apply_chat_template.return_value = [1, 2, 3]
        fn = make_formatting_func(tok)
        with pytest.raises(DataFormatError, match="non-str"):
            fn({"messages": [{"role": "user", "content": "x"}]})


class TestTextShape:
    def test_text_passthrough(self) -> None:
        fn = make_formatting_func(MagicMock())
        assert fn({"text": "raw prose here"}) == "raw prose here"

    def test_non_str_text_rejected(self) -> None:
        fn = make_formatting_func(MagicMock())
        with pytest.raises(DataFormatError, match="must be str"):
            fn({"text": 123})  # type: ignore[dict-item]


class TestRejections:
    def test_preference_row_rejected_with_routing_hint(self) -> None:
        fn = make_formatting_func(MagicMock())
        with pytest.raises(DataFormatError, match="DPOTrainer"):
            fn({"prompt": "p", "chosen": "c", "rejected": "r"})

    def test_unknown_shape_rejected(self) -> None:
        fn = make_formatting_func(MagicMock())
        with pytest.raises(DataFormatError, match="neither"):
            fn({"something_else": "x"})


class TestSchemaUnificationNones:
    """When `datasets.Dataset` combines mixed-shape rows, each row
    gains all columns, with the ones that don't apply set to `None`.
    The dispatcher must route on the non-None column, not merely on
    key presence."""

    def test_prose_row_with_messages_none_takes_text_path(self) -> None:
        tok = MagicMock()
        fn = make_formatting_func(tok)
        out = fn({"text": "hello", "messages": None})
        assert out == "hello"
        tok.apply_chat_template.assert_not_called()

    def test_instruction_row_with_text_none_takes_messages_path(self) -> None:
        tok = MagicMock()
        tok.apply_chat_template.return_value = "<rendered>"
        fn = make_formatting_func(tok)
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        out = fn({"text": None, "messages": msgs})
        assert out == "<rendered>"
        tok.apply_chat_template.assert_called_once()

    def test_all_nones_rejected(self) -> None:
        fn = make_formatting_func(MagicMock())
        with pytest.raises(DataFormatError, match="neither"):
            fn({"text": None, "messages": None})
