"""`prepare_tokenizer` pad/EOS/template invariants."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dlm.data.errors import TokenizerBringupError
from dlm.data.tokenizer_bringup import prepare_tokenizer


def _mock_tokenizer(
    *,
    pad_token: str | None,
    eos_token: str | None,
    unk_token: str | None,
    chat_template: str | None,
) -> MagicMock:
    tok = MagicMock()
    tok.pad_token = pad_token
    tok.eos_token = eos_token
    tok.unk_token = unk_token
    tok.chat_template = chat_template

    def _add_special(mapping: dict[str, str]) -> None:
        if "pad_token" in mapping:
            tok.pad_token = mapping["pad_token"]

    tok.add_special_tokens.side_effect = _add_special
    return tok


class TestPadTokenRule:
    def test_existing_distinct_pad_preserved(self) -> None:
        tok = _mock_tokenizer(
            pad_token="<|pad|>",
            eos_token="<|eot|>",
            unk_token=None,
            chat_template="{{messages}}",
        )
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok):
            result = prepare_tokenizer("org/x", "a" * 40)
        assert result.tokenizer_grew is False
        assert result.pad_token == "<|pad|>"

    def test_pad_equals_eos_falls_back_to_unk(self) -> None:
        tok = _mock_tokenizer(
            pad_token="<|eos|>",
            eos_token="<|eos|>",
            unk_token="<|unk|>",
            chat_template="{{messages}}",
        )
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok):
            result = prepare_tokenizer("org/x", "a" * 40)
        assert result.tokenizer_grew is False
        assert result.pad_token == "<|unk|>"

    def test_no_pad_no_unk_adds_new_token(self) -> None:
        tok = _mock_tokenizer(
            pad_token=None,
            eos_token="<|eos|>",
            unk_token=None,
            chat_template="{{messages}}",
        )
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok):
            result = prepare_tokenizer("org/x", "a" * 40)
        assert result.tokenizer_grew is True
        assert result.pad_token == "<|pad|>"

    def test_unk_equals_eos_forces_pad_literal(self) -> None:
        tok = _mock_tokenizer(
            pad_token=None,
            eos_token="<|eos|>",
            unk_token="<|eos|>",
            chat_template="{{messages}}",
        )
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok):
            result = prepare_tokenizer("org/x", "a" * 40)
        assert result.tokenizer_grew is True


class TestChatTemplate:
    def test_missing_template_raises(self) -> None:
        tok = _mock_tokenizer(
            pad_token="<|pad|>",
            eos_token="<|eot|>",
            unk_token=None,
            chat_template=None,
        )
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tok),
            pytest.raises(TokenizerBringupError, match="chat_template"),
        ):
            prepare_tokenizer("org/no-template", "a" * 40)

    def test_empty_template_raises(self) -> None:
        tok = _mock_tokenizer(
            pad_token="<|pad|>",
            eos_token="<|eot|>",
            unk_token=None,
            chat_template="   ",
        )
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tok),
            pytest.raises(TokenizerBringupError),
        ):
            prepare_tokenizer("org/empty", "a" * 40)
