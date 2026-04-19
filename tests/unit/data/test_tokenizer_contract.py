"""`tokenizer_grew` + `modules_to_save_for_growth` — canonical Sprint 12b contract."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from dlm.data.tokenizer_contract import modules_to_save_for_growth, tokenizer_grew


@dataclass
class _FakeTokenizer:
    """Minimal stand-in: matches the two methods the predicate touches.

    Real tokenizers (BPE / SentencePiece / Unigram) expose the same
    two-method surface — `vocab_size` property + `get_added_vocab()`
    returning a `dict[str, int]`. Faking it is safe.
    """

    vocab_size: int
    added: dict[str, int] = field(default_factory=dict)

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self.added)


class TestTokenizerGrew:
    def test_identical_tokenizers_not_grown(self) -> None:
        a = _FakeTokenizer(vocab_size=32000, added={"<|im_end|>": 32001})
        b = _FakeTokenizer(vocab_size=32000, added={"<|im_end|>": 32001})
        assert tokenizer_grew(a, b) is False

    def test_vocab_size_change_detected(self) -> None:
        """Sprint 07 pad-fallback path: `add_special_tokens` bumps vocab_size."""
        base = _FakeTokenizer(vocab_size=32000)
        final = _FakeTokenizer(vocab_size=32001, added={"<|pad|>": 32000})
        assert tokenizer_grew(base, final) is True

    def test_added_token_set_change_detected(self) -> None:
        """Rare: vocab size identical, but added-tokens set differs."""
        base = _FakeTokenizer(vocab_size=32001, added={"<|a|>": 32000})
        final = _FakeTokenizer(vocab_size=32001, added={"<|b|>": 32000})
        assert tokenizer_grew(base, final) is True

    def test_bpe_like_qwen_shape(self) -> None:
        """Qwen-style BPE: large vocab + a handful of added specials."""
        base = _FakeTokenizer(
            vocab_size=151936,
            added={"<|im_start|>": 151644, "<|im_end|>": 151645},
        )
        final = _FakeTokenizer(
            vocab_size=151936,
            added={"<|im_start|>": 151644, "<|im_end|>": 151645},
        )
        assert tokenizer_grew(base, final) is False

    def test_sentencepiece_like_llama_shape(self) -> None:
        """Llama-family SentencePiece: smaller vocab, minimal added tokens."""
        base = _FakeTokenizer(vocab_size=128000, added={"<|begin_of_text|>": 128000})
        final = _FakeTokenizer(vocab_size=128001, added={"<|begin_of_text|>": 128000})
        assert tokenizer_grew(base, final) is True

    def test_pad_fallback_case(self) -> None:
        """Canonical Sprint 07 pad-fallback flow: vocab grew by exactly one."""
        base = _FakeTokenizer(vocab_size=49152)
        final = _FakeTokenizer(vocab_size=49153, added={"<|pad|>": 49152})
        assert tokenizer_grew(base, final) is True


class TestModulesToSave:
    def test_grown_returns_embed_and_lm_head(self) -> None:
        assert modules_to_save_for_growth(True) == ["embed_tokens", "lm_head"]

    def test_unchanged_returns_empty(self) -> None:
        assert modules_to_save_for_growth(False) == []

    @pytest.mark.parametrize("grew", [True, False])
    def test_returns_new_list_each_call(self, grew: bool) -> None:
        """Callers mutate the returned list; must not share state."""
        first = modules_to_save_for_growth(grew)
        second = modules_to_save_for_growth(grew)
        first.append("extra")
        assert "extra" not in second
