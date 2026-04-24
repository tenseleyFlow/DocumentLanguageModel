"""Pre-tokenize pass — cache glue, row shape, row passthrough.

Unit tests use a stub tokenizer so the cache-consumer logic is
covered without pulling in a real HF model. The bit-identity check
against TRL's own tokenization path lives in the slow integration
suite (``tests/integration/train/test_cache_speedup.py``) where a
real tokenizer is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from transformers import PreTrainedTokenizerBase

from dlm.directives.cache import TokenizedCache
from dlm.train.tokenization import (
    _as_int_list,
    TokenizationStats,
    pretokenize_rows,
)


class _StubTokenizer:
    """Deterministic tokenizer stand-in.

    - `messages` → flattened content string → charcode list.
    - `text` → charcode list of the input string.

    The result is stable across runs (pure function of the input) so
    miss-then-hit tests can assert bit-identity without instantiating a
    real BPE tokenizer. The stub also exposes `backend_tokenizer` so
    `tokenizer_sha256` takes the fast-path.
    """

    eos_token = "<eos>"

    class _Backend:
        def __init__(self, salt: str = "stub-v1") -> None:
            self._salt = salt

        def to_str(self) -> str:
            return self._salt

    def __init__(self, *, salt: str = "stub-v1") -> None:
        self.backend_tokenizer = self._Backend(salt)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        return_dict: bool,
        add_generation_prompt: bool = False,  # noqa: ARG002
    ) -> dict[str, list[int]]:
        assert tokenize, "pretokenize path must request tokenize"
        assert return_dict, "pretokenize path must request return_dict"
        text = "|".join(f"{m['role']}:{m['content']}" for m in messages)
        ids = [ord(c) for c in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def __call__(self, *, text: str) -> dict[str, list[int]]:
        ids = [ord(c) for c in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _tok(stub: _StubTokenizer) -> PreTrainedTokenizerBase:
    """Cast the stub to the tokenizer protocol — production callers
    pass a real ``PreTrainedTokenizerBase``; tests supply a duck-typed
    stand-in that exposes the same three methods the module touches."""
    return cast(PreTrainedTokenizerBase, stub)


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "messages": [
                {"role": "user", "content": "Q?"},
                {"role": "assistant", "content": "A."},
            ],
            "_dlm_section_id": "aaaaaaaaaaaaaaaa",
        },
        {"text": "prose body", "_dlm_section_id": "bbbbbbbbbbbbbbbb"},
    ]


class TestBasicPretokenize:
    def test_no_cache_tokenizes_every_row(self) -> None:
        tok = _StubTokenizer()
        out, stats = pretokenize_rows(_rows(), tokenizer=_tok(tok), sequence_len=512)
        assert len(out) == 2
        assert stats.total_sections == 2
        assert stats.cache_hits == 0
        assert stats.cache_misses == 2
        assert stats.cache_bytes_after == 0
        # Output shape is TRL's is_processed shape.
        for row in out:
            assert set(row.keys()) == {"input_ids", "attention_mask", "_dlm_section_id"}
            assert isinstance(row["input_ids"], list)
            assert all(isinstance(x, int) for x in row["input_ids"])
            assert len(row["attention_mask"]) == len(row["input_ids"])

    def test_messages_round_trip(self) -> None:
        tok = _StubTokenizer()
        out, _stats = pretokenize_rows(_rows()[:1], tokenizer=_tok(tok), sequence_len=512)
        expected_text = "user:Q?|assistant:A."
        assert out[0]["input_ids"] == [ord(c) for c in expected_text]

    def test_text_row_appends_eos(self) -> None:
        tok = _StubTokenizer()
        row = {"text": "hello", "_dlm_section_id": "cc" * 8}
        out, _stats = pretokenize_rows([row], tokenizer=_tok(tok), sequence_len=32)
        assert out[0]["input_ids"] == [ord(c) for c in "hello" + tok.eos_token]

    def test_text_row_with_eos_already_present_not_duplicated(self) -> None:
        tok = _StubTokenizer()
        row = {"text": "hi" + tok.eos_token, "_dlm_section_id": "dd" * 8}
        out, _stats = pretokenize_rows([row], tokenizer=_tok(tok), sequence_len=32)
        assert out[0]["input_ids"] == [ord(c) for c in "hi" + tok.eos_token]

    def test_preference_row_passthrough(self) -> None:
        tok = _StubTokenizer()
        pref = {
            "prompt": "p",
            "chosen": "c",
            "rejected": "r",
            "_dlm_section_id": "ff" * 8,
        }
        out, stats = pretokenize_rows([pref], tokenizer=_tok(tok), sequence_len=32)
        # Preference rows untouched — DPOTrainer owns their tokenization.
        assert out[0] == pref
        # They don't count toward miss/hit (the sprint scopes the cache
        # to SFT-path sections).
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0

    def test_unknown_row_raises(self) -> None:
        tok = _StubTokenizer()
        bad = {"_dlm_section_id": "aa" * 8}
        with pytest.raises(ValueError, match="has neither"):
            pretokenize_rows([bad], tokenizer=_tok(tok), sequence_len=32)


class TestCacheIntegration:
    def test_miss_then_hit_is_bit_identical(self, tmp_path: Path) -> None:
        tok = _StubTokenizer()
        cache = TokenizedCache.open(tmp_path / "cache")

        first_out, first_stats = pretokenize_rows(
            _rows(), tokenizer=_tok(tok), sequence_len=256, cache=cache
        )
        assert first_stats.cache_hits == 0
        assert first_stats.cache_misses == 2

        # Reopen cache (fresh session) so `_touched_this_run` is reset.
        cache.save_manifest()
        cache2 = TokenizedCache.open(tmp_path / "cache")
        second_out, second_stats = pretokenize_rows(
            _rows(), tokenizer=_tok(tok), sequence_len=256, cache=cache2
        )
        assert second_stats.cache_hits == 2
        assert second_stats.cache_misses == 0

        # Byte-identical input_ids + attention_mask across runs.
        for a, b in zip(first_out, second_out, strict=True):
            assert a["input_ids"] == b["input_ids"]
            assert a["attention_mask"] == b["attention_mask"]

    def test_tokenizer_sha_bump_invalidates(self, tmp_path: Path) -> None:
        tok_a = _StubTokenizer(salt="tok-A")
        tok_b = _StubTokenizer(salt="tok-B")
        cache = TokenizedCache.open(tmp_path / "cache")

        _out_a, stats_a = pretokenize_rows(
            _rows(), tokenizer=_tok(tok_a), sequence_len=256, cache=cache
        )
        assert stats_a.cache_misses == 2

        _out_b, stats_b = pretokenize_rows(
            _rows(), tokenizer=_tok(tok_b), sequence_len=256, cache=cache
        )
        # Different tokenizer sha → all misses again.
        assert stats_b.cache_misses == 2
        assert stats_b.cache_hits == 0

    def test_sequence_len_bump_invalidates(self, tmp_path: Path) -> None:
        tok = _StubTokenizer()
        cache = TokenizedCache.open(tmp_path / "cache")

        _out_a, stats_a = pretokenize_rows(
            _rows(), tokenizer=_tok(tok), sequence_len=128, cache=cache
        )
        assert stats_a.cache_misses == 2

        _out_b, stats_b = pretokenize_rows(
            _rows(), tokenizer=_tok(tok), sequence_len=512, cache=cache
        )
        # New sequence_len → new key → misses.
        assert stats_b.cache_misses == 2
        assert stats_b.cache_hits == 0

    def test_rows_without_section_id_skip_cache(self, tmp_path: Path) -> None:
        tok = _StubTokenizer()
        cache = TokenizedCache.open(tmp_path / "cache")

        rows = [{"text": "unkeyed prose"}]  # no _dlm_section_id
        out, stats = pretokenize_rows(rows, tokenizer=_tok(tok), sequence_len=32, cache=cache)
        assert len(out) == 1
        assert stats.cache_misses == 1
        # The cache didn't gain an entry because the row had no key.
        assert cache.entry_count == 0

    def test_cache_bytes_after_reflects_disk_state(self, tmp_path: Path) -> None:
        tok = _StubTokenizer()
        cache = TokenizedCache.open(tmp_path / "cache")
        _, stats = pretokenize_rows(_rows(), tokenizer=_tok(tok), sequence_len=256, cache=cache)
        assert stats.cache_bytes_after == cache.total_bytes
        assert stats.cache_bytes_after > 0


class TestStatsDataclass:
    def test_frozen(self) -> None:
        import dataclasses

        s = TokenizationStats(
            total_sections=2,
            cache_hits=1,
            cache_misses=1,
            total_tokenize_seconds=0.001,
            cache_bytes_after=100,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.total_sections = 3  # type: ignore[misc]


class TestAsIntList:
    def test_numpy_batch_of_one_is_flattened(self) -> None:
        arr = np.asarray([[1, 2, 3]], dtype=np.int64)
        assert _as_int_list(arr) == [1, 2, 3]

    def test_tolist_like_object_is_flattened(self) -> None:
        class _FakeTensor:
            def tolist(self) -> list[list[int]]:
                return [[4, 5, 6]]

        assert _as_int_list(_FakeTensor()) == [4, 5, 6]

    def test_plain_iterable_falls_back_to_iteration(self) -> None:
        assert _as_int_list((7, 8, 9)) == [7, 8, 9]
