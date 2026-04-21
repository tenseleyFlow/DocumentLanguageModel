"""Unit tests for vocab_gap (pure-compute + render paths).

`compute_vocab_gap` is driven with hand-crafted token lists so we
don't need to materialize a real tokenizer in the unit suite.
"""

from __future__ import annotations

import pytest

from dlm.train.cpt.vocab_gap import (
    VocabGapReport,
    compute_vocab_gap,
    render_report,
)


class TestComputeVocabGap:
    def test_empty_inputs(self) -> None:
        r = compute_vocab_gap([], text="", unk_token_id=None, decoded_tokens=[])
        assert r.total_tokens == 0
        assert r.total_words == 0
        assert r.tokens_per_word == 0.0
        assert r.unk_hits == 0
        assert r.top_tokens == []
        assert not r.has_unk

    def test_one_to_one_word_ratio(self) -> None:
        r = compute_vocab_gap(
            [1, 2, 3],
            text="one two three",
            unk_token_id=None,
            decoded_tokens=["one", "two", "three"],
        )
        assert r.total_tokens == 3
        assert r.total_words == 3
        assert r.tokens_per_word == pytest.approx(1.0)

    def test_split_heavy_tokenizer(self) -> None:
        # "supercalifragilistic" → 4 subword pieces (simulated).
        r = compute_vocab_gap(
            [1, 2, 3, 4],
            text="supercalifragilistic",
            unk_token_id=None,
            decoded_tokens=["super", "cali", "fragi", "listic"],
        )
        assert r.tokens_per_word == pytest.approx(4.0)

    def test_unk_hits_counted(self) -> None:
        r = compute_vocab_gap(
            [1, 0, 2, 0, 0],
            text="hello world foo",
            unk_token_id=0,
            decoded_tokens=["hello", "<unk>", "world", "<unk>", "<unk>"],
        )
        assert r.unk_hits == 3
        assert r.has_unk

    def test_unk_ignored_when_id_is_none(self) -> None:
        r = compute_vocab_gap(
            [1, 2, 3],
            text="hello world",
            unk_token_id=None,
            decoded_tokens=["hello", "world", "!"],
        )
        assert r.unk_hits == 0

    def test_top_tokens_sorted_by_frequency_desc(self) -> None:
        decoded = ["the", "cat", "the", "the", "cat", "sat"]
        r = compute_vocab_gap(
            list(range(len(decoded))),
            text="the cat the the cat sat",
            unk_token_id=None,
            decoded_tokens=decoded,
        )
        assert r.top_tokens[0] == ("the", 3)
        assert r.top_tokens[1] == ("cat", 2)
        assert r.top_tokens[2] == ("sat", 1)

    def test_top_n_cap_respected(self) -> None:
        decoded = ["a", "b", "c", "d", "e"] * 2  # five unique
        r = compute_vocab_gap(
            list(range(len(decoded))),
            text=" ".join(decoded),
            unk_token_id=None,
            decoded_tokens=decoded,
            top_n=3,
        )
        assert len(r.top_tokens) == 3

    def test_top_n_zero_returns_empty(self) -> None:
        r = compute_vocab_gap(
            [1, 2],
            text="a b",
            unk_token_id=None,
            decoded_tokens=["a", "b"],
            top_n=0,
        )
        assert r.top_tokens == []


class TestComputeVocabGapValidation:
    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            compute_vocab_gap(
                [1, 2, 3],
                text="hi",
                unk_token_id=None,
                decoded_tokens=["a", "b"],
            )

    def test_negative_top_n_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_n must be non-negative"):
            compute_vocab_gap([], text="", unk_token_id=None, decoded_tokens=[], top_n=-1)


class TestRenderReport:
    def _basic(self, **overrides: object) -> VocabGapReport:
        defaults: dict[str, object] = {
            "total_tokens": 50,
            "total_words": 30,
            "tokens_per_word": 50 / 30,
            "unk_hits": 0,
            "top_tokens": [("the", 5), ("cat", 3)],
        }
        defaults.update(overrides)
        return VocabGapReport(**defaults)  # type: ignore[arg-type]

    def test_headline_format(self) -> None:
        text = render_report(self._basic())
        assert "vocabulary gap report" in text
        assert "tokens per word : 1.67" in text
        assert "50 tokens / 30 words" in text
        assert "<unk> hits      : 0" in text

    def test_unk_warning_present_when_hits(self) -> None:
        text = render_report(self._basic(unk_hits=7))
        assert "WARNING" in text
        assert "different base model" in text

    def test_no_unk_warning_when_zero(self) -> None:
        text = render_report(self._basic(unk_hits=0))
        assert "WARNING" not in text

    def test_top_tokens_rendered(self) -> None:
        text = render_report(self._basic())
        assert "top tokens:" in text
        assert "the" in text
        assert "5" in text
        assert "cat" in text

    def test_empty_top_tokens_omits_section(self) -> None:
        text = render_report(self._basic(top_tokens=[]))
        assert "top tokens" not in text

    def test_line_width_bounded(self) -> None:
        # Realistic worst case: long decoded token, large frequency.
        r = self._basic(top_tokens=[("aVeryLongSubwordToken", 999)])
        text = render_report(r)
        for line in text.splitlines():
            assert len(line) <= 80
