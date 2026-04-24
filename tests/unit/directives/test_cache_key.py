"""CacheKey + tokenizer_sha256 — determinism + sensitivity."""

from __future__ import annotations

import pytest

from dlm.directives.cache_key import CacheKey, tokenizer_sha256


class TestCacheKey:
    def test_as_filename_is_deterministic(self) -> None:
        key = CacheKey(
            section_id="ab12cd34ef567890",
            tokenizer_sha="a" * 64,
            sequence_len=2048,
        )
        assert key.as_filename() == "ab12cd34ef567890.aaaaaaaaaaaa.seq2048.npz"

    def test_shard_is_first_two_hex(self) -> None:
        key = CacheKey(section_id="ff12cd34ef567890", tokenizer_sha="a" * 64, sequence_len=1024)
        assert key.shard() == "ff"

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            # section_id changes → different filename
            (
                CacheKey("ab12cd34ef567890", "a" * 64, 2048),
                CacheKey("ab12cd34ef567891", "a" * 64, 2048),
            ),
            # tokenizer_sha changes → different filename
            (
                CacheKey("ab12cd34ef567890", "a" * 64, 2048),
                CacheKey("ab12cd34ef567890", "b" * 64, 2048),
            ),
            # sequence_len changes → different filename
            (
                CacheKey("ab12cd34ef567890", "a" * 64, 2048),
                CacheKey("ab12cd34ef567890", "a" * 64, 1024),
            ),
        ],
    )
    def test_each_input_changes_filename(self, a: CacheKey, b: CacheKey) -> None:
        assert a.as_filename() != b.as_filename()


class _FakeBackendTokenizer:
    """Stand-in for `tokenizer.backend_tokenizer` with `to_str`."""

    def __init__(self, canonical: str) -> None:
        self._canonical = canonical

    def to_str(self) -> str:
        return self._canonical


class _BrokenBackendTokenizer:
    def to_str(self) -> str:
        raise RuntimeError("boom")


class _FakeTokenizer:
    """Minimal shape for tokenizer_sha256 — just enough attrs."""

    def __init__(self, *, canonical: str | None = None, vocab_size: int = 32000) -> None:
        self.backend_tokenizer: object | None = (
            _FakeBackendTokenizer(canonical) if canonical else None
        )
        self.vocab_size = vocab_size
        self.model_max_length = 2048
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.bos_token = "<s>"
        self.unk_token = "<unk>"
        self.cls_token = ""
        self.sep_token = ""
        self.mask_token = ""
        self.added_tokens_decoder: dict[int, str] = {}


class TestTokenizerSha256:
    def test_fast_tokenizer_path(self) -> None:
        tok = _FakeTokenizer(canonical='{"type": "BPE", "vocab": {"a": 1}}')
        sha = tokenizer_sha256(tok)
        assert len(sha) == 64
        # Deterministic: same input → same sha
        assert tokenizer_sha256(tok) == sha

    def test_canonical_change_flips_sha(self) -> None:
        tok_a = _FakeTokenizer(canonical='{"v": 1}')
        tok_b = _FakeTokenizer(canonical='{"v": 2}')
        assert tokenizer_sha256(tok_a) != tokenizer_sha256(tok_b)

    def test_legacy_path_when_no_backend(self) -> None:
        tok = _FakeTokenizer()  # no backend_tokenizer
        sha = tokenizer_sha256(tok)
        assert len(sha) == 64

    def test_legacy_vocab_change_flips_sha(self) -> None:
        tok_a = _FakeTokenizer(vocab_size=32000)
        tok_b = _FakeTokenizer(vocab_size=64000)
        assert tokenizer_sha256(tok_a) != tokenizer_sha256(tok_b)

    def test_pinned_on_instance(self) -> None:
        tok = _FakeTokenizer(canonical='{"v": 1}')
        sha1 = tokenizer_sha256(tok)
        # Swap canonical underneath — pinned value persists
        tok.backend_tokenizer = _FakeBackendTokenizer('{"v": 2}')
        sha2 = tokenizer_sha256(tok)
        assert sha1 == sha2

    def test_backend_to_str_failure_falls_back_to_legacy(self) -> None:
        tok = _FakeTokenizer()
        tok.backend_tokenizer = _BrokenBackendTokenizer()
        sha = tokenizer_sha256(tok)
        assert len(sha) == 64
