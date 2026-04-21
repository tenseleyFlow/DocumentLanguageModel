"""TokenizedCache — hit/miss, LRU, atomic put, tokenizer-sha invalidation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlm.directives.cache import CachedTokens, TokenizedCache
from dlm.directives.cache_key import CacheKey


def _tokens(length: int) -> CachedTokens:
    return CachedTokens(
        input_ids=np.arange(length, dtype=np.int64),
        attention_mask=np.ones(length, dtype=np.int64),
    )


def _key(section_id: str = "a1b2c3d4e5f67890", sequence_len: int = 2048) -> CacheKey:
    return CacheKey(section_id=section_id, tokenizer_sha="a" * 64, sequence_len=sequence_len)


class TestOpen:
    def test_creates_layout(self, tmp_path: Path) -> None:
        root = tmp_path / "c"
        TokenizedCache.open(root)
        assert root.is_dir()
        assert (root / "entries").is_dir()

    def test_empty_cache(self, tmp_path: Path) -> None:
        cache = TokenizedCache.open(tmp_path / "c")
        assert cache.entry_count == 0
        assert cache.total_bytes == 0

    def test_corrupt_manifest_starts_fresh(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        root = tmp_path / "c"
        root.mkdir()
        (root / "manifest.json").write_text("{not valid json")
        caplog.set_level(logging.WARNING, logger="dlm.directives.cache")
        cache = TokenizedCache.open(root)
        assert cache.entry_count == 0
        assert any("unreadable" in rec.message for rec in caplog.records)


class TestGetPut:
    def test_miss_then_hit(self, tmp_path: Path) -> None:
        cache = TokenizedCache.open(tmp_path / "c")
        key = _key()
        assert cache.get(key) is None
        assert cache.misses == 1
        cache.put(key, _tokens(4))
        hit = cache.get(key)
        assert hit is not None
        assert hit.input_ids.tolist() == [0, 1, 2, 3]
        assert cache.hits == 1

    def test_bit_identical_round_trip(self, tmp_path: Path) -> None:
        cache = TokenizedCache.open(tmp_path / "c")
        key = _key()
        original = _tokens(16)
        cache.put(key, original)
        retrieved = cache.get(key)
        assert retrieved is not None
        assert np.array_equal(retrieved.input_ids, original.input_ids)
        assert np.array_equal(retrieved.attention_mask, original.attention_mask)

    def test_survives_reopen(self, tmp_path: Path) -> None:
        root = tmp_path / "c"
        cache = TokenizedCache.open(root)
        key = _key()
        cache.put(key, _tokens(8))
        cache.save_manifest(tokenizer_sha="a" * 64)

        cache2 = TokenizedCache.open(root)
        assert cache2.entry_count == 1
        assert cache2.get(key) is not None


class TestInvalidation:
    def test_different_tokenizer_sha_misses(self, tmp_path: Path) -> None:
        cache = TokenizedCache.open(tmp_path / "c")
        key_a = CacheKey(section_id="aa" * 8, tokenizer_sha="a" * 64, sequence_len=2048)
        key_b = CacheKey(section_id="aa" * 8, tokenizer_sha="b" * 64, sequence_len=2048)
        cache.put(key_a, _tokens(4))
        # New tokenizer sha → miss
        assert cache.get(key_b) is None

    def test_different_sequence_len_misses(self, tmp_path: Path) -> None:
        cache = TokenizedCache.open(tmp_path / "c")
        key_a = CacheKey(section_id="aa" * 8, tokenizer_sha="a" * 64, sequence_len=2048)
        key_b = CacheKey(section_id="aa" * 8, tokenizer_sha="a" * 64, sequence_len=1024)
        cache.put(key_a, _tokens(4))
        assert cache.get(key_b) is None

    def test_missing_file_recovers(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """If the on-disk entry vanishes under us, get() should treat
        it as a miss and clean up the stale manifest row."""
        import logging

        cache = TokenizedCache.open(tmp_path / "c")
        key = _key()
        cache.put(key, _tokens(4))
        # Delete the entry file behind the cache's back
        entry_file = next((tmp_path / "c" / "entries").rglob("*.npz"))
        entry_file.unlink()

        caplog.set_level(logging.WARNING, logger="dlm.directives.cache")
        assert cache.get(key) is None
        assert any("missing" in rec.message for rec in caplog.records)
        assert cache.entry_count == 0


class TestLRUEviction:
    def test_evicts_when_over_budget(self, tmp_path: Path) -> None:
        import time

        # Very tight budget so every subsequent put triggers eviction.
        cache = TokenizedCache.open(tmp_path / "c", max_bytes=100)
        key_a = _key("aa" * 8)
        key_b = _key("bb" * 8)
        cache.put(key_a, _tokens(20))
        cache._touched_this_run.clear()  # simulate prior run
        time.sleep(0.01)
        cache.put(key_b, _tokens(20))
        cache._touched_this_run.clear()
        # At this point both entries are from "prior runs" and the
        # cache is already over budget. Next put should evict.
        key_c = _key("cc" * 8)
        cache.put(key_c, _tokens(20))
        # key_c just inserted → present
        assert cache.get(key_c) is not None
        # Both older entries should have been evicted to make room
        assert cache.get(key_a) is None
        assert cache.get(key_b) is None

    def test_current_run_entries_protected(self, tmp_path: Path) -> None:
        """LRU must not evict entries inserted in the current run."""
        cache = TokenizedCache.open(tmp_path / "c", max_bytes=100)
        key_a = _key("aa" * 8)
        key_b = _key("bb" * 8)
        cache.put(key_a, _tokens(20))
        cache.put(key_b, _tokens(20))
        # Both were put in this run — both must still resolve
        assert cache.get(key_a) is not None
        assert cache.get(key_b) is not None


class TestPruneClear:
    def test_prune_removes_old_entries(self, tmp_path: Path) -> None:
        cache = TokenizedCache.open(tmp_path / "c")
        key = _key()
        cache.put(key, _tokens(4))
        # Force-age the entry
        entry = next(iter(cache._manifest.values()))
        entry.last_access_ts = 0  # Jan 1 1970

        removed = cache.prune(older_than_seconds=60)
        assert removed == 1
        assert cache.entry_count == 0

    def test_clear_wipes_everything(self, tmp_path: Path) -> None:
        cache = TokenizedCache.open(tmp_path / "c")
        cache.put(_key("aa" * 8), _tokens(4))
        cache.put(_key("bb" * 8), _tokens(4))
        count = cache.clear()
        assert count == 2
        assert cache.entry_count == 0


class TestAtomicPut:
    def test_no_tmp_files_left_on_success(self, tmp_path: Path) -> None:
        cache = TokenizedCache.open(tmp_path / "c")
        cache.put(_key(), _tokens(4))
        tmp_files = list((tmp_path / "c" / "entries").rglob("*.tmp"))
        assert tmp_files == []

    def test_manifest_save_is_atomic(self, tmp_path: Path) -> None:
        cache = TokenizedCache.open(tmp_path / "c")
        cache.put(_key(), _tokens(4))
        cache.save_manifest(tokenizer_sha="a" * 64)
        # No tmp manifest left behind
        assert not (tmp_path / "c" / "manifest.json.tmp").exists()
        assert (tmp_path / "c" / "manifest.json").is_file()
