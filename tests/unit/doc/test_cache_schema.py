"""`TrainingConfig.cache` — schema shape + validators."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlm.doc.schema import CacheConfig, TrainingConfig


class TestCacheConfigDefaults:
    def test_defaults_match_pre_v9_behavior(self) -> None:
        cfg = CacheConfig()
        assert cfg.enabled is True
        assert cfg.max_bytes == 10 * 1024 * 1024 * 1024  # 10 GiB
        assert cfg.prune_older_than_days == 90

    def test_defaults_are_frozen(self) -> None:
        cfg = CacheConfig()
        # Pydantic frozen model → assignment raises ValidationError.
        with pytest.raises(ValidationError):
            cfg.enabled = False  # type: ignore[misc]


class TestCacheConfigValidators:
    def test_max_bytes_must_be_at_least_one(self) -> None:
        with pytest.raises(ValidationError):
            CacheConfig(max_bytes=0)

    def test_max_bytes_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            CacheConfig(max_bytes=-1)

    def test_prune_days_must_be_at_least_one(self) -> None:
        with pytest.raises(ValidationError):
            CacheConfig(prune_older_than_days=0)

    def test_prune_days_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            CacheConfig(prune_older_than_days=-5)


class TestCacheConfigForbidsUnknownKeys:
    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc:
            CacheConfig(mystery_field=123)  # type: ignore[call-arg]
        assert "mystery_field" in str(exc.value) or "extra_forbidden" in str(exc.value)


class TestTrainingConfigWiring:
    def test_default_cache_is_populated(self) -> None:
        tc = TrainingConfig()
        assert isinstance(tc.cache, CacheConfig)
        assert tc.cache.enabled is True
        assert tc.cache.max_bytes == 10 * 1024 * 1024 * 1024

    def test_explicit_override_wins(self) -> None:
        tc = TrainingConfig(
            cache=CacheConfig(enabled=False, max_bytes=1024, prune_older_than_days=7)
        )
        assert tc.cache.enabled is False
        assert tc.cache.max_bytes == 1024
        assert tc.cache.prune_older_than_days == 7
