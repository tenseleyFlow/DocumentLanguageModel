from __future__ import annotations

import logging
import os

from dlm.train.cache import DISABLE_ENV_VAR, disabled_cache, is_cache_disabled, set_disable_flag


class TestCacheDisableFlag:
    def test_disabled_false_by_default(self, monkeypatch) -> None:
        monkeypatch.delenv(DISABLE_ENV_VAR, raising=False)
        assert is_cache_disabled() is False

    def test_set_disable_flag_sets_env_and_logs(self, monkeypatch, caplog) -> None:
        monkeypatch.delenv(DISABLE_ENV_VAR, raising=False)
        with caplog.at_level(logging.INFO):
            set_disable_flag("cli flag")
        assert is_cache_disabled() is True
        assert "tokenized cache disabled (cli flag)" in caplog.text

    def test_disabled_cache_restores_missing_prior_value(self, monkeypatch) -> None:
        monkeypatch.delenv(DISABLE_ENV_VAR, raising=False)
        with disabled_cache("scoped test"):
            assert is_cache_disabled() is True
        assert DISABLE_ENV_VAR not in os.environ
        assert is_cache_disabled() is False

    def test_disabled_cache_restores_prior_value(self, monkeypatch) -> None:
        monkeypatch.setenv(DISABLE_ENV_VAR, "0")
        with disabled_cache("scoped test"):
            assert is_cache_disabled() is True
        assert is_cache_disabled() is False

    def test_disabled_cache_preserves_existing_disabled_state(self, monkeypatch) -> None:
        monkeypatch.setenv(DISABLE_ENV_VAR, "1")
        with disabled_cache("already disabled"):
            assert is_cache_disabled() is True
        assert is_cache_disabled() is True
