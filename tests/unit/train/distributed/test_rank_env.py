"""Rank / world_size detection from DDP env vars (audit-08 B1/M1)."""

from __future__ import annotations

import pytest

from dlm.train.distributed.rank_env import detect_rank, detect_world_size, is_rank_zero


class TestDetectWorldSize:
    def test_default_single_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        assert detect_world_size() == 1

    def test_empty_string_defaults_to_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "")
        assert detect_world_size() == 1

    def test_reads_env_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "4")
        assert detect_world_size() == 4

    def test_zero_clamped_to_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "0")
        assert detect_world_size() == 1

    def test_negative_clamped_to_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "-2")
        assert detect_world_size() == 1

    def test_malformed_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "not-an-int")
        with pytest.raises(ValueError, match="WORLD_SIZE"):
            detect_world_size()


class TestDetectRank:
    def test_default_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert detect_rank() == 0

    def test_rank_takes_precedence_over_local_rank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RANK", "3")
        monkeypatch.setenv("LOCAL_RANK", "1")
        assert detect_rank() == 3

    def test_falls_back_to_local_rank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "2")
        assert detect_rank() == 2

    def test_malformed_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RANK", "oops")
        with pytest.raises(ValueError, match="RANK"):
            detect_rank()


class TestIsRankZero:
    def test_single_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert is_rank_zero() is True

    def test_non_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RANK", "2")
        assert is_rank_zero() is False

    def test_rank_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RANK", "0")
        assert is_rank_zero() is True
