"""Unit tests for Sprint 42's sway-backed preference judge."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from dlm.preference import JudgeInvocationError, SwayJudge


class FakeFtView:
    def __init__(
        self,
        scores: dict[tuple[str, str], float],
        *,
        calls: list[tuple[str, str]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._scores = scores
        self._calls = calls if calls is not None else []
        self._error = error

    def logprob_of(self, prompt: str, completion: str) -> float:
        self._calls.append((prompt, completion))
        if self._error is not None:
            raise self._error
        return self._scores[(prompt, completion)]


class FakeBackend:
    def __init__(
        self,
        scores: dict[tuple[str, str], float],
        *,
        error: Exception | None = None,
    ) -> None:
        self._scores = scores
        self._error = error
        self.enter_count = 0
        self.calls: list[tuple[str, str]] = []

    @contextmanager
    def as_finetuned(self) -> Iterator[FakeFtView]:
        self.enter_count += 1
        yield FakeFtView(self._scores, calls=self.calls, error=self._error)


def _backend_factory(backend: FakeBackend):
    calls: list[Path] = []

    def _factory(dlm_path: Path) -> FakeBackend:
        calls.append(dlm_path)
        return backend

    return calls, _factory


class TestSwayJudge:
    def test_scores_pair_via_finetuned_backend_and_normalizes_length(self) -> None:
        backend = FakeBackend(
            {
                ("Explain DGEMM", "tiny"): -1.0,
                ("Explain DGEMM", "significantly better"): -2.5,
            }
        )
        _, factory = _backend_factory(backend)
        judge = SwayJudge(Path("/tmp/example.dlm"), backend_factory=factory)

        score = judge.score_pair("Explain DGEMM", "tiny", "significantly better")

        assert score.score_a == pytest.approx(-1.0)
        assert score.score_b == pytest.approx(-0.5)
        assert score.preferred == "b"
        assert score.margin == pytest.approx(-0.5)
        assert "mean-logprob delta" in (score.reasoning or "")
        assert backend.enter_count == 1
        assert backend.calls == [
            ("Explain DGEMM", "tiny"),
            ("Explain DGEMM", "significantly better"),
        ]

    def test_backend_factory_is_cached_across_calls(self) -> None:
        backend = FakeBackend({("p", "a"): -0.2, ("p", "b"): -0.1})
        factory_calls, factory = _backend_factory(backend)
        judge = SwayJudge(Path("/tmp/example.dlm"), backend_factory=factory)

        judge.score_pair("p", "a", "b")
        judge.score_pair("p", "a", "b")

        assert factory_calls == [Path("/tmp/example.dlm")]
        assert backend.enter_count == 2

    def test_scoring_failures_are_wrapped(self) -> None:
        backend = FakeBackend({("p", "a"): -0.2}, error=RuntimeError("backend exploded"))
        _, factory = _backend_factory(backend)
        judge = SwayJudge(Path("/tmp/example.dlm"), backend_factory=factory)

        with pytest.raises(JudgeInvocationError, match="failed to score candidates"):
            judge.score_pair("p", "a", "b")
