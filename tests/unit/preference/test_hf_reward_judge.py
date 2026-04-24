"""Unit tests for Sprint 42's HF reward-model judge runtime."""

from __future__ import annotations

from collections import deque

import pytest

from dlm.preference import HfRewardModelJudge, JudgeInvocationError


class FakeScalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class FakeLogits:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def numel(self) -> int:
        return len(self._values)

    def reshape(self, *_shape: int) -> FakeLogits:
        return self

    def __getitem__(self, idx: int) -> FakeScalar:
        return FakeScalar(self._values[idx])


class FakeBatch(dict[str, object]):
    def to(self, device: str) -> FakeBatch:
        self["__device__"] = device
        return self


class FakeTokenizer:
    def __init__(self, *, use_chat_template: bool = False) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        if use_chat_template:
            self.chat_template = "fake-template"

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.calls.append(("apply_chat_template", (messages,), dict(kwargs)))
        return f"templated::{messages[0]['content']}::{messages[1]['content']}"

    def __call__(self, *args: object, **kwargs: object) -> FakeBatch:
        self.calls.append(("tokenizer", args, dict(kwargs)))
        return FakeBatch({"input_ids": object()})


class FakeModel:
    def __init__(self, logits: list[FakeLogits]) -> None:
        self._logits = deque(logits)
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object):
        self.calls.append(dict(kwargs))

        class Output:
            def __init__(self, logits: FakeLogits) -> None:
                self.logits = logits

        return Output(self._logits.popleft())


def _loader_factory(tokenizer: FakeTokenizer, model: FakeModel):
    calls: list[tuple[str, str]] = []

    def _loader(hf_id: str, device: str):
        calls.append((hf_id, device))
        from dlm.preference.judge import _LoadedRewardJudge

        return _LoadedRewardJudge(model=model, tokenizer=tokenizer, device=device)

    return calls, _loader


class TestHfRewardModelJudge:
    def test_scores_pair_and_caches_loaded_bundle(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel([FakeLogits([0.2]), FakeLogits([0.9])])
        calls, loader = _loader_factory(tokenizer, model)
        judge = HfRewardModelJudge("reward/model", device="cpu", loader=loader)

        score = judge.score_pair("What is DGEMM?", "bad", "good")

        assert score.score_a == pytest.approx(0.2)
        assert score.score_b == pytest.approx(0.9)
        assert score.preferred == "b"
        assert calls == [("reward/model", "cpu")]
        assert len(model.calls) == 2
        first_call = tokenizer.calls[0]
        assert first_call[0] == "tokenizer"
        assert first_call[1] == ("What is DGEMM?",)
        assert first_call[2]["text_pair"] == "bad"

    def test_chat_template_path_is_used_when_available(self) -> None:
        tokenizer = FakeTokenizer(use_chat_template=True)
        model = FakeModel([FakeLogits([0.4]), FakeLogits([0.1])])
        _, loader = _loader_factory(tokenizer, model)
        judge = HfRewardModelJudge("reward/model", device="cpu", loader=loader)

        judge.score_pair("prompt", "cand-a", "cand-b")

        assert tokenizer.calls[0][0] == "apply_chat_template"
        assert tokenizer.calls[1][0] == "tokenizer"
        assert tokenizer.calls[1][1] == ("templated::prompt::cand-a",)

    def test_non_scalar_logits_are_refused(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel([FakeLogits([0.1, 0.2]), FakeLogits([0.3])])
        _, loader = _loader_factory(tokenizer, model)
        judge = HfRewardModelJudge("reward/model", device="cpu", loader=loader)

        with pytest.raises(JudgeInvocationError, match="single scalar logit"):
            judge.score_pair("prompt", "a", "b")

    def test_missing_logits_are_refused(self) -> None:
        tokenizer = FakeTokenizer()

        class NoLogitsModel:
            def __call__(self, **_kwargs: object):
                class Output:
                    pass

                return Output()

        calls: list[tuple[str, str]] = []

        def loader(hf_id: str, device: str):
            calls.append((hf_id, device))
            from dlm.preference.judge import _LoadedRewardJudge

            return _LoadedRewardJudge(model=NoLogitsModel(), tokenizer=tokenizer, device=device)

        judge = HfRewardModelJudge("reward/model", device="cpu", loader=loader)

        with pytest.raises(JudgeInvocationError, match="no `.logits`"):
            judge.score_pair("prompt", "a", "b")
        assert calls == [("reward/model", "cpu")]
