"""Unit tests for Sprint 42's HF reward-model judge runtime."""

from __future__ import annotations

import builtins
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dlm.preference import (
    HfRewardModelJudge,
    InvalidJudgeSpecError,
    JudgeInvocationError,
    JudgeUnavailableError,
)
from dlm.preference.judge import (
    _default_reward_loader,
    _encode_reward_input,
    _extract_reward_scalar,
    _move_to_device,
    _resolve_reward_device,
)


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


class FakeTensor:
    def __init__(self) -> None:
        self.device: str | None = None

    def to(self, device: str) -> FakeTensor:
        self.device = device
        return self


class FakeTokenizer:
    def __init__(
        self,
        *,
        use_chat_template: bool = False,
        template_error: Exception | None = None,
        template_returns_non_string: bool = False,
    ) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self._template_error = template_error
        self._template_returns_non_string = template_returns_non_string
        if use_chat_template:
            self.chat_template = "fake-template"

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.calls.append(("apply_chat_template", (messages,), dict(kwargs)))
        if self._template_error is not None:
            raise self._template_error
        if self._template_returns_non_string:
            return ""  # type: ignore[return-value]
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


class FakeTorchScalarLogits:
    def __init__(self, value: float) -> None:
        self._value = value

    def numel(self) -> int:
        return 1

    def item(self) -> float:
        return self._value


class FakePretrainedRewardModel:
    def __init__(self) -> None:
        self.device: str | None = None
        self.eval_called = False

    def to(self, device: str) -> FakePretrainedRewardModel:
        self.device = device
        return self

    def eval(self) -> None:
        self.eval_called = True


def _loader_factory(tokenizer: FakeTokenizer, model: FakeModel):
    calls: list[tuple[str, str]] = []

    def _loader(hf_id: str, device: str):
        calls.append((hf_id, device))
        from dlm.preference.judge import _LoadedRewardJudge

        return _LoadedRewardJudge(model=model, tokenizer=tokenizer, device=device)

    return calls, _loader


class TestHfRewardModelJudge:
    def test_blank_selector_is_rejected(self) -> None:
        with pytest.raises(InvalidJudgeSpecError, match="include a model id"):
            HfRewardModelJudge("   ")

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

    def test_missing_torch_is_reported(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel([FakeLogits([0.2]), FakeLogits([0.1])])
        _, loader = _loader_factory(tokenizer, model)
        judge = HfRewardModelJudge("reward/model", device="cpu", loader=loader)
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object):
            if name == "torch":
                raise ImportError("no torch here")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            pytest.raises(JudgeUnavailableError, match="requires torch"),
        ):
            judge.score_pair("prompt", "a", "b")

    def test_default_loader_path_is_used_when_no_loader_is_supplied(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel([FakeLogits([0.7]), FakeLogits([0.1])])

        def fake_default_loader(hf_id: str, device: str):
            from dlm.preference.judge import _LoadedRewardJudge

            assert hf_id == "reward/model"
            assert device == "cpu"
            return _LoadedRewardJudge(model=model, tokenizer=tokenizer, device=device)

        judge = HfRewardModelJudge("reward/model", device="cpu")
        with patch("dlm.preference.judge._default_reward_loader", side_effect=fake_default_loader):
            score = judge.score_pair("prompt", "a", "b")

        assert score.preferred == "a"


class TestHfRewardHelpers:
    def test_default_reward_loader_requires_transformers(self) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object):
            if name == "transformers":
                raise ImportError("missing transformers")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            pytest.raises(JudgeUnavailableError, match="requires transformers"),
        ):
            _default_reward_loader("reward/model", "cpu")

    def test_default_reward_loader_moves_model_and_sets_eval(self) -> None:
        model = FakePretrainedRewardModel()
        tokenizer = FakeTokenizer()

        class AutoModelForSequenceClassification:
            @staticmethod
            def from_pretrained(hf_id: str) -> FakePretrainedRewardModel:
                assert hf_id == "reward/model"
                return model

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(hf_id: str) -> FakeTokenizer:
                assert hf_id == "reward/model"
                return tokenizer

        fake_transformers = SimpleNamespace(
            AutoModelForSequenceClassification=AutoModelForSequenceClassification,
            AutoTokenizer=AutoTokenizer,
        )

        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            loaded = _default_reward_loader("reward/model", "mps")

        assert loaded.model is model
        assert loaded.tokenizer is tokenizer
        assert loaded.device == "mps"
        assert model.device == "mps"
        assert model.eval_called is True

    def test_resolve_reward_device_respects_explicit_request(self) -> None:
        assert _resolve_reward_device("cuda:3") == "cuda:3"

    def test_resolve_reward_device_returns_cpu_when_torch_is_missing(self) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            assert _resolve_reward_device("auto") == "cpu"

    def test_resolve_reward_device_prefers_cuda_then_mps_then_cpu(self) -> None:
        torch_cuda = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        )
        torch_mps = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        )
        torch_cpu = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        )

        with patch.dict("sys.modules", {"torch": torch_cuda}):
            assert _resolve_reward_device("auto") == "cuda"
        with patch.dict("sys.modules", {"torch": torch_mps}):
            assert _resolve_reward_device("auto") == "mps"
        with patch.dict("sys.modules", {"torch": torch_cpu}):
            assert _resolve_reward_device("auto") == "cpu"

    def test_encode_reward_input_falls_back_when_template_raises(self) -> None:
        tokenizer = FakeTokenizer(use_chat_template=True, template_error=RuntimeError("boom"))

        encoded = _encode_reward_input(tokenizer, "prompt", "candidate")

        assert isinstance(encoded, FakeBatch)
        assert tokenizer.calls[-1][0] == "tokenizer"
        assert tokenizer.calls[-1][1] == ("prompt",)
        assert tokenizer.calls[-1][2]["text_pair"] == "candidate"

    def test_encode_reward_input_falls_back_when_template_returns_non_string(self) -> None:
        tokenizer = FakeTokenizer(use_chat_template=True, template_returns_non_string=True)

        encoded = _encode_reward_input(tokenizer, "prompt", "candidate")

        assert isinstance(encoded, FakeBatch)
        assert tokenizer.calls[-1][0] == "tokenizer"

    def test_move_to_device_moves_mapping_values(self) -> None:
        tensor = FakeTensor()
        payload = {"input_ids": tensor, "meta": "keep"}

        moved = _move_to_device(payload, "mps")

        assert moved["input_ids"] is tensor
        assert tensor.device == "mps"
        assert moved["meta"] == "keep"

    def test_move_to_device_returns_unmodified_non_mapping_values(self) -> None:
        value = object()
        assert _move_to_device(value, "cpu") is value

    def test_extract_reward_scalar_uses_item_fallback(self) -> None:
        assert _extract_reward_scalar(FakeTorchScalarLogits(0.75)) == pytest.approx(0.75)

    def test_extract_reward_scalar_rejects_unreadable_values(self) -> None:
        class UnreadableLogits:
            def numel(self) -> int:
                return 1

        with pytest.raises(JudgeInvocationError, match="unreadable scalar logit"):
            _extract_reward_scalar(UnreadableLogits())
