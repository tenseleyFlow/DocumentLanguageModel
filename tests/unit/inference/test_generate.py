"""`build_generate_kwargs` — deterministic vs. sampled argument resolution."""

from __future__ import annotations

import pytest

from dlm.inference.generate import DEFAULT_MAX_NEW_TOKENS, build_generate_kwargs


class TestDeterministicPath:
    def test_temperature_zero_is_deterministic(self) -> None:
        kwargs = build_generate_kwargs(max_new_tokens=32, temperature=0.0)
        assert kwargs["do_sample"] is False
        assert kwargs["num_beams"] == 1
        # Temperature must NOT leak through when do_sample=False.
        assert "temperature" not in kwargs
        assert kwargs["max_new_tokens"] == 32

    def test_default_max_new_tokens(self) -> None:
        kwargs = build_generate_kwargs()
        assert kwargs["max_new_tokens"] == DEFAULT_MAX_NEW_TOKENS


class TestSampledPath:
    def test_non_zero_temperature_flips_sampling(self) -> None:
        kwargs = build_generate_kwargs(max_new_tokens=100, temperature=0.7)
        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == pytest.approx(0.7)
        assert "num_beams" not in kwargs

    def test_top_p_threaded_when_sampling(self) -> None:
        kwargs = build_generate_kwargs(temperature=0.5, top_p=0.9)
        assert kwargs["top_p"] == pytest.approx(0.9)

    def test_top_k_threaded_when_sampling(self) -> None:
        kwargs = build_generate_kwargs(temperature=0.5, top_k=40)
        assert kwargs["top_k"] == 40

    def test_top_p_ignored_on_deterministic_path(self) -> None:
        kwargs = build_generate_kwargs(temperature=0.0, top_p=0.9)
        assert "top_p" not in kwargs


class TestCommon:
    def test_repetition_penalty_threaded_both_paths(self) -> None:
        kwargs_det = build_generate_kwargs(temperature=0.0, repetition_penalty=1.1)
        assert kwargs_det["repetition_penalty"] == pytest.approx(1.1)
        kwargs_sample = build_generate_kwargs(temperature=0.5, repetition_penalty=1.1)
        assert kwargs_sample["repetition_penalty"] == pytest.approx(1.1)


class TestValidation:
    def test_zero_max_new_tokens_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_new_tokens"):
            build_generate_kwargs(max_new_tokens=0)

    def test_negative_max_new_tokens_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_new_tokens"):
            build_generate_kwargs(max_new_tokens=-5)

    def test_negative_temperature_rejected(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            build_generate_kwargs(temperature=-0.1)
