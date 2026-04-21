"""Unit tests for embed_warmup.

Uses fake model objects — we're testing requires_grad flipping and
the modules_to_save extension logic, not PyTorch.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from dlm.train.cpt.embed_warmup import (
    EmbedWarmupCallback,
    extend_modules_to_save_for_embed_warmup,
    unfreeze_embeddings_for,
)


class _FakeParam:
    def __init__(self, requires_grad: bool = False) -> None:
        self.requires_grad = requires_grad


def _model(*, embed_frozen: bool = True, head_frozen: bool = True, tied: bool = False) -> Any:
    embed_param = _FakeParam(requires_grad=not embed_frozen)
    head_param = embed_param if tied else _FakeParam(requires_grad=not head_frozen)
    embed_module = SimpleNamespace(weight=embed_param)
    head_module = SimpleNamespace(weight=head_param)
    return SimpleNamespace(
        get_input_embeddings=lambda: embed_module,
        get_output_embeddings=lambda: head_module,
    )


class TestUnfreezeContextManager:
    def test_unfreezes_both_embeddings(self) -> None:
        model = _model(embed_frozen=True, head_frozen=True)
        with unfreeze_embeddings_for(model) as weights:
            assert all(w.requires_grad for w in weights)
            assert len(weights) == 2

    def test_restores_originals_on_exit(self) -> None:
        model = _model(embed_frozen=True, head_frozen=True)
        with unfreeze_embeddings_for(model):
            pass
        assert model.get_input_embeddings().weight.requires_grad is False
        assert model.get_output_embeddings().weight.requires_grad is False

    def test_preserves_per_weight_original_state(self) -> None:
        model = _model(embed_frozen=True, head_frozen=False)
        with unfreeze_embeddings_for(model):
            pass
        assert model.get_input_embeddings().weight.requires_grad is False
        assert model.get_output_embeddings().weight.requires_grad is True

    def test_restores_on_exception(self) -> None:
        model = _model(embed_frozen=True, head_frozen=True)
        with pytest.raises(RuntimeError, match="boom"):
            with unfreeze_embeddings_for(model):
                raise RuntimeError("boom")
        assert model.get_input_embeddings().weight.requires_grad is False
        assert model.get_output_embeddings().weight.requires_grad is False

    def test_tied_weights_deduplicated(self) -> None:
        model = _model(tied=True)
        with unfreeze_embeddings_for(model) as weights:
            assert len(weights) == 1  # same object once
            assert weights[0].requires_grad is True


class TestExtendModulesToSave:
    def test_zero_warmup_passes_through_none(self) -> None:
        assert extend_modules_to_save_for_embed_warmup(None, embed_warmup_steps=0) is None

    def test_zero_warmup_passes_through_list(self) -> None:
        assert extend_modules_to_save_for_embed_warmup(["embed_tokens"], embed_warmup_steps=0) == [
            "embed_tokens"
        ]

    def test_warmup_on_with_no_existing(self) -> None:
        out = extend_modules_to_save_for_embed_warmup(None, embed_warmup_steps=50)
        assert out == ["embed_tokens", "lm_head"]

    def test_warmup_on_with_existing_tokenizer_grew(self) -> None:
        # tokenizer_grew already added both — result is stable (no duplicates).
        out = extend_modules_to_save_for_embed_warmup(
            ["embed_tokens", "lm_head"], embed_warmup_steps=50
        )
        assert out == ["embed_tokens", "lm_head"]

    def test_warmup_on_preserves_order_of_existing(self) -> None:
        out = extend_modules_to_save_for_embed_warmup(
            ["my_module", "embed_tokens"], embed_warmup_steps=1
        )
        assert out == ["my_module", "embed_tokens", "lm_head"]


class TestEmbedWarmupCallback:
    def test_rejects_negative_n_steps(self) -> None:
        with pytest.raises(ValueError, match="n_steps must be non-negative"):
            EmbedWarmupCallback(_model(), n_steps=-1)

    def test_on_train_begin_unfreezes_when_n_positive(self) -> None:
        model = _model(embed_frozen=True, head_frozen=True)
        cb = EmbedWarmupCallback(model, n_steps=10)
        cb.on_train_begin(args=None, state=None, control=None)
        assert model.get_input_embeddings().weight.requires_grad is True
        assert model.get_output_embeddings().weight.requires_grad is True

    def test_on_train_begin_is_noop_when_n_zero(self) -> None:
        model = _model(embed_frozen=True, head_frozen=True)
        cb = EmbedWarmupCallback(model, n_steps=0)
        cb.on_train_begin(args=None, state=None, control=None)
        assert model.get_input_embeddings().weight.requires_grad is False

    def test_step_end_at_budget_restores(self) -> None:
        model = _model(embed_frozen=True, head_frozen=True)
        cb = EmbedWarmupCallback(model, n_steps=5)
        cb.on_train_begin(args=None, state=None, control=None)
        # Steps before budget: still unfrozen.
        cb.on_step_end(args=None, state=SimpleNamespace(global_step=3), control=None)
        assert model.get_input_embeddings().weight.requires_grad is True
        # Step == budget: refreeze.
        cb.on_step_end(args=None, state=SimpleNamespace(global_step=5), control=None)
        assert model.get_input_embeddings().weight.requires_grad is False

    def test_train_end_restores_if_still_active(self) -> None:
        model = _model(embed_frozen=True, head_frozen=True)
        cb = EmbedWarmupCallback(model, n_steps=1000)  # will never fire
        cb.on_train_begin(args=None, state=None, control=None)
        cb.on_train_end(args=None, state=None, control=None)
        assert model.get_input_embeddings().weight.requires_grad is False

    def test_restore_is_idempotent(self) -> None:
        model = _model(embed_frozen=True, head_frozen=True)
        cb = EmbedWarmupCallback(model, n_steps=5)
        cb.on_train_begin(args=None, state=None, control=None)
        cb.on_step_end(args=None, state=SimpleNamespace(global_step=10), control=None)
        # Second restore via on_train_end: does not double-flip.
        cb.on_train_end(args=None, state=None, control=None)
        assert model.get_input_embeddings().weight.requires_grad is False
