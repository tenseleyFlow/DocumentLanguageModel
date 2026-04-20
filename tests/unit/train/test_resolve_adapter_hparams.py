"""`_resolve_adapter_hparams` — per-adapter LoRA config threading (20c B1)."""

from __future__ import annotations

import pytest

from dlm.doc.parser import ParsedDlm
from dlm.doc.schema import AdapterConfig, DlmFrontmatter, TrainingConfig
from dlm.doc.sections import Section, SectionType
from dlm.train.trainer import _resolve_adapter_hparams


def _parsed(training: TrainingConfig) -> ParsedDlm:
    return ParsedDlm(
        frontmatter=DlmFrontmatter(
            dlm_id="01HZ4X7TGZM3J1A2B3C4D5E6F7",
            base_model="smollm2-135m",
            training=training,
        ),
        sections=(Section(type=SectionType.PROSE, content="hi"),),
    )


class TestFlatConfig:
    def test_returns_flat_fields_when_adapter_name_is_none(self) -> None:
        training = TrainingConfig(
            lora_r=16, lora_alpha=32, lora_dropout=0.1, learning_rate=1e-3
        )
        r, alpha, dropout, lr = _resolve_adapter_hparams(_parsed(training), None)
        assert (r, alpha, dropout) == (16, 32, pytest.approx(0.1))
        assert lr == pytest.approx(1e-3)

    def test_returns_flat_fields_when_no_adapters_block(self) -> None:
        # adapter_name set but doc is single-adapter — falls back to flat.
        training = TrainingConfig(lora_r=16)
        r, _a, _d, _lr = _resolve_adapter_hparams(_parsed(training), "ghost")
        assert r == 16


class TestMultiAdapterConfig:
    def test_returns_per_adapter_values(self) -> None:
        training = TrainingConfig.model_validate(
            {
                "adapters": {
                    "knowledge": {"lora_r": 8, "lora_alpha": 16},
                    "tone": {
                        "lora_r": 4,
                        "lora_alpha": 8,
                        "lora_dropout": 0.2,
                        "learning_rate": 1e-4,
                    },
                }
            }
        )
        k_r, k_alpha, _k_drop, _k_lr = _resolve_adapter_hparams(
            _parsed(training), "knowledge"
        )
        t_r, t_alpha, t_drop, t_lr = _resolve_adapter_hparams(
            _parsed(training), "tone"
        )
        assert (k_r, k_alpha) == (8, 16)
        assert (t_r, t_alpha) == (4, 8)
        assert t_drop == pytest.approx(0.2)
        assert t_lr == pytest.approx(1e-4)

    def test_unknown_adapter_name_falls_back_to_flat(self) -> None:
        training = TrainingConfig.model_validate(
            {"adapters": {"knowledge": {"lora_r": 8}}}
        )
        # ghost isn't declared; we fall back to defaults rather than crash.
        r, _, _, _ = _resolve_adapter_hparams(_parsed(training), "ghost")
        assert r == 8  # flat default

    def test_per_adapter_defaults_when_not_overridden(self) -> None:
        training = TrainingConfig.model_validate(
            {"adapters": {"default_one": {}}}
        )
        r, alpha, dropout, lr = _resolve_adapter_hparams(
            _parsed(training), "default_one"
        )
        # AdapterConfig() defaults: r=8, alpha=16, dropout=0.05, lr=2e-4
        assert (r, alpha) == (8, 16)
        assert dropout == pytest.approx(0.05)
        assert lr == pytest.approx(2e-4)
