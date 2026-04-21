"""DoRA schema acceptance + LoraConfig wiring + trainer resolver."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlm.doc.schema import AdapterConfig, TrainingConfig


class TestSchemaAcceptsDora:
    def test_flat_training_accepts_dora(self) -> None:
        tc = TrainingConfig(adapter="dora")
        assert tc.adapter == "dora"

    def test_per_adapter_config_accepts_dora(self) -> None:
        cfg = AdapterConfig(adapter="dora")
        assert cfg.adapter == "dora"

    def test_lora_and_qlora_still_accepted(self) -> None:
        assert TrainingConfig(adapter="lora").adapter == "lora"
        assert TrainingConfig(adapter="qlora").adapter == "qlora"

    def test_unknown_adapter_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(adapter="bogus")  # type: ignore[arg-type]


class TestLoraConfigWiring:
    def _spec(self):  # type: ignore[no-untyped-def]
        from dlm.base_models.resolver import resolve

        return resolve("smollm2-135m")

    def test_use_dora_false_by_default(self) -> None:
        from dlm.train.adapter import build_lora_config

        cfg = build_lora_config(
            self._spec(),
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            tokenizer_grew=False,
        )
        # Attribute name on LoraConfig (peft >= 0.8) is `use_dora`.
        assert getattr(cfg, "use_dora", False) is False

    def test_use_dora_true_when_requested(self) -> None:
        from dlm.train.adapter import build_lora_config

        cfg = build_lora_config(
            self._spec(),
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            tokenizer_grew=False,
            use_dora=True,
        )
        assert cfg.use_dora is True


class TestTrainerAdapterResolver:
    def test_flat_adapter_type(self) -> None:
        from dlm.doc.parser import parse_text
        from dlm.train.trainer import _resolve_adapter_type

        body = (
            "---\n"
            "dlm_id: 01KPDRATEST000000000000000\n"
            "base_model: smollm2-135m\n"
            "training:\n"
            "  adapter: dora\n"
            "---\n"
            "body\n"
        )
        parsed = parse_text(body)
        assert _resolve_adapter_type(parsed, None) == "dora"

    def test_multi_adapter_picks_per_name(self) -> None:
        from dlm.doc.parser import parse_text
        from dlm.train.trainer import _resolve_adapter_type

        body = (
            "---\n"
            "dlm_id: 01KPDRATEST000000000000000\n"
            "base_model: smollm2-135m\n"
            "training:\n"
            "  adapters:\n"
            "    primary:\n"
            "      adapter: dora\n"
            "    secondary:\n"
            "      adapter: lora\n"
            "---\n"
            "::instruction#primary::\n### Q\nhi?\n### A\nhi.\n"
            "\n::instruction#secondary::\n### Q\nhi?\n### A\nhi.\n"
        )
        parsed = parse_text(body)
        assert _resolve_adapter_type(parsed, "primary") == "dora"
        assert _resolve_adapter_type(parsed, "secondary") == "lora"

    def test_unknown_adapter_name_falls_back_to_flat(self) -> None:
        from dlm.doc.parser import parse_text
        from dlm.train.trainer import _resolve_adapter_type

        body = (
            "---\n"
            "dlm_id: 01KPDRATEST000000000000000\n"
            "base_model: smollm2-135m\n"
            "training:\n"
            "  adapter: qlora\n"
            "---\n"
            "body\n"
        )
        parsed = parse_text(body)
        # No adapters block, so any adapter_name falls through to flat.
        assert _resolve_adapter_type(parsed, "whatever") == "qlora"
