"""Schema v8: `training.gate` block + v7 → v8 identity migration."""

from __future__ import annotations

import pytest

from dlm.doc.parser import parse_text
from dlm.doc.schema import (
    CURRENT_SCHEMA_VERSION,
    AdapterConfig,
    DlmFrontmatter,
    GateConfig,
    TrainingConfig,
)


class TestGateConfigDefaults:
    def test_disabled_by_default(self) -> None:
        cfg = GateConfig()
        assert cfg.enabled is False
        assert cfg.hidden_proj_dim == 64
        assert cfg.steps == 200
        assert cfg.lr == pytest.approx(3e-4)
        assert cfg.cold_start_floor == 4
        assert cfg.entropy_lambda == pytest.approx(0.01)

    def test_trainingconfig_has_gate_default(self) -> None:
        t = TrainingConfig()
        assert isinstance(t.gate, GateConfig)
        assert t.gate.enabled is False

    def test_bounds_rejected(self) -> None:
        with pytest.raises(ValueError, match="hidden_proj_dim"):
            GateConfig(hidden_proj_dim=0)
        with pytest.raises(ValueError, match="steps"):
            GateConfig(steps=0)
        with pytest.raises(ValueError, match="lr"):
            GateConfig(lr=0.0)
        with pytest.raises(ValueError, match="cold_start_floor"):
            GateConfig(cold_start_floor=0)


class TestGateRequiresMultipleAdapters:
    def test_enabled_without_adapters_refused(self) -> None:
        with pytest.raises(ValueError, match="two or more named adapters"):
            TrainingConfig(gate=GateConfig(enabled=True))

    def test_enabled_with_single_adapter_refused(self) -> None:
        with pytest.raises(ValueError, match="two or more named adapters"):
            TrainingConfig(
                adapters={"only": AdapterConfig()},
                gate=GateConfig(enabled=True),
            )

    def test_enabled_with_two_adapters_accepted(self) -> None:
        cfg = TrainingConfig(
            adapters={"a": AdapterConfig(), "b": AdapterConfig()},
            gate=GateConfig(enabled=True),
        )
        assert cfg.gate.enabled is True

    def test_disabled_gate_ok_without_adapters(self) -> None:
        """Default (disabled) gate must stay compatible with single-adapter
        documents — the cross-field rule applies only when `enabled`."""
        cfg = TrainingConfig()
        assert cfg.gate.enabled is False


class TestV7ToV8IdentityMigration:
    def test_current_schema_is_v8(self) -> None:
        assert CURRENT_SCHEMA_VERSION == 8

    def test_v7_document_parses_under_v8(self) -> None:
        body = (
            "---\n"
            "dlm_id: 01KPQ8GATETEST000000000000\n"
            "dlm_version: 7\n"
            "base_model: smollm2-135m\n"
            "---\n"
            "::instruction::\n### Q\nhi?\n### A\nhello.\n"
        )
        parsed = parse_text(body)
        # Migrator upgrades the version to CURRENT; gate picks up defaults.
        assert parsed.frontmatter.dlm_version == 8
        assert parsed.frontmatter.training.gate.enabled is False

    def test_v8_document_with_gate_block_parses(self) -> None:
        body = (
            "---\n"
            "dlm_id: 01KPQ8GATETEST000000000000\n"
            "dlm_version: 8\n"
            "base_model: smollm2-135m\n"
            "training:\n"
            "  adapters:\n"
            "    lexer: {}\n"
            "    runtime: {}\n"
            "  gate:\n"
            "    enabled: true\n"
            "    steps: 150\n"
            "---\n"
        )
        parsed = parse_text(body)
        fm = parsed.frontmatter
        assert isinstance(fm, DlmFrontmatter)
        assert fm.training.gate.enabled is True
        assert fm.training.gate.steps == 150
