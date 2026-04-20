"""`run_all` orchestration over named adapters."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from dlm.base_models import BASE_MODELS
from dlm.doc.parser import ParsedDlm
from dlm.doc.schema import (
    AdapterConfig,
    DlmFrontmatter,
    TrainingConfig,
)
from dlm.doc.sections import Section, SectionType
from dlm.store.manifest import Manifest, save_manifest
from dlm.store.paths import for_dlm
from dlm.train.multi_adapter.trainer import run_all


def _plan() -> SimpleNamespace:
    return SimpleNamespace(
        precision="bf16",
        attn_implementation="sdpa",
        use_qlora=False,
        quant_compute_dtype=None,
        micro_batch_size=1,
        grad_accum=1,
        effective_batch_size=1,
        gradient_checkpointing=False,
        est_peak_vram_gb=1.0,
        est_step_seconds=0.1,
        reason="test",
        to_dict=lambda: {"precision": "bf16"},
    )


def _mock_trainer_factory(**_: Any) -> MagicMock:
    sft = MagicMock()
    sft.state = SimpleNamespace(global_step=5, epoch=1.0, best_metric=0.9)
    sft.optimizer = SimpleNamespace(state_dict=lambda: {"lr": 1e-4})
    sft.lr_scheduler = SimpleNamespace(state_dict=lambda: {"step": 5})
    sft.scaler = None
    sft.control = SimpleNamespace(should_training_stop=False)
    sft.train.return_value = SimpleNamespace(training_loss=1.0)

    def _save_model(path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_config.json").write_text("{}")
        (p / "adapter_model.safetensors").write_bytes(b"\x00" * 32)

    sft.save_model.side_effect = _save_model
    return sft


def _multi_adapter_parsed(dlm_id: str) -> ParsedDlm:
    return ParsedDlm(
        frontmatter=DlmFrontmatter(
            dlm_id=dlm_id,
            base_model="smollm2-135m",
            training=TrainingConfig(
                seed=42,
                adapters={
                    "knowledge": AdapterConfig(),
                    "tone": AdapterConfig(lora_r=4),
                },
            ),
        ),
        sections=(
            Section(type=SectionType.PROSE, content="Shared domain prose."),
            Section(
                type=SectionType.INSTRUCTION,
                content="### Q\nfacts?\n### A\nfacts.",
                adapter="knowledge",
            ),
            Section(
                type=SectionType.INSTRUCTION,
                content="### Q\ntone?\n### A\ncrisp.",
                adapter="tone",
            ),
        ),
    )


def _single_adapter_parsed(dlm_id: str) -> ParsedDlm:
    return ParsedDlm(
        frontmatter=DlmFrontmatter(
            dlm_id=dlm_id,
            base_model="smollm2-135m",
            training=TrainingConfig(seed=42),
        ),
        sections=(
            Section(type=SectionType.PROSE, content="Single-adapter prose."),
        ),
    )


def _seed_store(tmp_path: Path, dlm_id: str) -> Any:
    store = for_dlm(dlm_id, home=tmp_path)
    store.ensure_layout()
    save_manifest(
        store.manifest, Manifest(dlm_id=dlm_id, base_model="smollm2-135m")
    )
    return store


class TestSingleAdapterPassthrough:
    def test_single_adapter_doc_yields_one_result(self, tmp_path: Path) -> None:
        dlm_id = "01HZ4X7TGZM3J1A2B3C4D5E6FA"
        store = _seed_store(tmp_path, dlm_id)
        results = run_all(
            store,
            _single_adapter_parsed(dlm_id),
            BASE_MODELS["smollm2-135m"],
            _plan(),
            mode="fresh",
            trainer_factory=_mock_trainer_factory,
        )
        assert len(results) == 1
        # Flat layout: version dir lives under adapter/versions/, not a named subdir.
        assert store.adapter_version(1).is_dir()


class TestMultiAdapterOrchestration:
    def test_trains_each_declared_adapter(self, tmp_path: Path) -> None:
        dlm_id = "01HZ4X7TGZM3J1A2B3C4D5E6FB"
        store = _seed_store(tmp_path, dlm_id)
        results = run_all(
            store,
            _multi_adapter_parsed(dlm_id),
            BASE_MODELS["smollm2-135m"],
            _plan(),
            mode="fresh",
            trainer_factory=_mock_trainer_factory,
        )
        assert len(results) == 2

    def test_each_adapter_gets_own_version_dir(self, tmp_path: Path) -> None:
        dlm_id = "01HZ4X7TGZM3J1A2B3C4D5E6FB"
        store = _seed_store(tmp_path, dlm_id)
        run_all(
            store,
            _multi_adapter_parsed(dlm_id),
            BASE_MODELS["smollm2-135m"],
            _plan(),
            mode="fresh",
            trainer_factory=_mock_trainer_factory,
        )
        assert store.adapter_version_for("knowledge", 1).is_dir()
        assert store.adapter_version_for("tone", 1).is_dir()

    def test_each_adapter_gets_own_current_pointer(self, tmp_path: Path) -> None:
        dlm_id = "01HZ4X7TGZM3J1A2B3C4D5E6FB"
        store = _seed_store(tmp_path, dlm_id)
        run_all(
            store,
            _multi_adapter_parsed(dlm_id),
            BASE_MODELS["smollm2-135m"],
            _plan(),
            mode="fresh",
            trainer_factory=_mock_trainer_factory,
        )
        assert store.resolve_current_adapter_for("knowledge") == (
            store.adapter_version_for("knowledge", 1).resolve()
        )
        assert store.resolve_current_adapter_for("tone") == (
            store.adapter_version_for("tone", 1).resolve()
        )

    def test_manifest_gets_one_run_per_adapter(self, tmp_path: Path) -> None:
        dlm_id = "01HZ4X7TGZM3J1A2B3C4D5E6FB"
        store = _seed_store(tmp_path, dlm_id)
        run_all(
            store,
            _multi_adapter_parsed(dlm_id),
            BASE_MODELS["smollm2-135m"],
            _plan(),
            mode="fresh",
            trainer_factory=_mock_trainer_factory,
        )
        from dlm.store.manifest import load_manifest

        manifest = load_manifest(store.manifest)
        assert len(manifest.training_runs) == 2

    def test_declaration_order_preserved(self, tmp_path: Path) -> None:
        dlm_id = "01HZ4X7TGZM3J1A2B3C4D5E6FB"
        store = _seed_store(tmp_path, dlm_id)
        results = run_all(
            store,
            _multi_adapter_parsed(dlm_id),
            BASE_MODELS["smollm2-135m"],
            _plan(),
            mode="fresh",
            trainer_factory=_mock_trainer_factory,
        )
        # Knowledge is declared first; its run_id should be lower.
        assert results[0].run_id < results[1].run_id

    def test_manifest_adapter_versions_populated(self, tmp_path: Path) -> None:
        """Audit-07 M1: multi-adapter runs bump per-adapter version dict,
        not the flat `adapter_version`."""
        dlm_id = "01HZ4X7TGZM3J1A2B3C4D5E6FB"
        store = _seed_store(tmp_path, dlm_id)
        run_all(
            store,
            _multi_adapter_parsed(dlm_id),
            BASE_MODELS["smollm2-135m"],
            _plan(),
            mode="fresh",
            trainer_factory=_mock_trainer_factory,
        )
        from dlm.store.manifest import load_manifest

        manifest = load_manifest(store.manifest)
        assert manifest.adapter_versions == {"knowledge": 1, "tone": 1}
        # Flat field stays at 0 (untouched) for multi-adapter stores.
        assert manifest.adapter_version == 0

    def test_training_run_summaries_carry_adapter_name(
        self, tmp_path: Path
    ) -> None:
        """Audit-07 M1: each TrainingRunSummary is tagged with the name."""
        dlm_id = "01HZ4X7TGZM3J1A2B3C4D5E6FB"
        store = _seed_store(tmp_path, dlm_id)
        run_all(
            store,
            _multi_adapter_parsed(dlm_id),
            BASE_MODELS["smollm2-135m"],
            _plan(),
            mode="fresh",
            trainer_factory=_mock_trainer_factory,
        )
        from dlm.store.manifest import load_manifest

        manifest = load_manifest(store.manifest)
        names = [r.adapter_name for r in manifest.training_runs]
        assert sorted(names, key=str) == ["knowledge", "tone"]
