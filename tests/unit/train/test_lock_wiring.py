"""Sprint 15 — `trainer.run()` lock validation + persistence wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from dlm.base_models import BASE_MODELS
from dlm.doc.parser import ParsedDlm
from dlm.doc.schema import DlmFrontmatter, TrainingConfig
from dlm.doc.sections import Section, SectionType
from dlm.lock import DlmLock, LockValidationError, load_lock, write_lock
from dlm.lock.schema import CURRENT_LOCK_VERSION
from dlm.store.manifest import Manifest, save_manifest
from dlm.store.paths import for_dlm
from dlm.train.trainer import run

# Cheap parsed helper duplicated from test_trainer; kept local so a future
# refactor in one test file doesn't ripple into the other.


def _parsed(tmp_path: Path, dlm_id: str = "01TEST0" + "0" * 19) -> ParsedDlm:
    doc = tmp_path / "doc.dlm"
    doc.write_text("placeholder .dlm body\n", encoding="utf-8")
    return ParsedDlm(
        frontmatter=DlmFrontmatter(
            dlm_id=dlm_id,
            base_model="smollm2-135m",
            training=TrainingConfig(seed=42),
        ),
        sections=(Section(type=SectionType.PROSE, content="x"),),
        source_path=doc,
    )


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
    sft.state = SimpleNamespace(global_step=10, epoch=1.0, best_metric=0.5)
    sft.optimizer = SimpleNamespace(state_dict=lambda: {})
    sft.lr_scheduler = SimpleNamespace(state_dict=lambda: {})
    sft.scaler = None
    sft.control = SimpleNamespace(should_training_stop=False)
    sft.train.return_value = SimpleNamespace(training_loss=0.5)

    def _save_model(path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_config.json").write_text("{}")
        (p / "adapter_model.safetensors").write_bytes(b"\x00" * 32)

    sft.save_model.side_effect = _save_model
    return sft


def _bootstrap_store(tmp_path: Path, dlm_id: str = "01TEST0" + "0" * 19):
    store = for_dlm(dlm_id, home=tmp_path)
    store.ensure_layout()
    save_manifest(store.manifest, Manifest(dlm_id=dlm_id, base_model="smollm2-135m"))
    return store


class TestFirstRunWritesLock:
    def test_fresh_run_creates_dlm_lock_with_run_id_1(self, tmp_path: Path) -> None:
        store = _bootstrap_store(tmp_path)
        parsed = _parsed(tmp_path)
        spec = BASE_MODELS["smollm2-135m"]

        run(store, parsed, spec, _plan(), trainer_factory=_mock_trainer_factory)

        loaded = load_lock(store.root)
        assert loaded is not None
        assert loaded.lock_version == CURRENT_LOCK_VERSION
        assert loaded.dlm_id == parsed.frontmatter.dlm_id
        assert loaded.seed == 42
        assert loaded.last_run_id == 1


class TestIgnoreModeSkipsLock:
    def test_ignore_mode_doesnt_write_lock(self, tmp_path: Path) -> None:
        store = _bootstrap_store(tmp_path)
        parsed = _parsed(tmp_path)
        spec = BASE_MODELS["smollm2-135m"]

        run(
            store,
            parsed,
            spec,
            _plan(),
            trainer_factory=_mock_trainer_factory,
            lock_mode="ignore",
        )
        assert load_lock(store.root) is None


class TestErrorSeverityAborts:
    def test_base_revision_drift_raises(self, tmp_path: Path) -> None:
        store = _bootstrap_store(tmp_path)
        parsed = _parsed(tmp_path)
        spec = BASE_MODELS["smollm2-135m"]

        # Seed the store with a lock whose recorded base_model_revision
        # differs from the real spec.revision — the validator must abort.
        from datetime import UTC, datetime

        forged = DlmLock(
            lock_version=CURRENT_LOCK_VERSION,
            created_at=datetime(2026, 4, 1, tzinfo=UTC),
            dlm_id=parsed.frontmatter.dlm_id,
            dlm_sha256="0" * 64,
            base_model_revision="totally-different-revision",
            hardware_tier="cpu",
            seed=42,
            determinism_class="best-effort",
            last_run_id=1,
        )
        write_lock(store.root, forged)

        with pytest.raises(LockValidationError, match="base_model_revision"):
            run(
                store,
                parsed,
                spec,
                _plan(),
                trainer_factory=_mock_trainer_factory,
            )


class TestUpdateModeOverrides:
    def test_update_mode_bypasses_validation_and_writes(self, tmp_path: Path) -> None:
        store = _bootstrap_store(tmp_path)
        parsed = _parsed(tmp_path)
        spec = BASE_MODELS["smollm2-135m"]

        from datetime import UTC, datetime

        forged = DlmLock(
            lock_version=CURRENT_LOCK_VERSION,
            created_at=datetime(2026, 4, 1, tzinfo=UTC),
            dlm_id=parsed.frontmatter.dlm_id,
            dlm_sha256="0" * 64,
            base_model_revision="totally-different-revision",
            hardware_tier="cpu",
            seed=42,
            determinism_class="best-effort",
            last_run_id=1,
        )
        write_lock(store.root, forged)

        # --update-lock should NOT raise despite the base-revision drift.
        run(
            store,
            parsed,
            spec,
            _plan(),
            trainer_factory=_mock_trainer_factory,
            lock_mode="update",
        )

        updated = load_lock(store.root)
        assert updated is not None
        assert updated.base_model_revision == spec.revision
