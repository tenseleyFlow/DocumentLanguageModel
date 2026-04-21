"""`dpo_phase.run()` end-to-end with a mocked DPOTrainer.

Mirrors `test_trainer.py`'s factory-seam pattern: we pass a MagicMock
factory so `run()` exercises preflight → lock → log → commit →
manifest → state-sidecar without importing HF/TRL or torch.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from dlm.base_models import BASE_MODELS
from dlm.doc.parser import ParsedDlm
from dlm.doc.schema import DlmFrontmatter, PreferenceConfig, TrainingConfig
from dlm.doc.sections import Section, SectionType
from dlm.store.manifest import Manifest, save_manifest
from dlm.store.paths import for_dlm
from dlm.train.preference.dpo_phase import run
from dlm.train.state_sidecar import STATE_FILENAME, STATE_SHA_FILENAME


def _parsed_with_preferences() -> ParsedDlm:
    pref_body = "### Prompt\nq?\n### Chosen\nc.\n### Rejected\nr.\n"
    return ParsedDlm(
        frontmatter=DlmFrontmatter(
            dlm_id="01KABCD" + "0" * 19,
            base_model="smollm2-135m",
            training=TrainingConfig(seed=42, preference=PreferenceConfig(enabled=True)),
        ),
        sections=(Section(type=SectionType.PREFERENCE, content=pref_body),),
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
        to_dict=lambda: {"precision": "bf16", "phase": "dpo"},
    )


def _mock_factory(**_: Any) -> MagicMock:
    dpo = MagicMock()
    dpo.state = SimpleNamespace(global_step=15, epoch=1.0, best_metric=None)
    dpo.optimizer = SimpleNamespace(state_dict=lambda: {"lr": 5e-6})
    dpo.lr_scheduler = SimpleNamespace(state_dict=lambda: {"step": 15})
    dpo.scaler = None
    dpo.control = SimpleNamespace(should_training_stop=False)

    dpo.train.return_value = SimpleNamespace(training_loss=0.42)

    def _save_model(path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_config.json").write_text("{}")
        (p / "adapter_model.safetensors").write_bytes(b"\x00" * 64)

    dpo.save_model.side_effect = _save_model
    return dpo


def _seed_prior_sft(store, dlm_id: str = "01DPOTEST") -> None:  # type: ignore[no-untyped-def]
    """Prime the store with a plausible post-SFT state.

    `allocate_next_version` picks the next vNNNN by scanning on-disk
    dirs — not by reading the manifest — so we materialize a v0001
    placeholder. The manifest entry keeps the schema side consistent.
    """
    store.ensure_layout()
    save_manifest(
        store.manifest,
        Manifest(dlm_id=dlm_id, base_model="smollm2-135m", adapter_version=1),
    )
    v0001 = store.adapter_version(1)
    v0001.mkdir(parents=True, exist_ok=True)
    (v0001 / "adapter_config.json").write_text("{}")


class TestRunHappyPath:
    def test_commits_next_adapter_version(self, tmp_path: Path) -> None:
        store = for_dlm("01DPOTEST", home=tmp_path)
        _seed_prior_sft(store)

        spec = BASE_MODELS["smollm2-135m"]
        result = run(
            store,
            _parsed_with_preferences(),
            spec,
            _plan(),
            reference_adapter_version=1,
            trainer_factory=_mock_factory,
        )

        # DPO writes the next adapter version on top of SFT's v0001.
        assert result.adapter_version == 2
        assert result.adapter_path.name == "v0002"
        assert (result.adapter_path / "adapter_config.json").exists()
        assert (result.adapter_path / STATE_FILENAME).exists()
        assert (result.adapter_path / STATE_SHA_FILENAME).exists()

    def test_manifest_gets_new_training_run_entry(self, tmp_path: Path) -> None:
        store = for_dlm("01DPOTEST", home=tmp_path)
        _seed_prior_sft(store)

        spec = BASE_MODELS["smollm2-135m"]
        run(
            store,
            _parsed_with_preferences(),
            spec,
            _plan(),
            reference_adapter_version=1,
            trainer_factory=_mock_factory,
        )

        from dlm.store.manifest import load_manifest

        manifest = load_manifest(store.manifest)
        assert manifest.adapter_version == 2
        assert len(manifest.training_runs) == 1
        assert manifest.training_runs[0].adapter_version == 2

    def test_result_carries_training_loss_from_mock(self, tmp_path: Path) -> None:
        store = for_dlm("01DPOTEST", home=tmp_path)
        _seed_prior_sft(store)

        spec = BASE_MODELS["smollm2-135m"]
        result = run(
            store,
            _parsed_with_preferences(),
            spec,
            _plan(),
            reference_adapter_version=1,
            trainer_factory=_mock_factory,
        )
        assert result.final_train_loss == 0.42
        # DPO phase doesn't wire eval — val metrics stay None.
        assert result.final_val_loss is None
        assert result.final_val_perplexity is None
        assert result.early_stopped is False

    def test_seed_defaults_to_training_config(self, tmp_path: Path) -> None:
        store = for_dlm("01DPOTEST", home=tmp_path)
        _seed_prior_sft(store)

        spec = BASE_MODELS["smollm2-135m"]
        result = run(
            store,
            _parsed_with_preferences(),
            spec,
            _plan(),
            reference_adapter_version=1,
            trainer_factory=_mock_factory,
        )
        assert result.seed == 42  # matches TrainingConfig(seed=42)


class TestRunSteps:
    def test_factory_receives_reference_adapter_version(self, tmp_path: Path) -> None:
        """The factory call should see the reference_adapter_version
        we passed into `run()`."""
        captured: dict[str, Any] = {}

        def _capturing_factory(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return _mock_factory(**kwargs)

        store = for_dlm("01DPOTEST3", home=tmp_path)
        store.ensure_layout()
        save_manifest(
            store.manifest,
            Manifest(dlm_id="01DPOTEST3", base_model="smollm2-135m", adapter_version=3),
        )
        # Seed adapter version dirs v0001..v0003 so allocate_next picks v0004.
        for n in (1, 2, 3):
            vn = store.adapter_version(n)
            vn.mkdir(parents=True, exist_ok=True)
            (vn / "adapter_config.json").write_text("{}")

        spec = BASE_MODELS["smollm2-135m"]
        run(
            store,
            _parsed_with_preferences(),
            spec,
            _plan(),
            reference_adapter_version=3,
            trainer_factory=_capturing_factory,
        )
        assert captured["reference_adapter_version"] == 3
