"""`orpo_phase.run()` end-to-end with a mocked ORPOTrainer.

Mirrors `test_dpo_phase.py` — we pass a MagicMock factory so `run()`
exercises preflight → lock → log → commit → manifest → state-sidecar
without importing HF/TRL or torch. Sprint 18 shipped ORPO without this
test; audit-07 B3 closes the 0% coverage gap.
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
from dlm.train.preference.orpo_phase import run
from dlm.train.state_sidecar import STATE_FILENAME, STATE_SHA_FILENAME


def _parsed_with_preferences() -> ParsedDlm:
    pref_body = "### Prompt\nq?\n### Chosen\nc.\n### Rejected\nr.\n"
    return ParsedDlm(
        frontmatter=DlmFrontmatter(
            dlm_id="01KABCD" + "0" * 19,
            base_model="smollm2-135m",
            training=TrainingConfig(
                seed=42,
                preference=PreferenceConfig(enabled=True, method="orpo"),
            ),
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
        to_dict=lambda: {"precision": "bf16", "phase": "orpo"},
    )


def _mock_factory(**_: Any) -> MagicMock:
    orpo = MagicMock()
    orpo.state = SimpleNamespace(global_step=12, epoch=1.0, best_metric=None)
    orpo.optimizer = SimpleNamespace(state_dict=lambda: {"lr": 5e-6})
    orpo.lr_scheduler = SimpleNamespace(state_dict=lambda: {"step": 12})
    orpo.scaler = None
    orpo.control = SimpleNamespace(should_training_stop=False)
    orpo.train.return_value = SimpleNamespace(training_loss=0.31)

    def _save_model(path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_config.json").write_text("{}")
        (p / "adapter_model.safetensors").write_bytes(b"\x00" * 64)

    orpo.save_model.side_effect = _save_model
    return orpo


def _seed_prior_sft(store, dlm_id: str = "01ORPOTEST") -> None:  # type: ignore[no-untyped-def]
    """Prime the store with a plausible post-SFT state.

    `allocate_next_version` scans on-disk dirs, so the prior adapter
    version must exist physically — not just on the manifest.
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
        store = for_dlm("01ORPOTEST", home=tmp_path)
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

        assert result.adapter_version == 2
        assert result.adapter_path.name == "v0002"
        assert (result.adapter_path / "adapter_config.json").exists()
        assert (result.adapter_path / STATE_FILENAME).exists()
        assert (result.adapter_path / STATE_SHA_FILENAME).exists()

    def test_manifest_gets_new_training_run_entry(self, tmp_path: Path) -> None:
        store = for_dlm("01ORPOTEST", home=tmp_path)
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
        store = for_dlm("01ORPOTEST", home=tmp_path)
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
        assert result.final_train_loss == 0.31
        # ORPO phase doesn't wire eval — val metrics stay None.
        assert result.final_val_loss is None
        assert result.final_val_perplexity is None
        assert result.early_stopped is False

    def test_seed_defaults_to_training_config(self, tmp_path: Path) -> None:
        store = for_dlm("01ORPOTEST", home=tmp_path)
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
        assert result.seed == 42


class TestRunSteps:
    def test_factory_receives_reference_adapter_version(self, tmp_path: Path) -> None:
        captured: dict[str, Any] = {}

        def _capturing_factory(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return _mock_factory(**kwargs)

        store = for_dlm("01ORPOTEST3", home=tmp_path)
        store.ensure_layout()
        save_manifest(
            store.manifest,
            Manifest(
                dlm_id="01ORPOTEST3",
                base_model="smollm2-135m",
                adapter_version=3,
            ),
        )
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

    def test_explicit_seed_overrides_frontmatter(self, tmp_path: Path) -> None:
        store = for_dlm("01ORPOTEST", home=tmp_path)
        _seed_prior_sft(store)

        spec = BASE_MODELS["smollm2-135m"]
        result = run(
            store,
            _parsed_with_preferences(),
            spec,
            _plan(),
            reference_adapter_version=1,
            seed=7,
            trainer_factory=_mock_factory,
        )
        assert result.seed == 7

    def test_max_steps_threaded_to_factory(self, tmp_path: Path) -> None:
        captured: dict[str, Any] = {}

        def _capturing_factory(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return _mock_factory(**kwargs)

        store = for_dlm("01ORPOTEST", home=tmp_path)
        _seed_prior_sft(store)
        spec = BASE_MODELS["smollm2-135m"]
        run(
            store,
            _parsed_with_preferences(),
            spec,
            _plan(),
            reference_adapter_version=1,
            max_steps=25,
            trainer_factory=_capturing_factory,
        )
        assert captured["max_steps"] == 25


class TestLockModes:
    def test_ignore_mode_skips_lock_write(self, tmp_path: Path) -> None:
        """lock_mode='ignore' means no lock file is written."""
        store = for_dlm("01ORPOTEST", home=tmp_path)
        _seed_prior_sft(store)

        spec = BASE_MODELS["smollm2-135m"]
        run(
            store,
            _parsed_with_preferences(),
            spec,
            _plan(),
            reference_adapter_version=1,
            lock_mode="ignore",
            trainer_factory=_mock_factory,
        )
        # ignore mode: lock validation + lock write are both skipped.
        # The training run itself still lands.
        from dlm.store.manifest import load_manifest

        manifest = load_manifest(store.manifest)
        assert len(manifest.training_runs) == 1


class TestLogEvents:
    def test_log_contains_orpo_phase_start_event(self, tmp_path: Path) -> None:
        """Verify the dedicated ORPO start event lands in the JSONL log."""
        import json

        store = for_dlm("01ORPOTEST", home=tmp_path)
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

        rows = [
            json.loads(line)
            for line in result.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        types = {r.get("type") for r in rows}
        assert "orpo_phase_start" in types

    def test_log_contains_run_complete_event(self, tmp_path: Path) -> None:
        import json

        store = for_dlm("01ORPOTEST", home=tmp_path)
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

        rows = [
            json.loads(line)
            for line in result.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        run_complete = [r for r in rows if r.get("type") == "run_complete"]
        assert len(run_complete) == 1
        assert run_complete[0]["adapter_version"] == 2
        assert run_complete[0]["early_stopped"] is False
