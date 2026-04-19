"""`trainer.run()` end-to-end with a mocked SFTTrainer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from dlm.base_models import BASE_MODELS
from dlm.doc.parser import ParsedDlm
from dlm.doc.schema import DlmFrontmatter, TrainingConfig
from dlm.doc.sections import Section, SectionType
from dlm.store.manifest import Manifest, save_manifest
from dlm.store.paths import for_dlm
from dlm.train.state_sidecar import STATE_FILENAME, STATE_SHA_FILENAME
from dlm.train.trainer import run


def _parsed() -> ParsedDlm:
    return ParsedDlm(
        frontmatter=DlmFrontmatter(
            dlm_id="01KABCD" + "0" * 19,  # 26 chars total
            base_model="smollm2-135m",
            training=TrainingConfig(seed=42),
        ),
        sections=(
            Section(type=SectionType.PROSE, content="Sample prose for training."),
            Section(type=SectionType.PROSE, content="Another sample."),
        ),
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
    """Build a mock that looks like a real SFTTrainer to the orchestrator."""
    sft = MagicMock()
    sft.state = SimpleNamespace(global_step=20, epoch=1.0, best_metric=0.87)
    sft.optimizer = SimpleNamespace(state_dict=lambda: {"lr": 1e-4})
    sft.lr_scheduler = SimpleNamespace(state_dict=lambda: {"step": 20})
    sft.scaler = None
    # Audit-05 M2: explicit control stub so `_hf_early_stop_flag` reads
    # False rather than a truthy MagicMock auto-attr.
    sft.control = SimpleNamespace(should_training_stop=False)

    train_result = SimpleNamespace(training_loss=1.23)
    sft.train.return_value = train_result

    def _save_model(path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        # Write placeholder adapter files so the checkpoint commit
        # has real bytes to flush.
        (p / "adapter_config.json").write_text("{}")
        (p / "adapter_model.safetensors").write_bytes(b"\x00" * 64)

    sft.save_model.side_effect = _save_model
    return sft


class TestRunHappyPath:
    def test_fresh_run_commits_adapter_and_appends_manifest(self, tmp_path: Path) -> None:
        store = for_dlm("01TEST", home=tmp_path)
        store.ensure_layout()
        save_manifest(store.manifest, Manifest(dlm_id="01TEST", base_model="smollm2-135m"))

        spec = BASE_MODELS["smollm2-135m"]
        result = run(
            store,
            _parsed(),
            spec,
            _plan(),
            mode="fresh",
            trainer_factory=_mock_trainer_factory,
        )

        # Adapter committed at v0001.
        assert result.adapter_version == 1
        assert result.adapter_path.name == "v0001"
        assert (result.adapter_path / "adapter_config.json").exists()
        # Training state + sha sidecar persisted.
        assert (result.adapter_path / STATE_FILENAME).exists()
        assert (result.adapter_path / STATE_SHA_FILENAME).exists()

        # Current pointer flipped.
        assert store.resolve_current_adapter() == result.adapter_path

        # Manifest has one run appended.
        from dlm.store.manifest import load_manifest

        manifest = load_manifest(store.manifest)
        assert len(manifest.training_runs) == 1
        run_summary = manifest.training_runs[0]
        assert run_summary.run_id == 1
        assert run_summary.adapter_version == 1
        assert run_summary.seed == 42
        assert run_summary.steps == 20
        assert run_summary.final_train_loss == 1.23
        assert run_summary.status == "completed"

        # Log file written with banner + complete event.
        assert result.log_path.exists()
        lines = result.log_path.read_text().strip().splitlines()
        parsed = [json.loads(line) for line in lines]
        assert parsed[0]["type"] == "banner"
        assert any(p.get("type") == "run_complete" for p in parsed)

    def test_second_run_gets_run_id_two(self, tmp_path: Path) -> None:
        store = for_dlm("01TEST", home=tmp_path)
        store.ensure_layout()
        save_manifest(store.manifest, Manifest(dlm_id="01TEST", base_model="smollm2-135m"))
        spec = BASE_MODELS["smollm2-135m"]

        r1 = run(store, _parsed(), spec, _plan(), trainer_factory=_mock_trainer_factory)
        r2 = run(store, _parsed(), spec, _plan(), trainer_factory=_mock_trainer_factory)
        assert r1.run_id == 1
        assert r2.run_id == 2
        assert r1.adapter_version == 1
        assert r2.adapter_version == 2

    def test_seed_defaults_from_frontmatter(self, tmp_path: Path) -> None:
        store = for_dlm("01TEST", home=tmp_path)
        store.ensure_layout()
        save_manifest(store.manifest, Manifest(dlm_id="01TEST", base_model="smollm2-135m"))
        spec = BASE_MODELS["smollm2-135m"]

        result = run(
            store,
            _parsed(),
            spec,
            _plan(),
            trainer_factory=_mock_trainer_factory,
        )
        assert result.seed == 42  # from TrainingConfig default

    def test_content_hashes_updated_with_current_sections(self, tmp_path: Path) -> None:
        """Audit-04 M2: manifest.content_hashes reflects the CURRENT .dlm after run."""
        store = for_dlm("01TEST", home=tmp_path)
        store.ensure_layout()
        save_manifest(store.manifest, Manifest(dlm_id="01TEST", base_model="smollm2-135m"))
        spec = BASE_MODELS["smollm2-135m"]

        parsed = _parsed()
        expected_sids = {s.section_id for s in parsed.sections}

        run(store, parsed, spec, _plan(), trainer_factory=_mock_trainer_factory)

        from dlm.store.manifest import load_manifest

        manifest = load_manifest(store.manifest)
        assert set(manifest.content_hashes.keys()) == expected_sids

    def test_replay_corpus_appended_with_new_sections(self, tmp_path: Path) -> None:
        """Audit-04 M1: new sections are written into the replay corpus."""
        from dlm.replay import ReplayStore

        store = for_dlm("01TEST", home=tmp_path)
        store.ensure_layout()
        save_manifest(store.manifest, Manifest(dlm_id="01TEST", base_model="smollm2-135m"))
        spec = BASE_MODELS["smollm2-135m"]

        parsed = _parsed()
        run(store, parsed, spec, _plan(), trainer_factory=_mock_trainer_factory)

        replay = ReplayStore.at(store.replay_corpus, store.replay_index)
        entries = replay.load()
        # First run: every section is `new` → every section ends up in the corpus.
        corpus_sids = {e.section_id for e in entries}
        expected_sids = {s.section_id for s in parsed.sections}
        assert corpus_sids == expected_sids

    def test_second_run_reclassifies_unchanged_and_does_not_duplicate(self, tmp_path: Path) -> None:
        """After run #1 writes content_hashes, run #2 sees everything as unchanged."""
        from dlm.replay import ReplayStore

        store = for_dlm("01TEST", home=tmp_path)
        store.ensure_layout()
        save_manifest(store.manifest, Manifest(dlm_id="01TEST", base_model="smollm2-135m"))
        spec = BASE_MODELS["smollm2-135m"]

        parsed = _parsed()
        run(store, parsed, spec, _plan(), trainer_factory=_mock_trainer_factory)
        run(store, parsed, spec, _plan(), trainer_factory=_mock_trainer_factory)

        replay = ReplayStore.at(store.replay_corpus, store.replay_index)
        entries = replay.load()
        # Section count, not frame count — identical sections shouldn't re-append.
        assert len({e.section_id for e in entries}) == len(parsed.sections)
        # Second run's delta should be all-unchanged → no new frames appended.
        assert len(entries) == len(parsed.sections)

    def test_manifest_run_summary_links_to_summary_json(self, tmp_path: Path) -> None:
        """Audit-05 M3: TrainingRunSummary.summary_path resolves to the JSON file."""
        from dlm.eval import load_summary
        from dlm.store.manifest import load_manifest

        store = for_dlm("01TEST", home=tmp_path)
        store.ensure_layout()
        save_manifest(store.manifest, Manifest(dlm_id="01TEST", base_model="smollm2-135m"))
        spec = BASE_MODELS["smollm2-135m"]

        result = run(store, _parsed(), spec, _plan(), trainer_factory=_mock_trainer_factory)
        manifest = load_manifest(store.manifest)
        run_summary = manifest.training_runs[0]

        # summary_path populated + resolves to the real file.
        assert run_summary.summary_path is not None
        resolved = store.root / run_summary.summary_path
        assert resolved.exists()
        # Loaded summary matches the trainer's written values.
        loaded = load_summary(resolved)
        assert loaded.run_id == result.run_id
        assert loaded.adapter_version == result.adapter_version

    def test_training_summary_json_written(self, tmp_path: Path) -> None:
        """Sprint 10: every run writes `logs/train-*.summary.json`."""
        from dlm.eval import load_summary

        store = for_dlm("01TEST", home=tmp_path)
        store.ensure_layout()
        save_manifest(store.manifest, Manifest(dlm_id="01TEST", base_model="smollm2-135m"))
        spec = BASE_MODELS["smollm2-135m"]

        result = run(
            store,
            _parsed(),
            spec,
            _plan(),
            trainer_factory=_mock_trainer_factory,
        )

        assert result.summary_path.exists()
        summary = load_summary(result.summary_path)
        assert summary.run_id == result.run_id
        assert summary.adapter_version == result.adapter_version
        assert summary.seed == result.seed
        assert summary.steps == result.steps
        assert summary.early_stopped is False  # mock runs full schedule
        assert summary.determinism_class == result.determinism.class_

    def test_resume_mode_sees_prior_adapter(self, tmp_path: Path) -> None:
        """Audit-04 m12: mode='resume' propagates + prior adapter is resolvable."""
        store = for_dlm("01TEST", home=tmp_path)
        store.ensure_layout()
        save_manifest(store.manifest, Manifest(dlm_id="01TEST", base_model="smollm2-135m"))
        spec = BASE_MODELS["smollm2-135m"]

        # First run lays down v0001 and flips the current pointer.
        r1 = run(
            store,
            _parsed(),
            spec,
            _plan(),
            mode="fresh",
            trainer_factory=_mock_trainer_factory,
        )
        assert store.resolve_current_adapter() == r1.adapter_path

        # Second run in resume mode: the factory should observe mode='resume'
        # AND a resolvable prior-adapter path while it's inside the call.
        observed: dict[str, object] = {}

        def _capturing_factory(**kwargs: Any) -> MagicMock:
            observed["mode"] = kwargs["mode"]
            observed["resume_path"] = kwargs["store"].resolve_current_adapter()
            return _mock_trainer_factory(**kwargs)

        r2 = run(
            store,
            _parsed(),
            spec,
            _plan(),
            mode="resume",
            trainer_factory=_capturing_factory,
        )
        assert observed["mode"] == "resume"
        assert observed["resume_path"] == r1.adapter_path
        # And the new run commits a fresh version on top.
        assert r2.adapter_version == 2


class TestEvalCadence:
    """Audit-05 M2: eval_steps + early-stop config are threaded into SFTConfig."""

    def test_default_eval_steps_with_max_steps(self) -> None:
        from dlm.train.trainer import _default_eval_steps

        assert _default_eval_steps(max_steps=100) == 25  # quarter cadence
        assert _default_eval_steps(max_steps=3) == 1  # floor at 1

    def test_default_eval_steps_without_max_steps(self) -> None:
        from dlm.train.trainer import _default_eval_steps

        assert _default_eval_steps(max_steps=None) == 50
        assert _default_eval_steps(max_steps=0) == 50
        assert _default_eval_steps(max_steps=-1) == 50

    def test_default_early_stop_config(self) -> None:
        from dlm.train.trainer import _default_early_stop_config

        cfg = _default_early_stop_config()
        assert cfg.patience == 3
        assert cfg.threshold == 0.0
        assert cfg.metric == "eval_loss"
        assert cfg.greater_is_better is False

    def test_hf_early_stop_flag_respects_control(self) -> None:
        """`sft.control.should_training_stop` is the authoritative signal."""
        from dlm.train.trainer import _hf_early_stop_flag

        stopped = SimpleNamespace(control=SimpleNamespace(should_training_stop=True))
        running = SimpleNamespace(control=SimpleNamespace(should_training_stop=False))
        assert _hf_early_stop_flag(stopped) is True
        assert _hf_early_stop_flag(running) is False

    def test_hf_early_stop_flag_missing_control_returns_none(self) -> None:
        """Mock trainers without `control` fall through to heuristic."""
        from dlm.train.trainer import _hf_early_stop_flag

        no_control = SimpleNamespace()
        empty_control = SimpleNamespace(control=SimpleNamespace())
        assert _hf_early_stop_flag(no_control) is None
        assert _hf_early_stop_flag(empty_control) is None


class TestEarlyStoppedPropagation:
    """The trainer prefers the HF signal when available."""

    def test_early_stopped_true_when_hf_flag_set(self, tmp_path: Path) -> None:
        from dlm.eval import load_summary

        store = for_dlm("01TEST", home=tmp_path)
        store.ensure_layout()
        save_manifest(store.manifest, Manifest(dlm_id="01TEST", base_model="smollm2-135m"))
        spec = BASE_MODELS["smollm2-135m"]

        def _early_stop_factory(**kwargs: Any) -> MagicMock:
            sft = _mock_trainer_factory(**kwargs)
            sft.control = SimpleNamespace(should_training_stop=True)
            return sft

        result = run(store, _parsed(), spec, _plan(), trainer_factory=_early_stop_factory)
        assert result.early_stopped is True
        summary = load_summary(result.summary_path)
        assert summary.early_stopped is True


class TestRunBranches:
    def test_disk_preflight_refuses_start(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from dlm.train.errors import DiskSpaceError

        store = for_dlm("01LOWDISK", home=tmp_path)
        store.ensure_layout()
        save_manifest(store.manifest, Manifest(dlm_id="01LOWDISK", base_model="smollm2-135m"))
        spec = BASE_MODELS["smollm2-135m"]

        fake_usage = SimpleNamespace(total=1_000, used=0, free=1_000)
        with patch("dlm.train.disk_preflight.shutil.disk_usage", return_value=fake_usage):
            import pytest

            with pytest.raises(DiskSpaceError):
                run(
                    store,
                    _parsed(),
                    spec,
                    _plan(),
                    trainer_factory=_mock_trainer_factory,
                )
