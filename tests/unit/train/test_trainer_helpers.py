"""Unit tests for `trainer.py` private helpers (Sprint 13 coverage pass).

These helpers were under-covered because the public `run()` orchestrator
requires a real HF model, which only the slow integration test can
provide. The helpers themselves are pure Python / pydantic and worth
testing directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from dlm.base_models import BASE_MODELS
from dlm.directives import ExpandResult, SourceProvenance
from dlm.directives.discovery import DiscoveredConfig
from dlm.directives.schema import DlmTrainingConfig
from dlm.doc.parser import ParsedDlm
from dlm.doc.schema import DlmFrontmatter, SourceDirective, TrainingConfig
from dlm.doc.sections import Section, SectionType
from dlm.lock import LockDecision, LockSchemaError, Severity
from dlm.replay import ChangeSet
from dlm.train.trainer import (
    _append_change_set_to_replay,
    _append_training_run,
    _attach_dlm_trainer_callback,
    _build_candidate_lock,
    _compute_weight_distribution,
    _expand_directives,
    _maybe_float,
    _maybe_record_tokenization,
    _next_run_id,
    _sample_replay_rows,
    _utc_naive,
    _validate_or_abort_lock,
)

# --- _maybe_float -----------------------------------------------------------


class TestMaybeFloat:
    def test_none_returns_none(self) -> None:
        assert _maybe_float(None) is None

    def test_numeric_returns_float(self) -> None:
        assert _maybe_float(3) == 3.0
        assert _maybe_float(2.5) == 2.5

    def test_string_numeric_parses(self) -> None:
        assert _maybe_float("1.25") == 1.25

    def test_bad_string_returns_none(self) -> None:
        assert _maybe_float("not a number") is None

    def test_invalid_type_returns_none(self) -> None:
        assert _maybe_float(object()) is None


# --- _utc_naive -------------------------------------------------------------


class TestUtcNaive:
    def test_is_naive(self) -> None:
        ts = _utc_naive()
        assert ts.tzinfo is None

    def test_microseconds_zeroed(self) -> None:
        ts = _utc_naive()
        assert ts.microsecond == 0


# --- _sample_replay_rows ----------------------------------------------------


def _fake_change_set(new_count: int) -> ChangeSet:
    return ChangeSet(
        new=[Section(type=SectionType.PROSE, content=f"row {i}") for i in range(new_count)]
    )


class _EmptyReplay:
    def load(self) -> list[object]:
        return []

    def sample_rows(self, *, k: int, now: object, rng: object) -> list[dict[str, object]]:
        raise AssertionError("should not sample when empty")


class _WarmReplay:
    def __init__(self, entries: int = 10) -> None:
        self._entries = [f"entry-{i}" for i in range(entries)]
        self.last_k: int | None = None

    def load(self) -> list[str]:
        return list(self._entries)

    def sample_rows(self, *, k: int, now: object, rng: object) -> list[dict[str, object]]:
        self.last_k = k
        return [{"row": i} for i in range(min(k, len(self._entries)))]


class TestSampleReplayRows:
    def test_cold_corpus_returns_empty(self) -> None:
        replay = _EmptyReplay()
        out = _sample_replay_rows(
            replay,  # type: ignore[arg-type]
            change_set=_fake_change_set(5),
            seed=42,
            adapter_version=1,
        )
        assert out == []

    def test_warm_corpus_samples_k_equals_2x_new_floor_32(self) -> None:
        replay = _WarmReplay(entries=200)
        out = _sample_replay_rows(
            replay,  # type: ignore[arg-type]
            change_set=_fake_change_set(100),
            seed=42,
            adapter_version=1,
        )
        # k = max(32, 2 * 100) = 200; replay has 200 entries so all returned.
        assert replay.last_k == 200
        assert len(out) == 200

    def test_small_change_set_uses_min_k_of_32(self) -> None:
        replay = _WarmReplay(entries=100)
        _sample_replay_rows(
            replay,  # type: ignore[arg-type]
            change_set=_fake_change_set(0),  # |new| = 0 → k = max(32, 0) = 32
            seed=0,
            adapter_version=1,
        )
        assert replay.last_k == 32

    def test_deterministic_across_calls(self) -> None:
        """Same (seed, adapter_version) → same RNG state per call."""
        replay1 = _WarmReplay(entries=50)
        replay2 = _WarmReplay(entries=50)

        # Both use seed=7, adapter_version=3. The RNG seeds to 10, so
        # both sample_rows calls receive an equal-state Random instance.
        _sample_replay_rows(
            replay1,  # type: ignore[arg-type]
            change_set=_fake_change_set(5),
            seed=7,
            adapter_version=3,
        )
        _sample_replay_rows(
            replay2,  # type: ignore[arg-type]
            change_set=_fake_change_set(5),
            seed=7,
            adapter_version=3,
        )
        assert replay1.last_k == replay2.last_k


# --- _next_run_id + _append_training_run -----------------------------------


def _bootstrap_store(tmp_path: Path) -> object:
    """Make a minimal StorePath with a valid manifest for helper tests."""
    from dlm.store.manifest import Manifest, save_manifest
    from dlm.store.paths import for_dlm

    home = tmp_path / "dlm-home"
    store = for_dlm("01HZ4X7TGZM3J1A2B3C4D5E6F7", home=home)
    store.ensure_layout()
    save_manifest(store.manifest, Manifest(dlm_id=store.root.name, base_model="smollm2-135m"))
    return store


_SOURCE_PATH_SENTINEL = object()


def _parsed(
    tmp_path: Path,
    *,
    source_path: object = _SOURCE_PATH_SENTINEL,
    sections: tuple[Section, ...] | None = None,
    sources: tuple[SourceDirective, ...] | None = None,
) -> ParsedDlm:
    resolved_source_path: Path | None
    if source_path is _SOURCE_PATH_SENTINEL:
        resolved_source_path = tmp_path / "doc.dlm"
        resolved_source_path.write_text("placeholder .dlm body\n", encoding="utf-8")
    else:
        assert source_path is None or isinstance(source_path, Path)
        resolved_source_path = source_path
    return ParsedDlm(
        frontmatter=DlmFrontmatter(
            dlm_id="01HZ4X7TGZM3J1A2B3C4D5E6F7",
            base_model="smollm2-135m",
            training=TrainingConfig(seed=42, sources=sources),
        ),
        sections=sections or (Section(type=SectionType.PROSE, content="x"),),
        source_path=resolved_source_path,
    )


class TestNextRunId:
    def test_missing_manifest_returns_1(self, tmp_path: Path) -> None:
        """Edge case: manifest not yet written → fresh run."""
        from dlm.store.paths import for_dlm

        home = tmp_path / "dlm-home"
        store = for_dlm("01HZ4X7TGZM3J1A2B3C4D5E6F7", home=home)
        # Don't ensure_layout / save_manifest — leave manifest missing.
        assert _next_run_id(store) == 1

    def test_empty_training_runs_returns_1(self, tmp_path: Path) -> None:
        store = _bootstrap_store(tmp_path)
        assert _next_run_id(store) == 1  # type: ignore[arg-type]

    def test_with_prior_runs_returns_max_plus_one(self, tmp_path: Path) -> None:
        from dlm.store.manifest import TrainingRunSummary, load_manifest, save_manifest

        store = _bootstrap_store(tmp_path)
        manifest = load_manifest(store.manifest)  # type: ignore[attr-defined]
        updated = manifest.model_copy(
            update={
                "training_runs": [
                    TrainingRunSummary(
                        run_id=1, started_at=_utc_naive(), adapter_version=1, seed=0
                    ),
                    TrainingRunSummary(
                        run_id=5, started_at=_utc_naive(), adapter_version=1, seed=0
                    ),
                ],
            }
        )
        save_manifest(store.manifest, updated)  # type: ignore[attr-defined]
        assert _next_run_id(store) == 6  # type: ignore[arg-type]


class TestAppendTrainingRun:
    def test_summary_path_outside_store_recorded_absolute(self, tmp_path: Path) -> None:
        """The relative_to() ValueError branch: fallback to absolute path."""
        from dlm.store.manifest import load_manifest

        store = _bootstrap_store(tmp_path)
        # A path that can't be made relative to store.root.
        outside = tmp_path / "outside" / "summary.json"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.touch()

        _append_training_run(
            store=store,  # type: ignore[arg-type]
            run_id=1,
            adapter_version=1,
            seed=0,
            steps=10,
            final_train_loss=0.5,
            final_val_loss=None,
            base_model_revision="deadbeef",
            versions={"torch": "2.4.0"},
            current_sections=[],
            summary_path=outside,
        )

        manifest = load_manifest(store.manifest)  # type: ignore[attr-defined]
        assert len(manifest.training_runs) == 1
        recorded = manifest.training_runs[0].summary_path
        # Outside-store path is absolute (matches the input).
        assert recorded == str(outside)

    def test_summary_path_under_store_recorded_relative(self, tmp_path: Path) -> None:
        from dlm.store.manifest import load_manifest

        store = _bootstrap_store(tmp_path)
        # A path inside the store.
        store.logs.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
        inside = store.logs / "summary.json"  # type: ignore[attr-defined]
        inside.touch()

        _append_training_run(
            store=store,  # type: ignore[arg-type]
            run_id=1,
            adapter_version=1,
            seed=0,
            steps=10,
            final_train_loss=0.5,
            final_val_loss=None,
            base_model_revision="deadbeef",
            versions={"torch": "2.4.0"},
            current_sections=[],
            summary_path=inside,
        )

        manifest = load_manifest(store.manifest)  # type: ignore[attr-defined]
        assert len(manifest.training_runs) == 1
        recorded = manifest.training_runs[0].summary_path
        # Relative to store root, not absolute.
        assert recorded is not None
        assert not Path(recorded).is_absolute()


# --- _snapshot_training_state (scaler path) ---------------------------------


class _FakeOptimizer:
    def state_dict(self) -> dict[str, str]:
        return {"opt": "state"}


class _FakeScaler:
    def state_dict(self) -> dict[str, str]:
        return {"scaler": "state"}


class _FakeState:
    global_step = 42
    epoch = 1.5
    best_metric = None


class _FakeSft:
    def __init__(self, with_scaler: bool = False) -> None:
        self.optimizer = _FakeOptimizer()
        self.lr_scheduler = None
        self.state = _FakeState()
        self.scaler = _FakeScaler() if with_scaler else None


def _smollm_spec() -> object:
    from dlm.base_models import BASE_MODELS

    return BASE_MODELS["smollm2-135m"]


class TestSnapshotTrainingState:
    def test_captures_scaler_when_present(self) -> None:
        from dlm.train.trainer import _snapshot_training_state

        sft = _FakeSft(with_scaler=True)
        state = _snapshot_training_state(
            sft,
            spec=_smollm_spec(),  # type: ignore[arg-type]
            versions={"torch": "2.4.0"},
            use_qlora=False,
        )
        assert state["scaler_state_dict"] == {"scaler": "state"}
        assert state["global_step"] == 42
        assert state["use_qlora"] is False

    def test_no_scaler_leaves_none(self) -> None:
        from dlm.train.trainer import _snapshot_training_state

        sft = _FakeSft(with_scaler=False)
        state = _snapshot_training_state(
            sft,
            spec=_smollm_spec(),  # type: ignore[arg-type]
            versions={"torch": "2.4.0"},
            use_qlora=True,
        )
        assert state["scaler_state_dict"] is None
        assert state["use_qlora"] is True


class TestAttachDlmTrainerCallback:
    def test_returns_when_trainer_has_no_add_callback(self) -> None:
        _attach_dlm_trainer_callback(
            trainer=SimpleNamespace(),
            recorder=MagicMock(),
            run_id=1,
            step_logger=MagicMock(),
        )

    def test_warns_and_swallows_callback_attachment_errors(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING, logger="dlm.train.trainer")
        trainer = SimpleNamespace(add_callback=MagicMock(side_effect=RuntimeError("boom")))

        _attach_dlm_trainer_callback(
            trainer=trainer,
            recorder=MagicMock(),
            run_id=1,
            step_logger=MagicMock(),
        )

        assert "failed to attach DlmTrainerCallback" in caplog.text


class TestMaybeRecordTokenization:
    def test_missing_trainer_stats_is_a_no_op(self) -> None:
        recorder = MagicMock()

        _maybe_record_tokenization(
            recorder=recorder,
            run_id=1,
            trainer=SimpleNamespace(),
        )

        recorder.record_tokenization.assert_not_called()


class TestAppendChangeSetToReplay:
    def test_all_media_change_set_does_not_append(self) -> None:
        replay = MagicMock()
        change_set = SimpleNamespace(
            new=[
                Section(type=SectionType.IMAGE, content="", media_path="hero.png"),
                Section(
                    type=SectionType.AUDIO,
                    content="",
                    media_path="clip.wav",
                    media_transcript="spoken transcript",
                ),
            ]
        )

        _append_change_set_to_replay(
            replay,
            cast(ChangeSet, change_set),
            run_id=7,
        )

        replay.append_many.assert_not_called()


class TestBuildCandidateLock:
    def test_requires_source_path(self, tmp_path: Path) -> None:
        parsed = _parsed(tmp_path, source_path=None)

        with pytest.raises(ValueError, match="source_path is required"):
            _build_candidate_lock(
                parsed=parsed,
                spec=BASE_MODELS["smollm2-135m"],
                seed=42,
                run_id=1,
                versions={"torch": "2.4.0"},
                determinism_class="strict",
                capabilities=None,
            )


class TestValidateOrAbortLock:
    def test_default_mode_reraises_unreadable_prior_lock(self, tmp_path: Path) -> None:
        store = _bootstrap_store(tmp_path)
        parsed = _parsed(tmp_path)
        (store.root / "dlm.lock").write_text("{not json", encoding="utf-8")  # type: ignore[attr-defined]

        with pytest.raises(LockSchemaError):
            _validate_or_abort_lock(
                store=store,  # type: ignore[arg-type]
                parsed=parsed,
                spec=BASE_MODELS["smollm2-135m"],
                seed=42,
                run_id=1,
                versions={"torch": "2.4.0"},
                determinism_class="strict",
                capabilities=None,
                lock_mode="default",
            )

    def test_logs_warning_mismatches_when_validator_allows_proceed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import dlm.train.trainer as trainer_mod

        store = _bootstrap_store(tmp_path)
        parsed = _parsed(tmp_path)
        decision = LockDecision(
            action="proceed_with_warnings",
            mismatches=[(Severity.WARN, "torch minor-version drift")],
            should_write_lock=True,
        )
        monkeypatch.setattr(trainer_mod, "load_lock", lambda _root: object())
        monkeypatch.setattr(
            trainer_mod,
            "validate_lock",
            lambda _prior, _candidate, mode="default": decision,
        )
        caplog.set_level(logging.WARNING, logger="dlm.train.trainer")

        got = _validate_or_abort_lock(
            store=store,  # type: ignore[arg-type]
            parsed=parsed,
            spec=BASE_MODELS["smollm2-135m"],
            seed=42,
            run_id=1,
            versions={"torch": "2.4.0"},
            determinism_class="strict",
            capabilities=None,
            lock_mode="default",
        )

        assert got == decision
        assert "dlm.lock drift: torch minor-version drift" in caplog.text


class TestComputeWeightDistribution:
    def test_counts_rows_when_directive_weights_are_active(self, tmp_path: Path) -> None:
        parsed = _parsed(
            tmp_path,
            sections=(Section(type=SectionType.PROSE, content="note", tags={"kind": "note"}),),
        )
        discovered = (
            DiscoveredConfig(
                anchor=tmp_path,
                config=DlmTrainingConfig(weights={"kind": {"note": 2.0}}),
                ignore_rules=(),
            ),
        )

        dist = _compute_weight_distribution(parsed=parsed, directive_discovered=discovered)

        assert dist == {"kind": {"note": 1}}


class TestExpandDirectives:
    def test_returns_original_parsed_when_expansion_finds_no_sections(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        parsed = _parsed(
            tmp_path,
            sources=(SourceDirective(path="corpus"),),
        )
        discovered = (
            DiscoveredConfig(
                anchor=tmp_path,
                config=DlmTrainingConfig(),
                ignore_rules=(),
            ),
        )

        def _fake_expand_sources(
            parsed_arg: ParsedDlm,
            *,
            base_path: Path,
        ) -> ExpandResult:
            assert parsed_arg is parsed
            assert parsed.source_path is not None
            assert base_path == parsed.source_path.parent
            return ExpandResult(
                sections=(),
                provenance=(SourceProvenance(path="corpus", file_count=0, total_bytes=0),),
                discovered=discovered,
            )

        monkeypatch.setattr("dlm.directives.expand_sources", _fake_expand_sources)

        new_parsed, provenance, got_discovered = _expand_directives(parsed)

        assert new_parsed is parsed
        assert provenance[0].file_count == 0
        assert got_discovered == discovered

    def test_falls_back_to_cwd_and_logs_when_sections_expand(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        parsed = _parsed(
            tmp_path,
            source_path=None,
            sources=(SourceDirective(path="corpus"),),
        )
        captured: dict[str, Path] = {}

        def _fake_expand_sources(
            parsed_arg: ParsedDlm,
            *,
            base_path: Path,
        ) -> ExpandResult:
            captured["base_path"] = base_path
            assert parsed_arg is parsed
            return ExpandResult(
                sections=(Section(type=SectionType.PROSE, content="expanded prose"),),
                provenance=(SourceProvenance(path="corpus", file_count=1, total_bytes=14),),
                discovered=(
                    DiscoveredConfig(
                        anchor=base_path,
                        config=DlmTrainingConfig(),
                        ignore_rules=(),
                    ),
                ),
            )

        monkeypatch.setattr("dlm.directives.expand_sources", _fake_expand_sources)
        caplog.set_level(logging.INFO, logger="dlm.train.trainer")

        new_parsed, provenance, discovered = _expand_directives(parsed)

        assert captured["base_path"] == Path.cwd()
        assert len(new_parsed.sections) == len(parsed.sections) + 1
        assert provenance[0].path == "corpus"
        assert len(discovered) == 1
        assert "directives: expanded 1 file(s) across 1 source(s)" in caplog.text
