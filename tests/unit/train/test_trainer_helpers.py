"""Unit tests for `trainer.py` private helpers (Sprint 13 coverage pass).

These helpers were under-covered because the public `run()` orchestrator
requires a real HF model, which only the slow integration test can
provide. The helpers themselves are pure Python / pydantic and worth
testing directly.
"""

from __future__ import annotations

from pathlib import Path

from dlm.train.trainer import (
    _append_training_run,
    _maybe_float,
    _next_run_id,
    _sample_replay_rows,
    _utc_naive,
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


class _FakeChangeSet:
    def __init__(self, new_count: int) -> None:
        self.new = [object() for _ in range(new_count)]


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
            change_set=_FakeChangeSet(5),  # type: ignore[arg-type]
            seed=42,
            adapter_version=1,
        )
        assert out == []

    def test_warm_corpus_samples_k_equals_2x_new_floor_32(self) -> None:
        replay = _WarmReplay(entries=200)
        out = _sample_replay_rows(
            replay,  # type: ignore[arg-type]
            change_set=_FakeChangeSet(100),  # type: ignore[arg-type]
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
            change_set=_FakeChangeSet(0),  # |new| = 0 → k = max(32, 0) = 32
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
            change_set=_FakeChangeSet(5),  # type: ignore[arg-type]
            seed=7,
            adapter_version=3,
        )
        _sample_replay_rows(
            replay2,  # type: ignore[arg-type]
            change_set=_FakeChangeSet(5),  # type: ignore[arg-type]
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
