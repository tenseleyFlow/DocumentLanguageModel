"""Phase-orchestrator dispatcher tests.

Uses mock SFT/DPO runners so no HF/TRL imports happen at test time.
The heavy DPO path is covered by the slow integration suite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from dlm.doc.schema import PreferenceConfig
from dlm.doc.sections import Section, SectionType
from dlm.train.preference.errors import (
    NoPreferenceContentError,
    PriorAdapterRequiredError,
)
from dlm.train.preference.phase_orchestrator import (
    PhaseResult,
    has_preference_content,
    has_sft_content,
    run_phases,
)

# ---- helpers ---------------------------------------------------------------


def _prose(body: str = "some prose content") -> Section:
    return Section(type=SectionType.PROSE, content=body, start_line=1)


def _instruction() -> Section:
    return Section(
        type=SectionType.INSTRUCTION,
        content="### Q\nhi\n### A\nhello\n",
        start_line=1,
    )


def _pref() -> Section:
    return Section(
        type=SectionType.PREFERENCE,
        content=(
            "### Prompt\nq\n### Chosen\nc\n### Rejected\nr\n"
        ),
        start_line=1,
    )


@dataclass
class _FakeTraining:
    preference: PreferenceConfig


@dataclass
class _FakeFrontmatter:
    training: _FakeTraining


@dataclass
class _FakeParsed:
    sections: tuple[Section, ...]
    frontmatter: _FakeFrontmatter


@dataclass
class _FakeRunResult:
    adapter_version: int


def _parsed(
    sections: list[Section],
    *,
    dpo_enabled: bool | None = None,
) -> Any:
    """Build a fake ParsedDlm.

    `dpo_enabled=None` leaves the `enabled` field unset so
    `resolve_preference_enabled` sees this as "user didn't specify"
    and auto-enables when preference content is present.

    `dpo_enabled=True/False` sets it explicitly — simulating a user
    who wrote `training.preference.enabled: true/false` in their
    frontmatter.
    """
    pref = (
        PreferenceConfig() if dpo_enabled is None
        else PreferenceConfig(enabled=dpo_enabled)
    )
    return _FakeParsed(
        sections=tuple(sections),
        frontmatter=_FakeFrontmatter(training=_FakeTraining(preference=pref)),
    )


# ---- content detection ----------------------------------------------------


class TestHasSftContent:
    def test_prose_with_body_counts(self) -> None:
        assert has_sft_content([_prose("hello")]) is True

    def test_empty_prose_does_not_count(self) -> None:
        assert has_sft_content([_prose("   \n  ")]) is False

    def test_instruction_counts(self) -> None:
        assert has_sft_content([_instruction()]) is True

    def test_preference_only_has_no_sft(self) -> None:
        assert has_sft_content([_pref()]) is False

    def test_empty_sections(self) -> None:
        assert has_sft_content([]) is False


class TestHasPreferenceContent:
    def test_preference_present(self) -> None:
        assert has_preference_content([_pref()]) is True

    def test_no_preference_section(self) -> None:
        assert has_preference_content([_prose(), _instruction()]) is False

    def test_empty_sections(self) -> None:
        assert has_preference_content([]) is False


# ---- dispatcher ----------------------------------------------------------


class TestDispatcherSftOnly:
    def test_sft_phase_runs_sft_only_even_with_preference(self) -> None:
        sft = MagicMock(return_value=_FakeRunResult(adapter_version=1))
        dpo = MagicMock(return_value=_FakeRunResult(adapter_version=2))
        results = run_phases(
            store=MagicMock(),
            parsed=_parsed([_prose(), _pref()], dpo_enabled=True),
            spec=MagicMock(),
            plan=MagicMock(),
            phase="sft",
            sft_runner=sft,
            dpo_runner=dpo,
        )
        assert len(results) == 1
        assert results[0].phase == "sft"
        sft.assert_called_once()
        dpo.assert_not_called()

    def test_sft_phase_skips_when_no_sft_content(self) -> None:
        sft = MagicMock()
        dpo = MagicMock()
        results = run_phases(
            store=MagicMock(),
            parsed=_parsed([_pref()]),
            spec=MagicMock(),
            plan=MagicMock(),
            phase="sft",
            sft_runner=sft,
            dpo_runner=dpo,
        )
        assert results == []
        sft.assert_not_called()
        dpo.assert_not_called()


class TestDispatcherAllPhase:
    def test_runs_sft_then_dpo_when_both_enabled(self) -> None:
        sft = MagicMock(return_value=_FakeRunResult(adapter_version=3))
        dpo = MagicMock(return_value=_FakeRunResult(adapter_version=4))
        results = run_phases(
            store=MagicMock(),
            parsed=_parsed([_prose(), _pref()], dpo_enabled=True),
            spec=MagicMock(),
            plan=MagicMock(),
            phase="all",
            sft_runner=sft,
            dpo_runner=dpo,
        )
        assert [r.phase for r in results] == ["sft", "preference"]
        sft.assert_called_once()
        dpo.assert_called_once()
        # DPO is told which adapter version to use as reference.
        _, dpo_kwargs = dpo.call_args
        assert dpo_kwargs["reference_adapter_version"] == 3

    def test_skips_dpo_when_disabled(self) -> None:
        sft = MagicMock(return_value=_FakeRunResult(adapter_version=1))
        dpo = MagicMock()
        results = run_phases(
            store=MagicMock(),
            parsed=_parsed([_prose(), _pref()], dpo_enabled=False),
            spec=MagicMock(),
            plan=MagicMock(),
            phase="all",
            sft_runner=sft,
            dpo_runner=dpo,
        )
        assert [r.phase for r in results] == ["sft"]
        dpo.assert_not_called()

    def test_all_phase_auto_warns_when_enabled_but_no_preference_content(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        sft = MagicMock(return_value=_FakeRunResult(adapter_version=1))
        dpo = MagicMock()
        with caplog.at_level(logging.WARNING):
            results = run_phases(
                store=MagicMock(),
                parsed=_parsed([_prose()], dpo_enabled=True),
                spec=MagicMock(),
                plan=MagicMock(),
                phase="all",
                sft_runner=sft,
                dpo_runner=dpo,
            )
        assert [r.phase for r in results] == ["sft"]
        dpo.assert_not_called()
        assert any("no ::preference::" in rec.message for rec in caplog.records)


class TestDispatcherDpoOnly:
    def test_dpo_only_reads_adapter_from_manifest(self, tmp_path: Path) -> None:
        store = MagicMock()
        store.manifest = tmp_path / "manifest.json"

        sft = MagicMock()
        dpo = MagicMock(return_value=_FakeRunResult(adapter_version=5))

        # Seed a manifest via the real save/load path so _resolve_reference_adapter_version
        # doesn't raise.
        from dlm.store.manifest import Manifest, save_manifest

        m = Manifest(
            dlm_id="01HZ4X7TGZM3J1A2B3C4D5E6F7",
            base_model="smollm2-135m",
            adapter_version=4,
        )
        save_manifest(store.manifest, m)

        results = run_phases(
            store=store,
            parsed=_parsed([_pref()]),
            spec=MagicMock(),
            plan=MagicMock(),
            phase="preference",
            sft_runner=sft,
            dpo_runner=dpo,
        )
        assert [r.phase for r in results] == ["preference"]
        sft.assert_not_called()
        _, dpo_kwargs = dpo.call_args
        assert dpo_kwargs["reference_adapter_version"] == 4

    def test_dpo_only_raises_on_missing_preference(self, tmp_path: Path) -> None:
        store = MagicMock()
        store.manifest = tmp_path / "manifest.json"
        from dlm.store.manifest import Manifest, save_manifest

        save_manifest(
            store.manifest,
            Manifest(
                dlm_id="01HZ4X7TGZM3J1A2B3C4D5E6F7",
                base_model="smollm2-135m",
                adapter_version=2,
            ),
        )
        with pytest.raises(NoPreferenceContentError):
            run_phases(
                store=store,
                parsed=_parsed([_prose()]),
                spec=MagicMock(),
                plan=MagicMock(),
                phase="preference",
                sft_runner=MagicMock(),
                dpo_runner=MagicMock(),
            )

    def test_dpo_only_raises_without_prior_adapter(self, tmp_path: Path) -> None:
        store = MagicMock()
        store.manifest = tmp_path / "manifest.json"
        from dlm.store.manifest import Manifest, save_manifest

        save_manifest(
            store.manifest,
            Manifest(
                dlm_id="01HZ4X7TGZM3J1A2B3C4D5E6F7",
                base_model="smollm2-135m",
                adapter_version=0,  # no SFT ever run
            ),
        )
        with pytest.raises(PriorAdapterRequiredError):
            run_phases(
                store=store,
                parsed=_parsed([_pref()]),
                spec=MagicMock(),
                plan=MagicMock(),
                phase="preference",
                sft_runner=MagicMock(),
                dpo_runner=MagicMock(),
            )


class TestPhaseResult:
    def test_phase_result_is_frozen(self) -> None:
        import dataclasses

        pr = PhaseResult(phase="sft", result=_FakeRunResult(adapter_version=1))
        with pytest.raises(dataclasses.FrozenInstanceError):
            pr.phase = "dpo"  # type: ignore[misc]


class TestAutoEnableIntegration:
    """Auto-enable: when user didn't set `enabled` and preference
    content is present, DPO runs under `--phase all`."""

    def test_unset_enabled_with_preferences_auto_runs_dpo(self) -> None:
        sft = MagicMock(return_value=_FakeRunResult(adapter_version=1))
        dpo = MagicMock(return_value=_FakeRunResult(adapter_version=2))
        results = run_phases(
            store=MagicMock(),
            parsed=_parsed([_prose(), _pref()], dpo_enabled=None),
            spec=MagicMock(),
            plan=MagicMock(),
            phase="all",
            sft_runner=sft,
            dpo_runner=dpo,
        )
        assert [r.phase for r in results] == ["sft", "preference"]

    def test_explicit_false_blocks_auto_enable(self) -> None:
        sft = MagicMock(return_value=_FakeRunResult(adapter_version=1))
        dpo = MagicMock()
        results = run_phases(
            store=MagicMock(),
            parsed=_parsed([_prose(), _pref()], dpo_enabled=False),
            spec=MagicMock(),
            plan=MagicMock(),
            phase="all",
            sft_runner=sft,
            dpo_runner=dpo,
        )
        assert [r.phase for r in results] == ["sft"]
        dpo.assert_not_called()

    def test_unset_enabled_with_no_preferences_stays_off(self) -> None:
        sft = MagicMock(return_value=_FakeRunResult(adapter_version=1))
        dpo = MagicMock()
        results = run_phases(
            store=MagicMock(),
            parsed=_parsed([_prose()], dpo_enabled=None),
            spec=MagicMock(),
            plan=MagicMock(),
            phase="all",
            sft_runner=sft,
            dpo_runner=dpo,
        )
        assert [r.phase for r in results] == ["sft"]
        dpo.assert_not_called()
