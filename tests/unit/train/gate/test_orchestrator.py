"""Post-SFT gate orchestration — probe extraction + run_post_sft_gate."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

import dlm.train.gate.orchestrator as gate_orchestrator
from dlm.doc.parser import ParsedDlm
from dlm.doc.schema import AdapterConfig, DlmFrontmatter, GateConfig, TrainingConfig
from dlm.doc.sections import Section, SectionType
from dlm.metrics.events import RunStart
from dlm.metrics.recorder import MetricsRecorder
from dlm.store.paths import StorePath
from dlm.train.gate.errors import GateTrainingError
from dlm.train.gate.orchestrator import (
    GateProbe,
    probes_from_sections,
    run_post_sft_gate,
)


def _frontmatter(
    *,
    gate_enabled: bool = True,
    adapters: tuple[str, ...] = ("a", "b"),
) -> DlmFrontmatter:
    adapter_map = {name: AdapterConfig(lora_r=4) for name in adapters} if adapters else None
    return DlmFrontmatter(
        dlm_id="01HRSHWZ" + "0" * 18,
        dlm_version=8,
        base_model="smollm2-135m",
        training=TrainingConfig(
            adapters=adapter_map,
            gate=GateConfig(
                enabled=gate_enabled,
                cold_start_floor=4,
                steps=20,  # short for unit tests
            ),
        ),
    )


def _instruction(content: str, *, adapter: str | None) -> Section:
    return Section(
        type=SectionType.INSTRUCTION,
        content=content,
        start_line=0,
        adapter=adapter,
        tags=MappingProxyType({}),
    )


def _prose(content: str, *, adapter: str | None) -> Section:
    return Section(
        type=SectionType.PROSE,
        content=content,
        start_line=0,
        adapter=adapter,
        tags=MappingProxyType({}),
    )


def _preference(content: str, *, adapter: str | None) -> Section:
    return Section(
        type=SectionType.PREFERENCE,
        content=content,
        start_line=0,
        adapter=adapter,
        tags=MappingProxyType({}),
    )


def _parsed(sections: tuple[Section, ...], **fm_kwargs: object) -> ParsedDlm:
    return ParsedDlm(
        frontmatter=_frontmatter(**fm_kwargs),  # type: ignore[arg-type]
        sections=sections,
        source_path=None,
    )


class TestProbesFromSections:
    def test_drops_untagged_sections(self) -> None:
        sections = (
            _prose("hello", adapter=None),
            _prose("world", adapter="a"),
        )
        probes = probes_from_sections(_parsed(sections))
        assert probes == [GateProbe(adapter_name="a", prompt="world")]

    def test_extracts_instruction_question(self) -> None:
        body = "### Q\nWhat is lexing?\n### A\nTurning source into tokens.\n"
        probes = probes_from_sections(_parsed((_instruction(body, adapter="a"),)))
        assert probes == [GateProbe(adapter_name="a", prompt="What is lexing?")]

    def test_multiple_qa_uses_first_pair(self) -> None:
        body = (
            "### Q\nFirst question?\n### A\nFirst answer.\n\n"
            "### Q\nSecond question?\n### A\nSecond answer.\n"
        )
        probes = probes_from_sections(_parsed((_instruction(body, adapter="b"),)))
        assert probes[0].prompt == "First question?"

    def test_extracts_preference_prompt(self) -> None:
        body = "### Prompt\nWhich answer is better?\n### Chosen\nA\n### Rejected\nB\n"
        probes = probes_from_sections(_parsed((_preference(body, adapter="b"),)))
        assert probes == [GateProbe(adapter_name="b", prompt="Which answer is better?")]

    def test_unparseable_instruction_is_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        probes = probes_from_sections(_parsed((_instruction("no Q/A pairs here", adapter="a"),)))
        assert probes == []

    def test_prose_truncates_to_cap(self) -> None:
        long = "x" * 5000
        probes = probes_from_sections(_parsed((_prose(long, adapter="a"),)))
        assert len(probes) == 1
        assert len(probes[0].prompt) == 2048


class TestRunPostSftGate:
    def test_disabled_gate_returns_none(self, tmp_path: Path) -> None:
        parsed = _parsed(
            (_prose("x", adapter="a"),),
            gate_enabled=False,
        )
        store = StorePath(root=tmp_path)
        store.ensure_layout()
        recorder = MetricsRecorder(tmp_path)
        recorder.record_run_start(RunStart(run_id=1, adapter_version=1, phase="sft", seed=42))
        result = run_post_sft_gate(
            store,
            parsed,
            run_id=1,
            recorder=recorder,
            embed=lambda _p: _tensor(4),
            input_dim=4,
        )
        assert result is None

    def test_single_adapter_returns_none(self, tmp_path: Path) -> None:
        # A single-named-adapter doc can't carry an enabled gate (the
        # schema refuses it), so build the frontmatter with no adapter
        # map at all to simulate a gate that's "enabled" but has
        # nothing to route between.
        parsed = ParsedDlm(
            frontmatter=DlmFrontmatter(
                dlm_id="01HRSHWZ" + "0" * 18,
                dlm_version=8,
                base_model="smollm2-135m",
                training=TrainingConfig(
                    adapters=None,
                    gate=GateConfig(enabled=False),
                ),
            ),
            sections=(_prose("x", adapter="a"),),
            source_path=None,
        )
        store = StorePath(root=tmp_path)
        store.ensure_layout()
        recorder = MetricsRecorder(tmp_path)
        result = run_post_sft_gate(
            store,
            parsed,
            run_id=1,
            recorder=recorder,
            embed=lambda _p: _tensor(4),
            input_dim=4,
        )
        assert result is None

    def test_exactly_one_named_adapter_returns_none(self, tmp_path: Path) -> None:
        parsed = _parsed((_prose("x", adapter="solo"),), gate_enabled=False, adapters=("solo",))
        object.__setattr__(parsed.frontmatter.training.gate, "enabled", True)
        store = StorePath(root=tmp_path)
        store.ensure_layout()
        recorder = MetricsRecorder(tmp_path)
        result = run_post_sft_gate(
            store,
            parsed,
            run_id=1,
            recorder=recorder,
            embed=lambda _p: _tensor(4),
            input_dim=4,
        )
        assert result is None

    def test_cold_start_fallback_records_uniform_events(self, tmp_path: Path) -> None:
        parsed = _parsed((_prose("only-a", adapter="a"),))
        store = StorePath(root=tmp_path)
        store.ensure_layout()
        recorder = MetricsRecorder(tmp_path)
        recorder.record_run_start(RunStart(run_id=1, adapter_version=1, phase="sft", seed=42))
        result = run_post_sft_gate(
            store,
            parsed,
            run_id=1,
            recorder=recorder,
            embed=lambda _p: _tensor(4),
            input_dim=4,
        )
        assert result is not None
        assert result.mode == "uniform"

        # Gate config file was written (uniform mode).
        from dlm.train.gate.paths import gate_config_path

        assert gate_config_path(store).exists()

        # Events for every declared adapter, each with mean_weight = 1/N.
        from dlm.metrics.db import connect

        with connect(tmp_path) as conn:
            rows = list(
                conn.execute(
                    "SELECT adapter_name, mean_weight, sample_count, mode "
                    "FROM gate_events WHERE run_id = 1 ORDER BY adapter_name"
                )
            )
        assert len(rows) == 2
        assert {name for name, _w, _c, _m in rows} == {"a", "b"}
        for _name, weight, _count, mode in rows:
            assert mode == "uniform"
            assert weight == pytest.approx(0.5)

    def test_trained_mode_records_calibrated_mean_weight(self, tmp_path: Path) -> None:
        # Enough supervising samples for both adapters. Use two clear
        # clusters so training actually separates them.
        import torch

        sections: list[Section] = []
        for i in range(6):
            sections.append(_prose(f"alpha-{i}", adapter="a"))
            sections.append(_prose(f"beta-{i}", adapter="b"))
        parsed = _parsed(tuple(sections))

        def embed(prompt: str) -> torch.Tensor:
            # Cluster 'alpha' at +1, 'beta' at -1.
            sign = 1.0 if prompt.startswith("alpha") else -1.0
            return sign * torch.ones(4) + 0.05 * torch.randn(4)

        store = StorePath(root=tmp_path)
        store.ensure_layout()
        recorder = MetricsRecorder(tmp_path)
        recorder.record_run_start(RunStart(run_id=1, adapter_version=1, phase="sft", seed=42))
        result = run_post_sft_gate(
            store,
            parsed,
            run_id=1,
            recorder=recorder,
            embed=embed,
            input_dim=4,
        )
        assert result is not None
        assert result.mode == "trained"
        # Calibrated mean weights should be (approximately) the prior
        # split given a balanced supervision set — the average weight
        # summed across adapters is always 1.0.
        names = tuple(result.per_adapter_mean_weight.keys())
        assert set(names) == {"a", "b"}
        total = sum(result.per_adapter_mean_weight.values())
        assert total == pytest.approx(1.0, abs=1e-4)

        # gate_events rows reflect the calibrated weights.
        from dlm.metrics.db import connect

        with connect(tmp_path) as conn:
            rows = dict(
                conn.execute(
                    "SELECT adapter_name, mean_weight FROM gate_events WHERE run_id = 1"
                ).fetchall()
            )
        assert rows["a"] == pytest.approx(result.per_adapter_mean_weight["a"])
        assert rows["b"] == pytest.approx(result.per_adapter_mean_weight["b"])

    def test_gate_training_error_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parsed = _parsed(
            (
                _prose("alpha", adapter="a"),
                _prose("beta", adapter="b"),
            )
        )
        store = StorePath(root=tmp_path)
        store.ensure_layout()
        recorder = MetricsRecorder(tmp_path)
        recorder.record_run_start(RunStart(run_id=1, adapter_version=1, phase="sft", seed=42))

        def _raise_gate_error(*args: object, **kwargs: object) -> None:
            raise GateTrainingError("boom")

        monkeypatch.setattr(gate_orchestrator, "train_gate", _raise_gate_error)
        result = run_post_sft_gate(
            store,
            parsed,
            run_id=1,
            recorder=recorder,
            embed=lambda _p: _tensor(4),
            input_dim=4,
        )
        assert result is None

        # Divergence emits one `mode="diverged"` GateEvent per declared
        # adapter so `dlm show` surfaces the failure instead of silently
        # skipping the gate.
        from dlm.metrics import queries as _queries

        events = _queries.gate_events_for_run(tmp_path, 1)
        assert {e.adapter_name for e in events} == {"a", "b"}
        assert all(e.mode == "diverged" for e in events)
        assert all(e.mean_weight == 0.0 and e.sample_count == 0 for e in events)


def _tensor(d: int) -> Any:
    import torch

    return torch.zeros(d)
