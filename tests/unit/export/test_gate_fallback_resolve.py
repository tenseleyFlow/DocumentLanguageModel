"""`resolve_gate_mix` — gate-driven static --adapter-mix substitution."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

from dlm.doc.parser import ParsedDlm
from dlm.doc.schema import AdapterConfig, DlmFrontmatter, GateConfig, TrainingConfig
from dlm.doc.sections import Section, SectionType
from dlm.export.gate_fallback import resolve_gate_mix
from dlm.metrics.events import GateEvent, RunStart
from dlm.metrics.recorder import MetricsRecorder
from dlm.store.paths import StorePath
from dlm.train.gate.module import GateMetadata
from dlm.train.gate.paths import gate_config_path


def _parsed(
    *,
    gate_enabled: bool = True,
    adapters: tuple[str, ...] = ("a", "b"),
) -> ParsedDlm:
    adapter_map = {name: AdapterConfig(lora_r=4) for name in adapters} if adapters else None
    return ParsedDlm(
        frontmatter=DlmFrontmatter(
            dlm_id="01HRSHWZ" + "0" * 18,
            dlm_version=8,
            base_model="smollm2-135m",
            training=TrainingConfig(
                adapters=adapter_map,
                gate=GateConfig(enabled=gate_enabled),
            ),
        ),
        sections=(
            Section(
                type=SectionType.PROSE,
                content="body",
                start_line=0,
                adapter=None,
                tags=MappingProxyType({}),
            ),
        ),
        source_path=None,
    )


def _write_gate_config(store: StorePath, meta: GateMetadata) -> None:
    path = gate_config_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta.to_json()), encoding="utf-8")


def test_gate_disabled_returns_none(tmp_path: Path) -> None:
    store = StorePath(root=tmp_path)
    store.ensure_layout()
    assert resolve_gate_mix(store, _parsed(gate_enabled=False)) is None


def test_no_adapters_returns_none(tmp_path: Path) -> None:
    """Schema refuses gate.enabled without >=2 adapters, so build the
    parsed doc with `gate.enabled=False` + no adapters — the resolver
    must still return None because the adapter map is empty."""
    store = StorePath(root=tmp_path)
    store.ensure_layout()
    assert resolve_gate_mix(store, _parsed(gate_enabled=False, adapters=())) is None


def test_no_gate_config_returns_none(tmp_path: Path) -> None:
    store = StorePath(root=tmp_path)
    store.ensure_layout()
    assert resolve_gate_mix(store, _parsed()) is None


def test_uniform_mode_returns_uniform_mix(tmp_path: Path) -> None:
    store = StorePath(root=tmp_path)
    store.ensure_layout()
    _write_gate_config(
        store,
        GateMetadata(
            input_dim=576,
            hidden_proj_dim=64,
            adapter_names=("a", "b"),
            mode="uniform",
        ),
    )
    mix = resolve_gate_mix(store, _parsed())
    assert mix == [("a", 0.5), ("b", 0.5)]


def test_trained_mode_uses_latest_events(tmp_path: Path) -> None:
    store = StorePath(root=tmp_path)
    store.ensure_layout()
    _write_gate_config(
        store,
        GateMetadata(
            input_dim=576,
            hidden_proj_dim=64,
            adapter_names=("a", "b"),
            mode="trained",
        ),
    )
    recorder = MetricsRecorder(store.root)
    recorder.record_run_start(RunStart(run_id=1, adapter_version=1, phase="sft", seed=42))
    recorder.record_gate(
        GateEvent(run_id=1, adapter_name="a", mean_weight=0.7, sample_count=10, mode="trained")
    )
    recorder.record_gate(
        GateEvent(run_id=1, adapter_name="b", mean_weight=0.3, sample_count=10, mode="trained")
    )
    mix = resolve_gate_mix(store, _parsed())
    assert mix == [("a", 0.7), ("b", 0.3)]


def test_trained_mode_without_events_falls_back_to_uniform(tmp_path: Path) -> None:
    store = StorePath(root=tmp_path)
    store.ensure_layout()
    _write_gate_config(
        store,
        GateMetadata(
            input_dim=576,
            hidden_proj_dim=64,
            adapter_names=("a", "b"),
            mode="trained",
        ),
    )
    mix = resolve_gate_mix(store, _parsed())
    assert mix == [("a", 0.5), ("b", 0.5)]


def test_preserves_declared_adapter_order(tmp_path: Path) -> None:
    store = StorePath(root=tmp_path)
    store.ensure_layout()
    _write_gate_config(
        store,
        GateMetadata(
            input_dim=576,
            hidden_proj_dim=64,
            adapter_names=("zeta", "alpha"),  # on purpose: not sorted
            mode="trained",
        ),
    )
    recorder = MetricsRecorder(store.root)
    recorder.record_run_start(RunStart(run_id=1, adapter_version=1, phase="sft", seed=42))
    recorder.record_gate(
        GateEvent(run_id=1, adapter_name="zeta", mean_weight=0.4, sample_count=10, mode="trained")
    )
    recorder.record_gate(
        GateEvent(run_id=1, adapter_name="alpha", mean_weight=0.6, sample_count=10, mode="trained")
    )
    mix = resolve_gate_mix(store, _parsed(adapters=("zeta", "alpha")))
    # Order must match the config's adapter_names tuple, not alphabetic.
    assert mix == [("zeta", 0.4), ("alpha", 0.6)]
