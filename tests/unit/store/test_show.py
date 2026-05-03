"""Direct tests for `dlm.store.show:gather_store_view` + private helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dlm.store.show import (
    StoreView,
    StoreViewRequest,
    _summarize_base_security,
    _summarize_gate,
    _summarize_preference_mining,
    _summarize_training_cache,
    _summarize_training_sources_and_discovered,
    gather_store_view,
)

_DLM_ID = "01KPQ9X1000000000000000000"
_REV = "0123456789abcdef0123456789abcdef01234567"


def _write_doc(path: Path, *, body: str = "") -> None:
    payload = f"---\ndlm_id: {_DLM_ID}\ndlm_version: 14\nbase_model: smollm2-135m\n---\n"
    path.write_text(payload + body, encoding="utf-8")


def _parsed(home: Path, doc: Path) -> Any:
    from dlm.doc.parser import parse_file

    return parse_file(doc)


def test_gather_store_view_returns_uninitialized_when_manifest_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DLM_HOME", str(tmp_path / "home"))
    doc = tmp_path / "doc.dlm"
    _write_doc(doc)

    from dlm.store.paths import for_dlm

    parsed = _parsed(tmp_path / "home", doc)
    store = for_dlm(parsed.frontmatter.dlm_id)

    view = gather_store_view(StoreViewRequest(parsed=parsed, target_path=doc, store=store))

    assert isinstance(view, StoreView)
    assert view.inspection is None
    assert view.training_cache is None
    assert view.gate is None
    assert view.preference_mining is None
    assert view.base_security is None
    assert view.parsed_dlm_id == _DLM_ID
    assert view.parsed_base_model == "smollm2-135m"
    assert view.training_sources is None
    assert view.discovered_configs == []


def test_gather_store_view_populates_inspection_when_manifest_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("DLM_HOME", str(home))
    doc = tmp_path / "doc.dlm"
    _write_doc(doc)

    from dlm.store.manifest import Manifest, save_manifest
    from dlm.store.paths import for_dlm

    parsed = _parsed(home, doc)
    store = for_dlm(parsed.frontmatter.dlm_id)
    store.ensure_layout()
    save_manifest(
        store.manifest,
        Manifest(
            dlm_id=_DLM_ID,
            base_model="smollm2-135m",
            base_model_revision=_REV,
            source_path=doc.resolve(),
        ),
    )

    view = gather_store_view(StoreViewRequest(parsed=parsed, target_path=doc, store=store))

    assert view.inspection is not None
    assert view.inspection.dlm_id == _DLM_ID
    assert view.inspection.base_model == "smollm2-135m"
    assert view.training_cache is None  # tokenized_cache_dir doesn't exist yet
    assert view.gate is None  # no gate config + no events
    assert view.preference_mining is None  # no metrics
    assert view.base_security is not None
    assert view.base_security["base_model"] == "smollm2-135m"


def test_summarize_training_sources_returns_none_without_directives() -> None:
    parsed = SimpleNamespace(frontmatter=SimpleNamespace(training=SimpleNamespace(sources=[])))
    sources, discovered = _summarize_training_sources_and_discovered(parsed, Path("/tmp"))
    assert sources is None
    assert discovered == []


def test_summarize_training_sources_returns_declared_on_expand_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlm.directives.errors import DirectiveError

    directive = SimpleNamespace(
        path="docs",
        include=("**/*",),
        exclude=(),
        max_files=None,
        max_bytes_per_file=None,
    )
    parsed = SimpleNamespace(
        frontmatter=SimpleNamespace(training=SimpleNamespace(sources=[directive]))
    )

    def _raise(*args: object, **kwargs: object) -> None:
        raise DirectiveError("expansion failed")

    monkeypatch.setattr("dlm.store.show._expand_sources", _raise)

    sources, discovered = _summarize_training_sources_and_discovered(parsed, Path("/tmp"))
    assert sources is not None
    assert len(sources) == 1
    assert sources[0]["path"] == "docs"
    assert "file_count" not in sources[0]
    assert discovered == []


def test_summarize_training_sources_succeeds_with_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directive = SimpleNamespace(
        path="docs",
        include=("**/*",),
        exclude=(),
        max_files=None,
        max_bytes_per_file=None,
    )
    parsed = SimpleNamespace(
        frontmatter=SimpleNamespace(training=SimpleNamespace(sources=[directive]))
    )
    prov = SimpleNamespace(
        file_count=3,
        total_bytes=1024,
        skipped_binary=0,
        skipped_encoding=0,
        skipped_over_size=1,
    )
    discovered_config = SimpleNamespace(
        anchor=Path("/anchor"),
        config=SimpleNamespace(
            include=("*.py",),
            exclude=(),
            exclude_defaults=True,
            metadata={"k": "v"},
        ),
        ignore_rules=["pattern1"],
    )

    def _ok(*args: object, **kwargs: object) -> object:
        return SimpleNamespace(provenance=[prov], discovered=[discovered_config])

    monkeypatch.setattr("dlm.store.show._expand_sources", _ok)

    sources, discovered = _summarize_training_sources_and_discovered(parsed, Path("/tmp"))
    assert sources is not None
    assert sources[0]["file_count"] == 3
    assert sources[0]["total_bytes"] == 1024
    assert len(discovered) == 1
    assert discovered[0]["anchor"] == "/anchor"
    assert discovered[0]["has_training_yaml"] is True
    assert discovered[0]["has_ignore"] is True


def test_summarize_training_sources_pads_when_provenance_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d1 = SimpleNamespace(path="a", include=(), exclude=(), max_files=None, max_bytes_per_file=None)
    d2 = SimpleNamespace(path="b", include=(), exclude=(), max_files=None, max_bytes_per_file=None)
    parsed = SimpleNamespace(
        frontmatter=SimpleNamespace(training=SimpleNamespace(sources=[d1, d2]))
    )
    short_prov = SimpleNamespace(
        file_count=1, total_bytes=10, skipped_binary=0, skipped_encoding=0, skipped_over_size=0
    )

    def _ok(*args: object, **kwargs: object) -> object:
        # one declared has provenance, the other doesn't (defensive padding)
        discovered_no_cfg = SimpleNamespace(anchor=Path("/anchor"), config=None, ignore_rules=[])
        return SimpleNamespace(provenance=[short_prov], discovered=[discovered_no_cfg])

    monkeypatch.setattr("dlm.store.show._expand_sources", _ok)

    sources, discovered = _summarize_training_sources_and_discovered(parsed, Path("/tmp"))
    assert sources is not None
    assert len(sources) == 2
    assert sources[0]["file_count"] == 1
    assert "file_count" not in sources[1]  # padded with declared-only
    assert discovered[0]["has_training_yaml"] is False
    assert discovered[0]["exclude_defaults"] is True


def test_summarize_training_cache_none_when_dir_missing(tmp_path: Path) -> None:
    assert _summarize_training_cache(tmp_path / "missing", tmp_path) is None


def test_summarize_training_cache_with_no_last_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    monkeypatch.setattr(
        "dlm.directives.cache.TokenizedCache.open",
        classmethod(lambda cls, _path: SimpleNamespace(entry_count=5, total_bytes=2048)),
    )
    monkeypatch.setattr("dlm.metrics.queries.latest_tokenization", lambda root: None)

    snap = _summarize_training_cache(cache_dir, tmp_path)
    assert snap is not None
    assert snap["entry_count"] == 5
    assert snap["bytes"] == 2048
    assert snap["last_run_hit_rate"] is None
    assert snap["last_run_id"] is None


def test_summarize_training_cache_with_last_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    monkeypatch.setattr(
        "dlm.directives.cache.TokenizedCache.open",
        classmethod(lambda cls, _path: SimpleNamespace(entry_count=3, total_bytes=512)),
    )
    monkeypatch.setattr(
        "dlm.metrics.queries.latest_tokenization",
        lambda root: SimpleNamespace(hit_rate=0.75, run_id=42),
    )

    snap = _summarize_training_cache(cache_dir, tmp_path)
    assert snap is not None
    assert snap["last_run_hit_rate"] == 0.75
    assert snap["last_run_id"] == 42


def test_summarize_gate_none_when_no_cfg_no_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SimpleNamespace(root=tmp_path)
    monkeypatch.setattr(
        "dlm.train.gate.paths.gate_config_path", lambda s: tmp_path / "missing.json"
    )
    monkeypatch.setattr("dlm.metrics.queries.latest_gate_events", lambda root: [])

    assert _summarize_gate(store) is None  # type: ignore[arg-type]


def test_summarize_gate_diverged_when_no_cfg_but_diverged_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SimpleNamespace(root=tmp_path)
    event = SimpleNamespace(
        adapter_name="adapter-a",
        mean_weight=0.5,
        sample_count=10,
        mode="diverged",
        run_id=99,
    )
    monkeypatch.setattr(
        "dlm.train.gate.paths.gate_config_path", lambda s: tmp_path / "missing.json"
    )
    monkeypatch.setattr("dlm.metrics.queries.latest_gate_events", lambda root: [event])

    snap = _summarize_gate(store)  # type: ignore[arg-type]
    assert snap is not None
    assert snap["mode"] == "diverged"
    assert snap["last_run_id"] == 99
    assert snap["per_adapter"][0]["adapter_name"] == "adapter-a"


def test_summarize_gate_with_cfg_and_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path = tmp_path / "gate_config.json"
    cfg_path.write_text(
        json.dumps(
            {"mode": "trained", "adapter_names": ["a", "b"], "input_dim": 32, "hidden_proj_dim": 16}
        ),
        encoding="utf-8",
    )
    store = SimpleNamespace(root=tmp_path)
    event = SimpleNamespace(
        adapter_name="a", mean_weight=0.6, sample_count=5, mode="active", run_id=7
    )

    monkeypatch.setattr("dlm.train.gate.paths.gate_config_path", lambda s: cfg_path)
    monkeypatch.setattr("dlm.metrics.queries.latest_gate_events", lambda root: [event])

    snap = _summarize_gate(store)  # type: ignore[arg-type]
    assert snap is not None
    assert snap["mode"] == "trained"
    assert snap["last_run_id"] == 7
    assert snap["per_adapter"][0]["adapter_name"] == "a"


def test_summarize_gate_with_cfg_no_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path = tmp_path / "gate_config.json"
    cfg_path.write_text(
        json.dumps(
            {"mode": "trained", "adapter_names": ["a", "b"], "input_dim": 32, "hidden_proj_dim": 16}
        ),
        encoding="utf-8",
    )
    store = SimpleNamespace(root=tmp_path)

    monkeypatch.setattr("dlm.train.gate.paths.gate_config_path", lambda s: cfg_path)
    monkeypatch.setattr("dlm.metrics.queries.latest_gate_events", lambda root: [])

    snap = _summarize_gate(store)  # type: ignore[arg-type]
    assert snap is not None
    assert snap["last_run_id"] is None
    assert snap["per_adapter"] == [{"adapter_name": "a"}, {"adapter_name": "b"}]


def test_summarize_preference_mining_none_when_no_totals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlm.metrics.queries.preference_mining_totals", lambda root: None)
    assert _summarize_preference_mining(tmp_path) is None


def test_summarize_preference_mining_with_totals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    totals = SimpleNamespace(
        run_count=2, event_count=5, total_mined_pairs=10, total_skipped_prompts=3
    )
    last = SimpleNamespace(run_id=42)
    monkeypatch.setattr("dlm.metrics.queries.preference_mining_totals", lambda root: totals)
    monkeypatch.setattr("dlm.metrics.queries.latest_preference_mining", lambda root: last)
    monkeypatch.setattr(
        "dlm.metrics.queries.preference_mining_for_run", lambda root, run_id: ["row1", "row2"]
    )
    monkeypatch.setattr(
        "dlm.metrics.queries.preference_mining_to_dict", lambda rows: [{"key": "value"}]
    )

    snap = _summarize_preference_mining(tmp_path)
    assert snap is not None
    assert snap["run_count"] == 2
    assert snap["last_run_id"] == 42
    assert snap["last_run_event_count"] == 2
    assert snap["last_event"] == {"key": "value"}


def test_summarize_base_security_returns_dict_for_known_base() -> None:
    snap = _summarize_base_security("smollm2-135m")
    assert snap is not None
    assert snap["base_model"] == "smollm2-135m"
    assert "trust_remote_code" in snap


def test_summarize_base_security_returns_none_when_resolve_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlm.base_models.errors import UnknownBaseModelError

    def _raise(*args: object, **kwargs: object) -> None:
        raise UnknownBaseModelError("nope", known_keys=())

    monkeypatch.setattr("dlm.base_models.resolve", _raise)

    assert _summarize_base_security("definitely-not-real") is None
