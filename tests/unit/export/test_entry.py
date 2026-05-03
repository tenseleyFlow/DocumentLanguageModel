"""Direct tests for `dlm.export.entry` (per-target dispatcher)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dlm.export.entry import (
    LlamaServerPostExportRequest,
    LlamaServerPostExportResult,
    MlxServeExportRequest,
    ServerTargetExportResult,
    VllmExportRequest,
    run_llama_server_post_export,
    run_mlx_serve_target_export,
    run_vllm_target_export,
)


def _make_target(smoke_ok: bool, smoke_detail: str = "smoke ok") -> Any:
    """A fake ExportTarget with a configurable smoke result."""
    smoke_calls: list[Any] = []

    def _smoke_test(prepared: Any) -> Any:
        smoke_calls.append(prepared)
        return SimpleNamespace(attempted=True, ok=smoke_ok, detail=smoke_detail)

    target = SimpleNamespace(name="vllm", smoke_test=_smoke_test)
    target.smoke_calls = smoke_calls  # type: ignore[attr-defined]
    return target


def _vllm_request(
    *,
    target: Any,
    store: Any = None,
    spec: Any = None,
    no_smoke: bool = False,
) -> VllmExportRequest:
    return VllmExportRequest(
        target=target,
        store=store or SimpleNamespace(),
        spec=spec or SimpleNamespace(),
        served_model_name="dlm-test",
        training_sequence_len=2048,
        adapter_name=None,
        adapter_path_override=None,
        declared_adapter_names=None,
        adapter_mix=None,
        no_smoke=no_smoke,
    )


def _mlx_request(
    *,
    target: Any,
    store: Any = None,
    spec: Any = None,
    no_smoke: bool = False,
) -> MlxServeExportRequest:
    return MlxServeExportRequest(
        target=target,
        store=store or SimpleNamespace(),
        spec=spec or SimpleNamespace(),
        adapter_name=None,
        adapter_path_override=None,
        declared_adapter_names=None,
        adapter_mix=None,
        no_smoke=no_smoke,
    )


def test_run_vllm_target_export_finalizes_on_smoke_ok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = SimpleNamespace(name="vllm", export_dir=tmp_path)
    monkeypatch.setattr("dlm.export.targets.prepare_vllm_export", lambda **kwargs: prepared)
    finalize_calls: dict[str, object] = {}

    def _fake_finalize(**kwargs: object) -> Path:
        finalize_calls.update(kwargs)
        return tmp_path / "manifest.json"

    monkeypatch.setattr("dlm.export.targets.finalize_vllm_export", _fake_finalize)

    target = _make_target(smoke_ok=True)
    result = run_vllm_target_export(_vllm_request(target=target))

    assert isinstance(result, ServerTargetExportResult)
    assert result.prepared is prepared
    assert result.smoke is not None
    assert result.smoke.ok is True
    assert result.manifest_path == tmp_path / "manifest.json"
    assert finalize_calls["smoke_output_first_line"] == "smoke ok"


def test_run_vllm_target_export_skips_finalize_on_smoke_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = SimpleNamespace(export_dir=tmp_path)
    monkeypatch.setattr("dlm.export.targets.prepare_vllm_export", lambda **kwargs: prepared)
    finalize_called: list[bool] = []
    monkeypatch.setattr(
        "dlm.export.targets.finalize_vllm_export",
        lambda **kwargs: finalize_called.append(True),
    )

    target = _make_target(smoke_ok=False, smoke_detail="vllm broke")
    result = run_vllm_target_export(_vllm_request(target=target))

    assert result.manifest_path is None
    assert result.smoke is not None
    assert result.smoke.ok is False
    assert result.smoke.detail == "vllm broke"
    assert finalize_called == []


def test_run_vllm_target_export_skips_smoke_with_no_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = SimpleNamespace(export_dir=tmp_path)
    monkeypatch.setattr("dlm.export.targets.prepare_vllm_export", lambda **kwargs: prepared)

    finalize_calls: dict[str, object] = {}

    def _fake_finalize(**kwargs: object) -> Path:
        finalize_calls.update(kwargs)
        return tmp_path / "manifest.json"

    monkeypatch.setattr("dlm.export.targets.finalize_vllm_export", _fake_finalize)

    target = _make_target(smoke_ok=True)  # ignored when no_smoke=True
    result = run_vllm_target_export(_vllm_request(target=target, no_smoke=True))

    assert result.smoke is None
    assert result.manifest_path == tmp_path / "manifest.json"
    assert finalize_calls["smoke_output_first_line"] is None
    assert target.smoke_calls == []


def test_run_mlx_serve_target_export_finalizes_on_smoke_ok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = SimpleNamespace(export_dir=tmp_path)
    monkeypatch.setattr("dlm.export.targets.prepare_mlx_serve_export", lambda **kwargs: prepared)
    monkeypatch.setattr(
        "dlm.export.targets.finalize_mlx_serve_export",
        lambda **kwargs: tmp_path / "manifest.json",
    )

    target = _make_target(smoke_ok=True)
    result = run_mlx_serve_target_export(_mlx_request(target=target))

    assert result.manifest_path == tmp_path / "manifest.json"
    assert result.smoke is not None
    assert result.smoke.ok is True


def test_run_mlx_serve_target_export_skips_finalize_on_smoke_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = SimpleNamespace(export_dir=tmp_path)
    monkeypatch.setattr("dlm.export.targets.prepare_mlx_serve_export", lambda **kwargs: prepared)
    finalize_called: list[bool] = []
    monkeypatch.setattr(
        "dlm.export.targets.finalize_mlx_serve_export",
        lambda **kwargs: finalize_called.append(True),
    )

    target = _make_target(smoke_ok=False)
    result = run_mlx_serve_target_export(_mlx_request(target=target))

    assert result.manifest_path is None
    assert finalize_called == []


def test_run_mlx_serve_target_export_skips_smoke_with_no_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = SimpleNamespace(export_dir=tmp_path)
    monkeypatch.setattr("dlm.export.targets.prepare_mlx_serve_export", lambda **kwargs: prepared)
    monkeypatch.setattr(
        "dlm.export.targets.finalize_mlx_serve_export",
        lambda **kwargs: tmp_path / "manifest.json",
    )

    target = _make_target(smoke_ok=True)
    result = run_mlx_serve_target_export(_mlx_request(target=target, no_smoke=True))

    assert result.smoke is None
    assert target.smoke_calls == []


def test_run_llama_server_post_export_uses_path_override_when_given(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    override_dir = tmp_path / "override-adapter"
    prepare_calls: dict[str, object] = {}

    def _fake_prepare(**kwargs: object) -> object:
        prepare_calls.update(kwargs)
        return SimpleNamespace(name="llama-server")

    monkeypatch.setattr("dlm.export.targets.prepare_llama_server_export", _fake_prepare)

    target = _make_target(smoke_ok=True)
    request = LlamaServerPostExportRequest(
        target=target,
        store=SimpleNamespace(),  # type: ignore[arg-type]
        spec=SimpleNamespace(),  # type: ignore[arg-type]
        base_export=SimpleNamespace(  # type: ignore[arg-type]
            export_dir=tmp_path,
            manifest_path=tmp_path / "m.json",
            artifacts=[tmp_path / "a"],
        ),
        adapter_name=None,
        adapter_path_override=override_dir,
        training_sequence_len=1024,
        no_smoke=False,
    )
    result = run_llama_server_post_export(request)

    assert isinstance(result, LlamaServerPostExportResult)
    assert prepare_calls["adapter_dir"] == override_dir
    assert result.smoke is not None
    assert result.smoke.ok is True


def test_run_llama_server_post_export_resolves_default_current_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved_dir = tmp_path / "current"
    store = SimpleNamespace(
        resolve_current_adapter=lambda: resolved_dir,
        resolve_current_adapter_for=lambda name: tmp_path / "named",
    )
    prepare_calls: dict[str, object] = {}

    def _fake_prepare(**kwargs: object) -> object:
        prepare_calls.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("dlm.export.targets.prepare_llama_server_export", _fake_prepare)

    target = _make_target(smoke_ok=True)
    request = LlamaServerPostExportRequest(
        target=target,
        store=store,  # type: ignore[arg-type]
        spec=SimpleNamespace(),  # type: ignore[arg-type]
        base_export=SimpleNamespace(  # type: ignore[arg-type]
            export_dir=tmp_path, manifest_path=tmp_path / "m.json", artifacts=[]
        ),
        adapter_name=None,
        adapter_path_override=None,
        training_sequence_len=512,
        no_smoke=True,
    )
    result = run_llama_server_post_export(request)

    assert prepare_calls["adapter_dir"] == resolved_dir
    assert result.smoke is None


def test_run_llama_server_post_export_resolves_named_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    named_dir = tmp_path / "named-adapter"
    store = SimpleNamespace(
        resolve_current_adapter=lambda: tmp_path / "wrong",
        resolve_current_adapter_for=lambda name: named_dir if name == "extras" else None,
    )
    prepare_calls: dict[str, object] = {}

    monkeypatch.setattr(
        "dlm.export.targets.prepare_llama_server_export",
        lambda **kwargs: prepare_calls.update(kwargs) or SimpleNamespace(),
    )

    target = _make_target(smoke_ok=True)
    request = LlamaServerPostExportRequest(
        target=target,
        store=store,  # type: ignore[arg-type]
        spec=SimpleNamespace(),  # type: ignore[arg-type]
        base_export=SimpleNamespace(  # type: ignore[arg-type]
            export_dir=tmp_path, manifest_path=tmp_path / "m.json", artifacts=[]
        ),
        adapter_name="extras",
        adapter_path_override=None,
        training_sequence_len=None,
        no_smoke=True,
    )
    run_llama_server_post_export(request)

    assert prepare_calls["adapter_dir"] == named_dir
