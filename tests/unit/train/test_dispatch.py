"""Direct tests for `dlm.train.dispatch:run_train`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dlm.train.dispatch import (
    NoViableTrainingPlanError,
    TrainRequest,
    TrainResult,
    run_train,
)


def _fake_parsed(*, dlm_id: str = "01KPQ9X1000000000000000000") -> Any:
    """Minimal ParsedDlm shape: frontmatter.training.sequence_len + dlm_id."""
    training = SimpleNamespace(sequence_len=1024)
    frontmatter = SimpleNamespace(
        dlm_id=dlm_id,
        training=training,
        base_model="smollm2-135m",
    )
    return SimpleNamespace(frontmatter=frontmatter)


def _fake_spec() -> Any:
    """Minimal BaseModelSpec shape: params, effective_context_length, key, revision."""
    return SimpleNamespace(
        params=135_000_000,
        effective_context_length=4096,
        key="smollm2-135m",
        revision="0123456789abcdef",
    )


def _make_store(tmp_path: Path, *, manifest_exists: bool = True) -> Any:
    """Fake StorePath: ensure_layout no-op, manifest path optionally exists."""
    layout_calls: list[bool] = []
    manifest_path = tmp_path / "manifest.json"
    if manifest_exists:
        manifest_path.write_text("{}", encoding="utf-8")

    class _Store:
        manifest = manifest_path

        def ensure_layout(self) -> None:
            layout_calls.append(True)

    store = _Store()
    store.layout_calls = layout_calls  # type: ignore[attr-defined]
    return store


def _make_request(
    tmp_path: Path,
    *,
    parsed: Any | None = None,
    spec: Any | None = None,
    store: Any | None = None,
) -> TrainRequest:
    return TrainRequest(
        parsed=parsed or _fake_parsed(),  # type: ignore[arg-type]
        target_path=tmp_path / "doc.dlm",
        spec=spec or _fake_spec(),  # type: ignore[arg-type]
        store=store or _make_store(tmp_path),  # type: ignore[arg-type]
        phase="all",
        mode="fresh",
        seed=42,
        max_steps=None,
        lock_mode="default",
        world_size=1,
        strict_metrics=False,
        include_auto_mined=True,
    )


def test_run_train_returns_typed_result_on_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_obj = SimpleNamespace(name="fake-plan")
    caps_obj = SimpleNamespace()
    monkeypatch.setattr(
        "dlm.hardware.doctor",
        lambda **kwargs: SimpleNamespace(plan=plan_obj, capabilities=caps_obj),
    )
    captured: dict[str, object] = {}

    def _fake_run_phases(*args: object, **kwargs: object) -> list[object]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return ["pr1", "pr2"]

    monkeypatch.setattr(
        "dlm.train.preference.phase_orchestrator.run_phases",
        _fake_run_phases,
    )

    request = _make_request(tmp_path)
    result = run_train(request)

    assert isinstance(result, TrainResult)
    assert result.plan is plan_obj
    assert result.phase_results == ["pr1", "pr2"]
    assert request.store.layout_calls == [True]
    assert captured["kwargs"]["capabilities"] is caps_obj
    assert captured["kwargs"]["world_size"] == 1
    assert captured["kwargs"]["lock_mode"] == "default"


def test_run_train_raises_no_viable_plan_when_doctor_returns_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dlm.hardware.doctor",
        lambda **kwargs: SimpleNamespace(plan=None, capabilities=SimpleNamespace()),
    )

    request = _make_request(tmp_path)
    with pytest.raises(NoViableTrainingPlanError, match="no viable training plan"):
        run_train(request)


def test_run_train_provisions_manifest_when_missing_for_non_gated_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dlm.hardware.doctor",
        lambda **kwargs: SimpleNamespace(plan=SimpleNamespace(), capabilities=SimpleNamespace()),
    )
    monkeypatch.setattr(
        "dlm.train.preference.phase_orchestrator.run_phases",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr("dlm.base_models.is_gated", lambda spec: False)

    saved: dict[str, object] = {}

    def _fake_save(path: object, manifest: object) -> None:
        saved["path"] = path
        saved["manifest"] = manifest

    monkeypatch.setattr("dlm.store.manifest.save_manifest", _fake_save)

    store = _make_store(tmp_path, manifest_exists=False)
    request = _make_request(tmp_path, store=store)
    run_train(request)

    assert saved["path"] == store.manifest
    assert saved["manifest"].license_acceptance is None  # type: ignore[union-attr]
    assert saved["manifest"].base_model == "smollm2-135m"  # type: ignore[union-attr]


def test_run_train_provisions_manifest_with_acceptance_for_gated_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dlm.hardware.doctor",
        lambda **kwargs: SimpleNamespace(plan=SimpleNamespace(), capabilities=SimpleNamespace()),
    )
    monkeypatch.setattr(
        "dlm.train.preference.phase_orchestrator.run_phases",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr("dlm.base_models.is_gated", lambda spec: True)

    from datetime import datetime

    from dlm.base_models.license import LicenseAcceptance

    sentinel_acceptance = LicenseAcceptance(
        accepted_at=datetime(2026, 5, 1),
        license_url="https://example.test/lic",
        license_spdx="apache-2.0",
        via="cli_flag",
    )
    require_calls: dict[str, object] = {}

    def _fake_require(spec: object, *, accept_license: bool, via: str) -> LicenseAcceptance:
        require_calls["spec"] = spec
        require_calls["accept_license"] = accept_license
        require_calls["via"] = via
        return sentinel_acceptance

    monkeypatch.setattr("dlm.base_models.license.require_acceptance", _fake_require)

    saved: dict[str, object] = {}

    def _fake_save(path: object, manifest: object) -> None:
        saved["manifest"] = manifest

    monkeypatch.setattr("dlm.store.manifest.save_manifest", _fake_save)

    store = _make_store(tmp_path, manifest_exists=False)
    request = _make_request(tmp_path, store=store)
    run_train(request)

    assert require_calls["accept_license"] is True
    assert require_calls["via"] == "cli_flag"
    assert saved["manifest"].license_acceptance is sentinel_acceptance  # type: ignore[union-attr]
