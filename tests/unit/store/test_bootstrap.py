"""Direct tests for `dlm.store.bootstrap:run_init`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dlm.store.bootstrap import (
    InitRequest,
    InitResult,
    ScaffoldKind,
    run_init,
)
from dlm.store.manifest import load_manifest
from dlm.store.paths import for_dlm

_REV = "0123456789abcdef0123456789abcdef01234567"


def _spec(key: str = "smollm2-135m") -> Any:
    return SimpleNamespace(key=key, revision=_REV)


def _make_request(
    tmp_path: Path,
    *,
    template_name: str | None = None,
    scaffold_kind: ScaffoldKind = ScaffoldKind.TEXT,
    force: bool = False,
) -> InitRequest:
    return InitRequest(
        path=tmp_path / "doc.dlm",
        spec=_spec(),  # type: ignore[arg-type]
        acceptance=None,
        force=force,
        template_name=template_name,
        scaffold_kind=scaffold_kind,
    )


def test_run_init_writes_text_scaffold_and_provisions_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("DLM_HOME", str(home))

    request = _make_request(tmp_path)
    result = run_init(request)

    assert isinstance(result, InitResult)
    assert result.applied_template is None
    assert result.dlm_id  # minted ULID
    assert request.path.exists()
    body = request.path.read_text(encoding="utf-8")
    assert "::instruction::" in body
    assert "::image" not in body
    assert "::audio" not in body
    assert f"dlm_id: {result.dlm_id}" in body

    store = for_dlm(result.dlm_id, home=home)
    manifest = load_manifest(store.manifest)
    assert manifest.dlm_id == result.dlm_id
    assert manifest.base_model == "smollm2-135m"
    assert manifest.license_acceptance is None


def test_run_init_writes_vision_scaffold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DLM_HOME", str(tmp_path / "home"))

    request = _make_request(tmp_path, scaffold_kind=ScaffoldKind.VISION)
    run_init(request)

    body = request.path.read_text(encoding="utf-8")
    assert "::image" in body
    assert "dlm_version: 10" in body


def test_run_init_writes_audio_scaffold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DLM_HOME", str(tmp_path / "home"))

    request = _make_request(tmp_path, scaffold_kind=ScaffoldKind.AUDIO)
    run_init(request)

    body = request.path.read_text(encoding="utf-8")
    assert "::audio" in body
    assert "dlm_version: 11" in body


def test_run_init_applies_template_via_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DLM_HOME", str(tmp_path / "home"))

    sentinel_template = SimpleNamespace(meta=SimpleNamespace(name="custom", title="Custom"))
    apply_calls: dict[str, object] = {}

    def _fake_apply(
        name: str,
        target: Path,
        *,
        force: bool = False,
        accept_license: bool = False,
    ) -> object:
        apply_calls["name"] = name
        apply_calls["target"] = target
        apply_calls["force"] = force
        apply_calls["accept_license"] = accept_license
        return SimpleNamespace(template=sentinel_template, dlm_id="01ABC123")

    monkeypatch.setattr("dlm.templates.init.apply_template", _fake_apply)

    request = _make_request(tmp_path, template_name="my-template", force=True)
    result = run_init(request)

    assert result.dlm_id == "01ABC123"
    assert result.applied_template is not None
    assert result.applied_template.template is sentinel_template
    assert apply_calls["name"] == "my-template"
    assert apply_calls["force"] is True
    assert apply_calls["accept_license"] is True
