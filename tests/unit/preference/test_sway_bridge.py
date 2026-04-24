"""Direct helper coverage for sway-backed preference judge wiring."""

from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dlm.preference import JudgeUnavailableError
from dlm.preference.judge import (
    _build_sway_backend,
    _import_sway_bridge,
    _resolve_sway_trust_remote_code,
)


class FakeSwayError(Exception):
    pass


class FakeModelSpec:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeSysPath(list[str]):
    def __init__(self) -> None:
        super().__init__()
        self.inserted: list[str] = []

    def insert(self, index: int, value: str) -> None:  # type: ignore[override]
        self.inserted.append(value)
        super().insert(index, value)


def test_build_sway_backend_requires_importable_bridge() -> None:
    with (
        patch("dlm.preference.judge._import_sway_bridge", side_effect=ImportError("missing")),
        pytest.raises(JudgeUnavailableError, match="requires the sway bridge"),
    ):
        _build_sway_backend(Path("/tmp/example.dlm"))


def test_build_sway_backend_wraps_sway_resolution_errors() -> None:
    def resolve_dlm(_path: Path) -> object:
        raise FakeSwayError("no store")

    with (
        patch(
            "dlm.preference.judge._import_sway_bridge",
            return_value=(resolve_dlm, object(), FakeModelSpec, FakeSwayError),
        ),
        pytest.raises(JudgeUnavailableError, match="could not resolve"),
    ):
        _build_sway_backend(Path("/tmp/example.dlm"))


def test_build_sway_backend_wraps_generic_resolution_errors() -> None:
    def resolve_dlm(_path: Path) -> object:
        raise RuntimeError("boom")

    with (
        patch(
            "dlm.preference.judge._import_sway_bridge",
            return_value=(resolve_dlm, object(), FakeModelSpec, FakeSwayError),
        ),
        pytest.raises(JudgeUnavailableError, match="could not resolve"),
    ):
        _build_sway_backend(Path("/tmp/example.dlm"))


def test_build_sway_backend_requires_trained_adapter() -> None:
    handle = SimpleNamespace(adapter_path=None, base_model="base/model")

    def resolve_dlm(_path: Path) -> object:
        return handle

    with (
        patch(
            "dlm.preference.judge._import_sway_bridge",
            return_value=(resolve_dlm, object(), FakeModelSpec, FakeSwayError),
        ),
        pytest.raises(JudgeUnavailableError, match="requires a trained adapter"),
    ):
        _build_sway_backend(Path("/tmp/example.dlm"))


def test_build_sway_backend_wraps_backend_load_errors() -> None:
    handle = SimpleNamespace(adapter_path=Path("/tmp/adapter"), base_model="base/model")

    def resolve_dlm(_path: Path) -> object:
        return handle

    def build_backend(_spec: FakeModelSpec, *, adapter_path: Path) -> object:
        assert adapter_path == handle.adapter_path
        raise RuntimeError("backend blew up")

    with (
        patch(
            "dlm.preference.judge._import_sway_bridge",
            return_value=(resolve_dlm, build_backend, FakeModelSpec, FakeSwayError),
        ),
        patch("dlm.preference.judge._resolve_sway_trust_remote_code", return_value=False),
        pytest.raises(JudgeUnavailableError, match="could not load backend"),
    ):
        _build_sway_backend(Path("/tmp/example.dlm"))


def test_build_sway_backend_builds_model_spec_with_trust_remote_code() -> None:
    handle = SimpleNamespace(adapter_path=Path("/tmp/adapter"), base_model="base/model")
    seen: dict[str, object] = {}

    def resolve_dlm(_path: Path) -> object:
        return handle

    def build_backend(spec: FakeModelSpec, *, adapter_path: Path) -> object:
        seen["spec"] = spec
        seen["adapter_path"] = adapter_path
        return "backend"

    with (
        patch(
            "dlm.preference.judge._import_sway_bridge",
            return_value=(resolve_dlm, build_backend, FakeModelSpec, FakeSwayError),
        ),
        patch("dlm.preference.judge._resolve_sway_trust_remote_code", return_value=True),
    ):
        backend = _build_sway_backend(Path("/tmp/example.dlm"))

    assert backend == "backend"
    spec = seen["spec"]
    assert isinstance(spec, FakeModelSpec)
    assert spec.kwargs == {
        "kind": "hf",
        "base": "base/model",
        "adapter": handle.adapter_path,
        "trust_remote_code": True,
    }
    assert seen["adapter_path"] == handle.adapter_path


def test_import_sway_bridge_loads_modules_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = {
        "dlm_sway.backends": SimpleNamespace(build="build-backend"),
        "dlm_sway.core.errors": SimpleNamespace(SwayError=FakeSwayError),
        "dlm_sway.core.model": SimpleNamespace(ModelSpec=FakeModelSpec),
        "dlm_sway.integrations.dlm.resolver": SimpleNamespace(resolve_dlm="resolve-dlm"),
    }

    def fake_import_module(name: str) -> object:
        return modules[name]

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    resolve_dlm, build_backend, model_spec, sway_error = _import_sway_bridge()

    assert resolve_dlm == "resolve-dlm"
    assert build_backend == "build-backend"
    assert model_spec is FakeModelSpec
    assert sway_error is FakeSwayError


def test_import_sway_bridge_falls_back_to_local_src_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = {
        "dlm_sway.backends": SimpleNamespace(build="build-backend"),
        "dlm_sway.core.errors": SimpleNamespace(SwayError=FakeSwayError),
        "dlm_sway.core.model": SimpleNamespace(ModelSpec=FakeModelSpec),
        "dlm_sway.integrations.dlm.resolver": SimpleNamespace(resolve_dlm="resolve-dlm"),
    }
    calls = {"count": 0}

    def fake_import_module(name: str) -> object:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ImportError("first import fails")
        return modules[name]

    fake_sys_path = FakeSysPath()

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(sys, "path", fake_sys_path)
    resolve_dlm, build_backend, model_spec, sway_error = _import_sway_bridge()

    assert resolve_dlm == "resolve-dlm"
    assert build_backend == "build-backend"
    assert model_spec is FakeModelSpec
    assert sway_error is FakeSwayError
    assert fake_sys_path.inserted
    assert fake_sys_path.inserted[0].endswith("/sway/src")


def test_resolve_sway_trust_remote_code_returns_false_when_imports_are_missing() -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name in {"dlm.base_models", "dlm.doc.parser"}:
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        assert _resolve_sway_trust_remote_code(Path("/tmp/example.dlm")) is False


def test_resolve_sway_trust_remote_code_handles_parse_and_resolve_failures() -> None:
    fake_doc_parser = SimpleNamespace(
        parse_file=lambda _path: (_ for _ in ()).throw(RuntimeError("bad"))
    )
    fake_base_models = SimpleNamespace(resolve=lambda *_args, **_kwargs: object())

    with patch.dict(
        "sys.modules",
        {"dlm.doc.parser": fake_doc_parser, "dlm.base_models": fake_base_models},
    ):
        assert _resolve_sway_trust_remote_code(Path("/tmp/example.dlm")) is False

    parsed = SimpleNamespace(frontmatter=SimpleNamespace(base_model="custom-base"))
    fake_doc_parser = SimpleNamespace(parse_file=lambda _path: parsed)
    fake_base_models = SimpleNamespace(
        resolve=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no base"))
    )

    with patch.dict(
        "sys.modules",
        {"dlm.doc.parser": fake_doc_parser, "dlm.base_models": fake_base_models},
    ):
        assert _resolve_sway_trust_remote_code(Path("/tmp/example.dlm")) is False


@pytest.mark.parametrize("base_model", ["", "hf:org/model"])
def test_resolve_sway_trust_remote_code_short_circuits_for_non_registry_models(
    base_model: str,
) -> None:
    parsed = SimpleNamespace(frontmatter=SimpleNamespace(base_model=base_model))
    fake_doc_parser = SimpleNamespace(parse_file=lambda _path: parsed)
    fake_base_models = SimpleNamespace(resolve=lambda *_args, **_kwargs: object())

    with patch.dict(
        "sys.modules",
        {"dlm.doc.parser": fake_doc_parser, "dlm.base_models": fake_base_models},
    ):
        assert _resolve_sway_trust_remote_code(Path("/tmp/example.dlm")) is False


def test_resolve_sway_trust_remote_code_returns_spec_flag() -> None:
    parsed = SimpleNamespace(frontmatter=SimpleNamespace(base_model="qwen3-1.7b"))
    fake_doc_parser = SimpleNamespace(parse_file=lambda _path: parsed)
    fake_base_models = SimpleNamespace(
        resolve=lambda *_args, **_kwargs: SimpleNamespace(trust_remote_code=True)
    )

    with patch.dict(
        "sys.modules",
        {"dlm.doc.parser": fake_doc_parser, "dlm.base_models": fake_base_models},
    ):
        assert _resolve_sway_trust_remote_code(Path("/tmp/example.dlm")) is True
