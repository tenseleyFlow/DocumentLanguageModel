"""Sprint 26 X1 — `dlm.export.sway_json` helper-module unit tests.

Exists at this path (rather than under ``tests/unit/cli/``) because
CI's ``Coverage gate — src/dlm/export = 100%`` runs ``pytest tests/unit/export``
exclusively. Without these tests in this dir, ``sway_json.py``'s
coverage shows 0% under the gate even though the CLI-side tests
exercise everything.

The CLI-flag-wiring test (``--emit-sway-json`` shows up in
``dlm export --help``) lives at ``tests/unit/cli/test_export_sway_json.py``
because it touches the CLI surface, not just the helper.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from dlm.export.sway_json import SwayJsonExportError, write_sway_json


def _install_fake_dlm_sway(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject minimal `dlm_sway.integrations.dlm.{autogen,resolver}`
    modules so write_sway_json runs without the real ``[sway]`` extra
    installed. Mirrors the fakes in ``tests/unit/cli/test_export_sway_json.py``
    so the helper-module surface is exercised independently of the
    CLI flag wiring test.
    """
    dlm_sway = types.ModuleType("dlm_sway")
    integrations = types.ModuleType("dlm_sway.integrations")
    integrations_dlm = types.ModuleType("dlm_sway.integrations.dlm")
    autogen = types.ModuleType("dlm_sway.integrations.dlm.autogen")
    resolver = types.ModuleType("dlm_sway.integrations.dlm.resolver")

    class _FakeHandle:
        dlm_id = "01TEST"

    def _resolve_dlm(_path: Path) -> _FakeHandle:
        return _FakeHandle()

    def _build_spec_dict(_handle: _FakeHandle, *, dlm_source: str) -> dict[str, object]:
        return {
            "version": 1,
            "models": {
                "base": {"kind": "hf", "base": "smollm2-135m"},
                "ft": {"kind": "hf", "base": "smollm2-135m"},
            },
            "defaults": {"seed": 0},
            "suite": [
                {"name": "dk", "kind": "delta_kl", "prompts": ["x"]},
            ],
            "dlm_source": dlm_source,
        }

    resolver.resolve_dlm = _resolve_dlm  # type: ignore[attr-defined]
    autogen.build_spec_dict = _build_spec_dict  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "dlm_sway", dlm_sway)
    monkeypatch.setitem(sys.modules, "dlm_sway.integrations", integrations)
    monkeypatch.setitem(sys.modules, "dlm_sway.integrations.dlm", integrations_dlm)
    monkeypatch.setitem(sys.modules, "dlm_sway.integrations.dlm.autogen", autogen)
    monkeypatch.setitem(sys.modules, "dlm_sway.integrations.dlm.resolver", resolver)


class TestWriteSwayJsonHelper:
    def test_writes_yaml_in_export_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Helper writes ``<export_dir>/sway.yaml`` and returns the path."""
        _install_fake_dlm_sway(monkeypatch)
        dlm_path = tmp_path / "doc.dlm"
        dlm_path.write_text("---\ndlm_id: 01TEST\n---\nbody\n", encoding="utf-8")
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        out = write_sway_json(dlm_path, export_dir)

        assert out == export_dir / "sway.yaml"
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "version: 1" in content
        assert "delta_kl" in content
        assert "dlm_source:" in content

    def test_creates_export_dir_if_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Caller may pass a non-existent export_dir — helper mkdirs it."""
        _install_fake_dlm_sway(monkeypatch)
        dlm_path = tmp_path / "doc.dlm"
        dlm_path.write_text("---\ndlm_id: 01\n---\n", encoding="utf-8")
        export_dir = tmp_path / "fresh" / "nested" / "export"

        out = write_sway_json(dlm_path, export_dir)
        assert out.exists()
        assert export_dir.is_dir()

    def test_dlm_sway_missing_raises_typed_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ImportError on dlm_sway → SwayJsonExportError with install hint."""
        for mod in list(sys.modules):
            if mod == "dlm_sway" or mod.startswith("dlm_sway."):
                monkeypatch.delitem(sys.modules, mod, raising=False)

        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name.startswith("dlm_sway"):
                raise ImportError("dlm-sway not installed (test stub)")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        dlm_path = tmp_path / "doc.dlm"
        dlm_path.write_text("---\ndlm_id: 01\n---\n", encoding="utf-8")

        with pytest.raises(SwayJsonExportError, match="pip install 'dlm\\[sway\\]'"):
            write_sway_json(dlm_path, tmp_path / "export")

    def test_autogen_failure_wrapped_in_typed_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sway-side error during build_spec_dict wraps to
        SwayJsonExportError so the dlm CLI sees a familiar exception
        family."""
        _install_fake_dlm_sway(monkeypatch)
        from dlm_sway.integrations.dlm import (
            autogen as fake_autogen,  # type: ignore[import-not-found]
        )

        def _raise(*_a: object, **_kw: object) -> dict[str, object]:
            raise RuntimeError("intentional autogen blowup for test")

        monkeypatch.setattr(fake_autogen, "build_spec_dict", _raise)

        dlm_path = tmp_path / "doc.dlm"
        dlm_path.write_text("---\ndlm_id: 01\n---\n", encoding="utf-8")
        with pytest.raises(SwayJsonExportError, match="intentional autogen blowup"):
            write_sway_json(dlm_path, tmp_path / "export")
