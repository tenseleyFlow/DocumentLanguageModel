"""CLI coverage for VL export when GGUF context is unavailable."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

from dlm.cli.app import app


def _scaffold_vl_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "vl.dlm"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_path / "home"),
            "init",
            str(doc),
            "--base",
            "paligemma-3b-mix-224",
            "--multimodal",
            "--i-accept-license",
        ],
    )
    assert result.exit_code == 0, result.output
    return doc


class _FakeVlDispatch:
    accepts_images = True
    accepts_audio = False

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def dispatch_export(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            export_dir=Path("/tmp/hf-snapshot"),
            manifest_path=Path("/tmp/hf-snapshot/snapshot_manifest.json"),
            artifacts=[],
            banner_lines=["[green]export:[/green] HF snapshot fallback reached"],
        )


def test_vl_export_without_cached_base_uses_snapshot_dispatch(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    doc = _scaffold_vl_doc(tmp_path)
    fake_dispatch = _FakeVlDispatch()

    def _raise_cache_miss(*_: object, **__: object) -> object:
        raise RuntimeError("google/paligemma-3b-mix-224 not found in local cache")

    monkeypatch.setattr("dlm.base_models.download_spec", _raise_cache_miss)
    monkeypatch.setattr("dlm.modality.modality_for", lambda _spec: fake_dispatch)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--home", str(tmp_path / "home"), "export", str(doc)],
    )

    assert result.exit_code == 0, result.output
    assert len(fake_dispatch.calls) == 1
    assert fake_dispatch.calls[0]["gguf_emission_context"] is None
    assert "HF snapshot fallback reached" in result.output
    assert "base model not in local cache" not in result.output
