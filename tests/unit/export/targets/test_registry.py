"""Sprint 41 substrate — export target registry + ollama wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm.export.dispatch import DispatchResult
from dlm.export.errors import UnknownExportTargetError
from dlm.export.targets import TARGETS, ExportTarget, available_targets, resolve_target


class TestRegistry:
    def test_ollama_target_is_registered(self) -> None:
        target = resolve_target("ollama")
        assert target.name == "ollama"
        assert isinstance(target, ExportTarget)
        assert TARGETS["ollama"] is target
        assert "llama-server" in TARGETS
        assert available_targets() == ("ollama", "llama-server")

    def test_unknown_target_lists_available_targets(self) -> None:
        with pytest.raises(
            UnknownExportTargetError,
            match="available targets: ollama, llama-server",
        ):
            resolve_target("vllm")


class TestOllamaWrapper:
    def test_prepare_passthrough_preserves_paths_and_banner_lines(self, tmp_path: Path) -> None:
        export_dir = tmp_path / "exports" / "Q4_K_M"
        export_dir.mkdir(parents=True)
        manifest_path = export_dir / "export_manifest.json"
        base = export_dir / "base.Q4_K_M.gguf"
        adapter = export_dir / "adapter.gguf"
        ctx = DispatchResult(
            export_dir=export_dir,
            manifest_path=manifest_path,
            artifacts=[base, adapter],
            banner_lines=["[green]export:[/green] ok"],
        )

        target = resolve_target("ollama")
        prepared = target.prepare(ctx)
        smoke = target.smoke_test(prepared)

        assert prepared.name == "ollama"
        assert prepared.export_dir == export_dir
        assert prepared.manifest_path == manifest_path
        assert prepared.artifacts == (base, adapter)
        assert prepared.extras["banner_lines"] == ["[green]export:[/green] ok"]
        assert target.launch_command(prepared) == []
        assert smoke.attempted is False
        assert smoke.ok is True
