"""llama-server smoke wiring."""

from __future__ import annotations

from pathlib import Path

from dlm.export.errors import TargetSmokeError
from dlm.export.targets.base import TargetResult
from dlm.export.targets.llama_server import LLAMA_SERVER_TARGET


def _vendor_tree(tmp_path: Path) -> Path:
    vendor = tmp_path / "vendor" / "llama.cpp"
    (vendor / "build" / "bin").mkdir(parents=True)
    server = vendor / "build" / "bin" / "llama-server"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    server.chmod(0o755)
    return vendor


def _prepared_target(tmp_path: Path) -> TargetResult:
    export_dir = tmp_path / "exports" / "Q4_K_M"
    export_dir.mkdir(parents=True)
    manifest_path = export_dir / "export_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    model = export_dir / "base.Q4_K_M.gguf"
    model.write_bytes(b"base")
    adapter = export_dir / "adapter.gguf"
    adapter.write_bytes(b"adapter")
    template = export_dir / "chat-template.jinja"
    template.write_text("{{ .Prompt }}\n", encoding="utf-8")
    return TargetResult(
        name="llama-server",
        export_dir=export_dir,
        manifest_path=manifest_path,
        artifacts=(model, adapter, template),
        config_path=template,
        extras={
            "model_path": model,
            "adapter_gguf_path": adapter,
            "context_length": 4096,
            "vendor_override": _vendor_tree(tmp_path),
        },
    )


class TestLlamaServerSmoke:
    def test_smoke_uses_absolute_runtime_argv(self, tmp_path: Path, monkeypatch: object) -> None:
        prepared = _prepared_target(tmp_path)
        seen: list[list[str]] = []

        def _fake_smoke(argv: list[str], **_: object) -> str:
            seen.append(list(argv))
            return "server replied"

        monkeypatch.setattr(
            "dlm.export.targets.llama_server.smoke_openai_compat_server", _fake_smoke
        )

        result = LLAMA_SERVER_TARGET.smoke_test(prepared)

        assert result.attempted is True
        assert result.ok is True
        assert result.detail == "server replied"
        assert len(seen) == 1
        argv = seen[0]
        assert argv[0].endswith("llama-server")
        assert "$SCRIPT_DIR" not in " ".join(argv)
        assert str(prepared.extras["model_path"]) in argv
        assert str(prepared.config_path) in argv
        assert str(prepared.extras["adapter_gguf_path"]) in argv
        assert "--host" in argv
        assert "--port" in argv

    def test_smoke_failure_returns_failed_result(self, tmp_path: Path, monkeypatch: object) -> None:
        prepared = _prepared_target(tmp_path)

        def _fake_smoke(argv: list[str], **_: object) -> str:
            _ = argv
            raise TargetSmokeError("boom")

        monkeypatch.setattr(
            "dlm.export.targets.llama_server.smoke_openai_compat_server", _fake_smoke
        )

        result = LLAMA_SERVER_TARGET.smoke_test(prepared)

        assert result.attempted is True
        assert result.ok is False
        assert result.detail == "boom"
