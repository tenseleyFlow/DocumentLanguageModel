"""Coverage-oriented tests for train/prompt/repl command bodies."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

from dlm.cli.app import app


def _init_doc(tmp_path: Path, *, base: str = "smollm2-135m") -> Path:
    doc = tmp_path / "doc.dlm"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_path / "home"),
            "init",
            str(doc),
            "--base",
            base,
        ],
    )
    assert result.exit_code == 0, result.output
    return doc


def _fake_doctor_result() -> object:
    return SimpleNamespace(plan=object(), capabilities=object())


class TestTrainCommandCoverage:
    def test_train_success_prints_phase_summary(self, tmp_path: Path, monkeypatch: Any) -> None:
        doc = _init_doc(tmp_path)
        runner = CliRunner()

        fake_result = SimpleNamespace(
            adapter_version=1,
            steps=3,
            seed=42,
            determinism=SimpleNamespace(class_="strict"),
            adapter_path=tmp_path / "adapter",
            log_path=tmp_path / "train.jsonl",
            final_train_loss=0.125,
        )
        fake_phase = SimpleNamespace(phase="sft", result=fake_result)

        monkeypatch.setattr("dlm.hardware.doctor", lambda **kwargs: _fake_doctor_result())
        monkeypatch.setattr("dlm.train.distributed.detect_world_size", lambda: 1)
        monkeypatch.setattr(
            "dlm.train.preference.phase_orchestrator.run_phases",
            lambda *args, **kwargs: [fake_phase],
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "train", str(doc), "--max-steps", "3"],
        )
        assert result.exit_code == 0, result.output
        assert "sft:" in result.output
        assert "adapter:" in result.output
        assert "0.125" in result.output

    def test_train_watch_with_rpc_starts_server(self, tmp_path: Path, monkeypatch: Any) -> None:
        doc = _init_doc(tmp_path)
        runner = CliRunner()

        fake_result = SimpleNamespace(
            adapter_version=1,
            steps=1,
            seed=7,
            determinism=SimpleNamespace(class_="strict"),
            adapter_path=tmp_path / "adapter",
            log_path=tmp_path / "train.jsonl",
            final_train_loss=None,
        )
        fake_phase = SimpleNamespace(phase="sft", result=fake_result)

        class _FakeQueue:
            capacity = 123

            def drain(self) -> list[object]:
                return []

        class _FakeServer:
            def __init__(self, *, host: str, port: int, token: str, queue: object) -> None:
                self.address = (host, port)

            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

        monkeypatch.setenv("DLM_PROBE_TOKEN", "secret")
        monkeypatch.setattr("dlm.hardware.doctor", lambda **kwargs: _fake_doctor_result())
        monkeypatch.setattr("dlm.train.distributed.detect_world_size", lambda: 1)
        monkeypatch.setattr(
            "dlm.train.preference.phase_orchestrator.run_phases",
            lambda *args, **kwargs: [fake_phase],
        )
        monkeypatch.setattr("dlm.train.inject.InjectedProbeQueue", _FakeQueue)
        monkeypatch.setattr("dlm.train.rpc.ProbeRpcServer", _FakeServer)
        monkeypatch.setattr("dlm.watch.loop.run_watch", lambda *args, **kwargs: 0)

        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "train",
                str(doc),
                "--watch",
                "--listen-rpc",
                "127.0.0.1:7777",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "rpc:" in result.output
        assert "watch:" in result.output

    def test_train_noop_watch_repl_and_bounded_rpc_refusals(
        self,
        tmp_path: Path,
        monkeypatch: Any,
    ) -> None:
        doc = _init_doc(tmp_path)
        runner = CliRunner()

        monkeypatch.setattr("dlm.hardware.doctor", lambda **kwargs: _fake_doctor_result())
        monkeypatch.setattr("dlm.train.distributed.detect_world_size", lambda: 1)
        monkeypatch.setattr(
            "dlm.train.preference.phase_orchestrator.run_phases",
            lambda *args, **kwargs: [],
        )

        no_op = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "train", str(doc)],
        )
        assert no_op.exit_code == 0, no_op.output
        assert "nothing to train" in no_op.output

        fake_result = SimpleNamespace(
            adapter_version=1,
            steps=1,
            seed=42,
            determinism=SimpleNamespace(class_="strict"),
            adapter_path=tmp_path / "adapter",
            log_path=tmp_path / "train.jsonl",
            final_train_loss=None,
        )
        fake_phase = SimpleNamespace(phase="sft", result=fake_result)
        monkeypatch.setattr(
            "dlm.train.preference.phase_orchestrator.run_phases",
            lambda *args, **kwargs: [fake_phase],
        )

        watch_repl = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "train", str(doc), "--watch", "--repl"],
        )
        assert watch_repl.exit_code == 2, watch_repl.output
        assert "not yet implemented" in watch_repl.output

        monkeypatch.setenv("DLM_PROBE_TOKEN", "secret")
        bounded_rpc = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "train",
                str(doc),
                "--listen-rpc",
                "127.0.0.1:7777",
                "--max-cycles",
                "1",
            ],
        )
        assert bounded_rpc.exit_code == 2, bounded_rpc.output
        assert "--watch for now" in bounded_rpc.output

    def test_multi_gpu_helper_and_strip(self, monkeypatch: Any) -> None:
        from rich.console import Console

        from dlm.cli.commands import _maybe_dispatch_multi_gpu, _strip_gpus_from_argv
        from dlm.train.distributed import UnsupportedGpuSpecError

        class _GpuSpec:
            def __init__(self, device_ids: tuple[int, ...]) -> None:
                self._device_ids = device_ids

            def resolve(self, device_count: int) -> tuple[int, ...]:
                return self._device_ids

        console = Console(stderr=True)
        monkeypatch.setattr(
            "dlm.train.distributed.parse_gpus",
            lambda raw: (_ for _ in ()).throw(UnsupportedGpuSpecError("bad gpus")),
        )
        assert _maybe_dispatch_multi_gpu("bogus", ["dlm", "train"], console) == 2

        monkeypatch.setattr("dlm.train.distributed.parse_gpus", lambda raw: _GpuSpec((0,)))
        import torch

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        assert _maybe_dispatch_multi_gpu("1", ["dlm", "train"], console) is None

        launched: dict[str, object] = {}
        monkeypatch.setattr("dlm.train.distributed.parse_gpus", lambda raw: _GpuSpec((1, 3)))
        monkeypatch.setattr(
            "dlm.train.distributed.launch_multi_gpu",
            lambda device_ids, cli_args, mixed_precision="bf16": (
                launched.update(
                    {
                        "device_ids": device_ids,
                        "cli_args": cli_args,
                        "mixed_precision": mixed_precision,
                    }
                )
                or 17
            ),
        )
        exit_code = _maybe_dispatch_multi_gpu(
            "1,3",
            ["dlm", "train", "doc.dlm", "--gpus", "1,3"],
            console,
        )
        assert exit_code == 17
        assert launched["device_ids"] == (1, 3)
        assert launched["cli_args"] == ["train", "doc.dlm"]
        assert _strip_gpus_from_argv(["dlm", "train", "--gpus=0,1", "doc.dlm"]) == [
            "train",
            "doc.dlm",
        ]

    def test_train_error_mappings(self, tmp_path: Path, monkeypatch: Any) -> None:
        doc = _init_doc(tmp_path)
        runner = CliRunner()

        from dlm.lock.errors import LockValidationError
        from dlm.train.errors import DiskSpaceError, OOMError, ResumeIntegrityError, TrainingError
        from dlm.train.preference.errors import (
            DpoPhaseError,
            NoPreferenceContentError,
            PriorAdapterRequiredError,
        )

        monkeypatch.setattr("dlm.hardware.doctor", lambda **kwargs: _fake_doctor_result())
        monkeypatch.setattr("dlm.train.distributed.detect_world_size", lambda: 1)

        cases = [
            (
                LockValidationError(path=tmp_path / "dlm.lock", reasons=["torch drift"]),
                "Re-run with",
            ),
            (DiskSpaceError(required_bytes=2_000_000_000, free_bytes=1_000_000_000), "disk:"),
            (ResumeIntegrityError("resume mismatch"), "resume:"),
            (NoPreferenceContentError("no preferences"), "dpo:"),
            (PriorAdapterRequiredError("need prior adapter"), "dpo:"),
            (DpoPhaseError("dpo failed"), "dpo:"),
            (TrainingError("trainer failed"), "training:"),
        ]
        for error, needle in cases:
            monkeypatch.setattr(
                "dlm.train.preference.phase_orchestrator.run_phases",
                lambda *args, _error=error, **kwargs: (_ for _ in ()).throw(_error),
            )
            result = runner.invoke(
                app,
                ["--home", str(tmp_path / "home"), "train", str(doc)],
            )
            assert result.exit_code == 1, result.output
            assert needle in result.output

        monkeypatch.setattr(
            "dlm.train.preference.phase_orchestrator.run_phases",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                OOMError(
                    step=5,
                    peak_bytes=2_000,
                    free_at_start_bytes=4_000,
                    current_grad_accum=1,
                    recommended_grad_accum=4,
                )
            ),
        )
        monkeypatch.setattr("dlm.train.format_oom_message", lambda **kwargs: "OOM advice")
        oom = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "train", str(doc)],
        )
        assert oom.exit_code == 1, oom.output
        assert "OOM advice" in oom.output


class TestPromptAndReplCoverage:
    def test_prompt_text_backend_reads_stdin_and_generates(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        doc = _init_doc(tmp_path)
        runner = CliRunner()

        class _FakeBackend:
            def load(self, spec: object, store: object, adapter_name: str | None = None) -> None:
                return None

            def generate(self, query: str, **kwargs: object) -> str:
                return f"reply:{query}"

        monkeypatch.setattr("dlm.hardware.doctor", lambda: SimpleNamespace(capabilities=object()))
        monkeypatch.setattr(
            "dlm.inference.backends.select_backend", lambda *args, **kwargs: "pytorch"
        )
        monkeypatch.setattr(
            "dlm.inference.backends.build_backend", lambda *args, **kwargs: _FakeBackend()
        )

        result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "prompt", str(doc)],
            input="hello from stdin\n",
        )
        assert result.exit_code == 0, result.output
        assert "reply:hello from stdin" in result.output

    def test_repl_success_and_adapter_validation(self, tmp_path: Path, monkeypatch: Any) -> None:
        doc = _init_doc(tmp_path)
        runner = CliRunner()

        adapter_bad = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "repl", str(doc), "--adapter", "knowledge"],
        )
        assert adapter_bad.exit_code == 2, adapter_bad.output
        assert "only valid on multi-adapter" in adapter_bad.output

        class _FakeBackend:
            def __init__(self) -> None:
                self._loaded = SimpleNamespace(tokenizer="tok")

            def load(self, spec: object, store: object, adapter_name: str | None = None) -> None:
                return None

        monkeypatch.setattr("dlm.hardware.doctor", lambda: SimpleNamespace(capabilities=object()))
        monkeypatch.setattr(
            "dlm.inference.backends.select_backend", lambda *args, **kwargs: "pytorch"
        )
        monkeypatch.setattr(
            "dlm.inference.backends.build_backend", lambda *args, **kwargs: _FakeBackend()
        )
        monkeypatch.setattr("dlm.repl.app.run_repl", lambda session, console: 5)

        repl_ok = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "repl", str(doc), "--backend", "pytorch"],
        )
        assert repl_ok.exit_code == 5, repl_ok.output

    def test_repl_error_mappings(self, tmp_path: Path, monkeypatch: Any) -> None:
        doc = _init_doc(tmp_path)
        runner = CliRunner()

        from dlm.base_models.errors import GatedModelError
        from dlm.inference import AdapterNotFoundError
        from dlm.inference.backends.select import UnsupportedBackendError

        original = doc.read_text(encoding="utf-8")
        fm_end = original.find("\n---\n", original.find("---") + 3)
        multi = tmp_path / "multi.dlm"
        multi.write_text(
            original[:fm_end] + "\ntraining:\n  adapters:\n    knowledge: {}\n" + original[fm_end:],
            encoding="utf-8",
        )
        unknown = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "repl", str(multi), "--adapter", "ghost"],
        )
        assert unknown.exit_code == 2, unknown.output
        assert "not declared" in unknown.output

        monkeypatch.setattr(
            "dlm.base_models.resolve",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                GatedModelError("hf/model", "https://license")
            ),
        )
        gated = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "repl", str(doc), "--backend", "pytorch"],
        )
        assert gated.exit_code == 1, gated.output
        assert "run `dlm train --i-accept-license` first" in gated.output

        monkeypatch.setattr("dlm.base_models.resolve", lambda *args, **kwargs: SimpleNamespace())
        monkeypatch.setattr("dlm.hardware.doctor", lambda: SimpleNamespace(capabilities=object()))
        monkeypatch.setattr(
            "dlm.inference.backends.select_backend",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                UnsupportedBackendError("backend not available")
            ),
        )
        unsupported = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "repl", str(doc), "--backend", "pytorch"],
        )
        assert unsupported.exit_code == 2, unsupported.output
        assert "backend not available" in unsupported.output

        class _MissingAdapterBackend:
            def load(self, spec: object, store: object, adapter_name: str | None = None) -> None:
                raise AdapterNotFoundError("missing adapter")

        monkeypatch.setattr(
            "dlm.inference.backends.select_backend", lambda *args, **kwargs: "pytorch"
        )
        monkeypatch.setattr(
            "dlm.inference.backends.build_backend",
            lambda *args, **kwargs: _MissingAdapterBackend(),
        )
        missing = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "repl", str(doc), "--backend", "pytorch"],
        )
        assert missing.exit_code == 1, missing.output
        assert "missing adapter" in missing.output

    def test_prompt_empty_query_and_repl_invalid_backend(
        self,
        tmp_path: Path,
        monkeypatch: Any,
    ) -> None:
        doc = _init_doc(tmp_path)
        runner = CliRunner()

        class _FakeBackend:
            def load(self, spec: object, store: object, adapter_name: str | None = None) -> None:
                return None

        monkeypatch.setattr("dlm.hardware.doctor", lambda: SimpleNamespace(capabilities=object()))
        monkeypatch.setattr(
            "dlm.inference.backends.select_backend", lambda *args, **kwargs: "pytorch"
        )
        monkeypatch.setattr(
            "dlm.inference.backends.build_backend", lambda *args, **kwargs: _FakeBackend()
        )

        prompt_result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "prompt", str(doc)],
            input="",
        )
        assert prompt_result.exit_code == 2, prompt_result.output
        assert "empty query" in prompt_result.output

        repl_result = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "repl", str(doc), "--backend", "bogus"],
        )
        assert repl_result.exit_code == 2, repl_result.output
        assert "--backend must be" in repl_result.output
