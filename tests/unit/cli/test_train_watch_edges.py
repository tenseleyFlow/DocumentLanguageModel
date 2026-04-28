"""Additional `dlm train` coverage for lock-mode and watch-loop tails."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import dlm.base_models as base_models
from dlm.cli.app import app
from dlm.modality.errors import ModalityError
from dlm.watch.loop import CycleResult


def _write_minimal_dlm(path: Path) -> None:
    path.write_text(
        "---\n"
        "dlm_id: 01TRAINWATCH00000000000000\n"
        "base_model: smollm2-135m\n"
        "training:\n"
        "  seed: 42\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )


def _parsed_doc() -> object:
    return SimpleNamespace(
        frontmatter=SimpleNamespace(
            base_model="smollm2-135m",
            dlm_id="01TRAINWATCH00000000000000",
            training=SimpleNamespace(sequence_len=2048),
        ),
        sections=[SimpleNamespace(content="body")],
    )


def _resolved_spec() -> object:
    return SimpleNamespace(
        key="smollm2-135m",
        revision="0123456789abcdef0123456789abcdef01234567",
        modality="text",
        params=135_000_000,
        effective_context_length=2048,
        requires_acceptance=False,
    )


def _fake_phase_result(tmp_path: Path) -> object:
    result = SimpleNamespace(
        adapter_version=1,
        steps=3,
        seed=42,
        determinism=SimpleNamespace(class_="strict"),
        adapter_path=tmp_path / "adapter",
        log_path=tmp_path / "train.jsonl",
        final_train_loss=0.25,
        final_val_loss=0.1,
    )
    return SimpleNamespace(phase="sft", result=result)


def _install_train_basics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dlm.doc.parser.parse_file", lambda path: _parsed_doc())
    monkeypatch.setattr(base_models, "resolve", lambda *args, **kwargs: _resolved_spec())
    monkeypatch.setattr("dlm.train.distributed.detect_world_size", lambda: 1)
    monkeypatch.setattr(
        "dlm.hardware.doctor",
        lambda **kwargs: SimpleNamespace(plan=object(), capabilities=object()),
    )


class TestTrainLockModeEdges:
    @pytest.mark.parametrize(
        ("flag", "expected"),
        [
            ("--strict-lock", "strict"),
            ("--update-lock", "update"),
            ("--ignore-lock", "ignore"),
        ],
    )
    def test_single_lock_flags_propagate_to_run_phases(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        flag: str,
        expected: str,
    ) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)
        _install_train_basics(monkeypatch)
        captured: dict[str, object] = {}

        def _fake_run_phases(*args: object, **kwargs: object) -> list[object]:
            captured["lock_mode"] = kwargs["lock_mode"]
            return []

        monkeypatch.setattr("dlm.train.preference.phase_orchestrator.run_phases", _fake_run_phases)

        result = CliRunner().invoke(
            app,
            ["--home", str(tmp_path), "train", str(doc), flag],
        )

        assert result.exit_code == 0, result.output
        assert captured["lock_mode"] == expected


class TestTrainWatchEdges:
    def test_modality_error_maps_to_training_prefix(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)
        _install_train_basics(monkeypatch)
        monkeypatch.setattr(
            "dlm.train.preference.phase_orchestrator.run_phases",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ModalityError("processor contract failed")
            ),
        )

        result = CliRunner().invoke(app, ["--home", str(tmp_path), "train", str(doc)])

        assert result.exit_code == 1, result.output
        assert "training:" in result.output
        assert "processor contract failed" in result.output

    def test_watch_rpc_logs_cycle_and_skip_messages(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)
        _install_train_basics(monkeypatch)
        monkeypatch.setenv("DLM_PROBE_TOKEN", "secret")
        fake_phase = _fake_phase_result(tmp_path)

        class _FakeQueue:
            capacity = 5

            def drain(self) -> list[object]:
                return []

        class _FakeServer:
            def __init__(self, *, host: str, port: int, token: str, queue: object) -> None:
                self.address = (host, port)
                self.start_calls = 0
                self.stop_calls = 0

            def start(self) -> None:
                self.start_calls += 1

            def stop(self) -> None:
                self.stop_calls += 1

        def _fake_watch(**kwargs: object) -> int:
            on_cycle = kwargs["on_cycle"]
            on_cycle(
                CycleResult(
                    ran=True,
                    new_sections=1,
                    removed_sections=0,
                    run_result=SimpleNamespace(final_train_loss=0.2, final_val_loss=0.1, steps=4),
                )
            )
            on_cycle(CycleResult(ran=False, new_sections=0, removed_sections=0))
            return 23

        monkeypatch.setattr(
            "dlm.train.preference.phase_orchestrator.run_phases",
            lambda *args, **kwargs: [fake_phase],
        )
        monkeypatch.setattr("dlm.train.inject.InjectedProbeQueue", _FakeQueue)
        monkeypatch.setattr("dlm.train.rpc.ProbeRpcServer", _FakeServer)
        monkeypatch.setattr("dlm.watch.loop.run_watch", _fake_watch)

        result = CliRunner().invoke(
            app,
            [
                "--home",
                str(tmp_path),
                "train",
                str(doc),
                "--watch",
                "--listen-rpc",
                "127.0.0.1:7777",
            ],
        )

        assert result.exit_code == 23, result.output
        normalized = " ".join(result.output.split())
        assert "rpc:" in normalized
        assert "watch:" in normalized
        assert "no new content, skipping retrain" in normalized

    def test_watch_keyboard_interrupt_stops_server_and_exits_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)
        _install_train_basics(monkeypatch)
        monkeypatch.setenv("DLM_PROBE_TOKEN", "secret")
        fake_phase = _fake_phase_result(tmp_path)
        holder: dict[str, object] = {}

        class _FakeQueue:
            capacity = 3

            def drain(self) -> list[object]:
                return []

        class _FakeServer:
            def __init__(self, *, host: str, port: int, token: str, queue: object) -> None:
                self.address = (host, port)
                self.stop_calls = 0
                holder["server"] = self

            def start(self) -> None:
                return None

            def stop(self) -> None:
                self.stop_calls += 1

        monkeypatch.setattr(
            "dlm.train.preference.phase_orchestrator.run_phases",
            lambda *args, **kwargs: [fake_phase],
        )
        monkeypatch.setattr("dlm.train.inject.InjectedProbeQueue", _FakeQueue)
        monkeypatch.setattr("dlm.train.rpc.ProbeRpcServer", _FakeServer)
        monkeypatch.setattr(
            "dlm.watch.loop.run_watch",
            lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
        )

        result = CliRunner().invoke(
            app,
            [
                "--home",
                str(tmp_path),
                "train",
                str(doc),
                "--watch",
                "--listen-rpc",
                "127.0.0.1:7777",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Ctrl-C received, exiting" in result.output
        server = holder["server"]
        assert isinstance(server, _FakeServer)
        assert server.stop_calls == 2
