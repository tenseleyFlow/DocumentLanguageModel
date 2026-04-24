"""Optional observability sinks: TensorBoard + W&B."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from dlm.metrics.events import EvalEvent, StepEvent
from dlm.metrics.sinks import (
    TensorBoardSink,
    WandbSink,
    tensorboard_available,
    wandb_available,
)


class _FakeWriter:
    def __init__(self, *, log_dir: str) -> None:
        self.log_dir = log_dir
        self.scalars: list[tuple[str, float, int]] = []
        self.flushed = False
        self.closed = False

    def add_scalar(self, name: str, value: float, step: int) -> None:
        self.scalars.append((name, value, step))

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True


class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, float], int]] = []
        self.finished = False

    def log(self, payload: dict[str, float], *, step: int) -> None:
        self.logged.append((payload, step))

    def finish(self) -> None:
        self.finished = True


class TestAvailabilityProbes:
    def test_tensorboard_available_true_when_spec_exists(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "importlib.util.find_spec",
            lambda name: object() if name == "tensorboard" else None,
        )
        assert tensorboard_available() is True

    def test_tensorboard_available_false_when_spec_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
        assert tensorboard_available() is False

    def test_wandb_available_true_when_spec_exists(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "importlib.util.find_spec",
            lambda name: object() if name == "wandb" else None,
        )
        assert wandb_available() is True

    def test_wandb_available_false_when_spec_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
        assert wandb_available() is False


class TestTensorBoardSink:
    def test_constructor_raises_cleanly_when_tensorboard_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        real_import = __import__

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch.utils.tensorboard":
                raise ImportError("missing tensorboard")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _fake_import)
        with pytest.raises(RuntimeError, match="requires `tensorboard`"):
            TensorBoardSink(tmp_path, run_id=7)

    def test_records_step_eval_and_close(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_module = ModuleType("torch.utils.tensorboard")
        fake_module.SummaryWriter = _FakeWriter  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake_module)

        sink = TensorBoardSink(tmp_path, run_id=7)
        sink.record_step(StepEvent(run_id=7, step=10, loss=1.2, lr=0.01, grad_norm=0.5))
        sink.record_eval(EvalEvent(run_id=7, step=10, val_loss=0.9, perplexity=2.0))
        sink.close()

        writer = sink._writer
        assert sink.log_dir == tmp_path / "tensorboard" / "run_0007"
        assert writer.scalars == [
            ("train/loss", 1.2, 10),
            ("train/lr", 0.01, 10),
            ("train/grad_norm", 0.5, 10),
            ("eval/val_loss", 0.9, 10),
            ("eval/perplexity", 2.0, 10),
        ]
        assert writer.flushed is True
        assert writer.closed is True


class TestWandbSink:
    def test_constructor_raises_cleanly_when_wandb_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        real_import = __import__

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "wandb":
                raise ImportError("missing wandb")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _fake_import)
        with pytest.raises(RuntimeError, match="requires `wandb`"):
            WandbSink(tmp_path, run_id=9, project="dlm")

    def test_records_payloads_and_close(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_run = _FakeRun()
        fake_wandb = ModuleType("wandb")
        fake_wandb.init = lambda **_kwargs: fake_run  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        sink = WandbSink(tmp_path, run_id=9, project="dlm")
        sink.record_step(StepEvent(run_id=9, step=3, loss=1.0, lr=0.02, grad_norm=0.4))
        sink.record_eval(EvalEvent(run_id=9, step=3, val_loss=0.8, perplexity=1.5))
        sink.record_step(StepEvent(run_id=9, step=4, loss=None, lr=None, grad_norm=None))
        sink.record_eval(EvalEvent(run_id=9, step=4, val_loss=None, perplexity=None))
        sink.close()

        assert fake_run.logged == [
            ({"train/loss": 1.0, "train/lr": 0.02, "train/grad_norm": 0.4}, 3),
            ({"eval/val_loss": 0.8, "eval/perplexity": 1.5}, 3),
        ]
        assert fake_run.finished is True
