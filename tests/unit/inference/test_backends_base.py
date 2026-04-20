"""InferenceBackend Protocol + PyTorchBackend shape."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dlm.inference.backends import InferenceBackend, PyTorchBackend


class TestProtocolShape:
    def test_pytorch_backend_satisfies_protocol(self) -> None:
        caps = MagicMock()
        backend = PyTorchBackend(caps)
        assert isinstance(backend, InferenceBackend)
        assert backend.name == "pytorch"

    def test_generate_before_load_raises(self) -> None:
        backend = PyTorchBackend(MagicMock())
        with pytest.raises(RuntimeError, match="before load"):
            backend.generate("hi")

    def test_unload_is_idempotent(self) -> None:
        backend = PyTorchBackend(MagicMock())
        backend.unload()
        backend.unload()


class TestPyTorchBackendLoadDelegation:
    def test_load_calls_load_for_inference(self) -> None:
        caps = MagicMock()
        base = MagicMock()
        store = MagicMock()

        with patch("dlm.inference.loader.load_for_inference") as m_load:
            loaded = MagicMock()
            m_load.return_value = loaded
            backend = PyTorchBackend(caps)
            backend.load(base, store, adapter_name="knowledge")

            m_load.assert_called_once_with(store, base, caps, adapter_name="knowledge")

    def test_generate_after_load_delegates(self) -> None:
        with (
            patch("dlm.inference.loader.load_for_inference") as m_load,
            patch("dlm.inference.generate.generate") as m_gen,
        ):
            m_loaded = MagicMock()
            m_loaded.model = MagicMock(name="model")
            m_loaded.tokenizer = MagicMock(name="tok")
            m_load.return_value = m_loaded
            m_gen.return_value = "hello world"

            backend = PyTorchBackend(MagicMock())
            backend.load(MagicMock(), MagicMock())
            out = backend.generate("prompt", max_new_tokens=32, temperature=0.0)

            assert out == "hello world"
            args, kwargs = m_gen.call_args
            assert args[0] is m_loaded.model
            assert args[1] is m_loaded.tokenizer
            assert args[2] == "prompt"
            assert kwargs == {"max_new_tokens": 32, "temperature": 0.0}

    def test_unload_clears_loaded(self) -> None:
        with patch("dlm.inference.loader.load_for_inference") as m_load:
            m_load.return_value = MagicMock()
            backend = PyTorchBackend(MagicMock())
            backend.load(MagicMock(), MagicMock())
            backend.unload()
            with pytest.raises(RuntimeError, match="before load"):
                backend.generate("x")


class _ToyBackend:
    """Duck-typed backend that satisfies the Protocol without inheriting."""

    name = "toy"

    def __init__(self) -> None:
        self.loaded = False

    def load(self, base: Any, store: Any, *, adapter_name: str | None = None) -> None:
        self.loaded = True

    def generate(self, prompt: str, **gen_kwargs: Any) -> str:
        return f"toy:{prompt}"

    def unload(self) -> None:
        self.loaded = False


def test_duck_typed_backend_satisfies_protocol() -> None:
    assert isinstance(_ToyBackend(), InferenceBackend)
