"""Direct tests for `dlm.inference.dispatch:run_prompt`."""

from __future__ import annotations

from typing import Any

import pytest

from dlm.inference.dispatch import PromptRequest, PromptResult, run_prompt


class _FakeBackend:
    def __init__(self) -> None:
        self.loaded_with: dict[str, Any] | None = None
        self.generate_with: dict[str, Any] | None = None

    def load(self, spec: object, store: object, *, adapter_name: str | None = None) -> None:
        self.loaded_with = {"spec": spec, "store": store, "adapter_name": adapter_name}

    def generate(self, query: str, **kwargs: object) -> str:
        self.generate_with = {"query": query, **kwargs}
        return "fake response"

    def unload(self) -> None:
        pass


def test_run_prompt_loads_backend_and_returns_typed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    monkeypatch.setattr(
        "dlm.inference.backends.build_backend",
        lambda *args, **kwargs: backend,
    )

    spec_obj = object()
    caps_obj = object()
    store_obj = object()

    request = PromptRequest(
        spec=spec_obj,  # type: ignore[arg-type]
        capabilities=caps_obj,  # type: ignore[arg-type]
        store=store_obj,  # type: ignore[arg-type]
        backend_name="pytorch",
        query="hello there",
        max_new_tokens=42,
        temperature=0.5,
        top_p=0.9,
        adapter="my-adapter",
    )

    result = run_prompt(request)

    assert isinstance(result, PromptResult)
    assert result.response == "fake response"
    assert result.backend_name == "pytorch"

    assert backend.loaded_with == {
        "spec": spec_obj,
        "store": store_obj,
        "adapter_name": "my-adapter",
    }
    assert backend.generate_with == {
        "query": "hello there",
        "max_new_tokens": 42,
        "temperature": 0.5,
        "top_p": 0.9,
    }
