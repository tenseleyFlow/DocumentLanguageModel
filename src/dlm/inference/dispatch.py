"""Domain dispatcher for `dlm prompt` (text path).

Lifts the build-backend → load → generate pipeline out of the CLI for
text-only bases. Callers (CLI, LSP inline-preview, future automation)
build a `PromptRequest`, call `run_prompt`, and render the typed
`PromptResult`. The dispatcher does no console I/O nor stdin reads;
the CLI resolves the query string from argv or stdin before dispatch.

Vision-language and audio-language paths still live in CLI helpers
(`_dispatch_vl_prompt`, `_dispatch_audio_prompt`); a follow-up phase
lifts those into modality-aware dispatchers under
`dlm.inference.dispatch_vl` / `dispatch_audio`.

External-module imports are dotted (e.g. `from dlm.inference import
backends as _backends; _backends.build_backend(...)`) so test fixtures
that monkeypatch `dlm.inference.backends.<name>` resolve at call time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dlm.inference import backends as _backends
from dlm.inference.backends.select import BackendName

if TYPE_CHECKING:
    from dlm.base_models.schema import BaseModelSpec
    from dlm.hardware.capabilities import Capabilities
    from dlm.store.paths import StorePath


@dataclass(frozen=True)
class PromptRequest:
    """Inputs to `run_prompt`.

    The CLI is responsible for selecting the backend kind (`auto` →
    `pytorch` / `mlx`), license-checking the spec, and resolving the
    query string from argv or stdin; the dispatcher receives all of
    those as already-typed objects.
    """

    spec: BaseModelSpec
    capabilities: Capabilities
    store: StorePath
    backend_name: BackendName
    query: str
    max_new_tokens: int
    temperature: float
    top_p: float | None
    adapter: str | None


@dataclass(frozen=True)
class PromptResult:
    """Outcome of `run_prompt`. The CLI writes `response` to stdout."""

    response: str
    backend_name: BackendName


def run_prompt(req: PromptRequest) -> PromptResult:
    """Build, load, and generate a single response for a text-only base."""
    backend_obj = _backends.build_backend(req.backend_name, req.capabilities)
    backend_obj.load(req.spec, req.store, adapter_name=req.adapter)
    response = backend_obj.generate(
        req.query,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return PromptResult(response=response, backend_name=req.backend_name)
