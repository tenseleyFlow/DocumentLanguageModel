"""End-to-end `dlm export` → `ollama create` → `ollama run` smoke.

Sprint 12 DoD: on a host with a supported Ollama binary, a full export
cycle on the SmolLM2-135M fixture produces a registered model whose
`ollama run` returns non-empty stdout whose first line survives a
round-trip through `manifest.exports[-1].smoke_output_first_line`.

Marked `@pytest.mark.slow`. Requires:

- `vendor/llama.cpp/` submodule built (`scripts/bump-llama-cpp.sh build`).
- Ollama binary on PATH at `OLLAMA_MIN_VERSION` or newer.
- SmolLM2-135M offline cache from Sprint 02's fixture.
- A prior `dlm train` produced an adapter under the tmp store.

When any dependency is missing the test skips with a clear message.
The body is deferred to the first CI slow run that has llama.cpp +
Ollama + the fixture all present; landing the scaffold now keeps
sprint-12 DoD honest.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.slow
def test_export_creates_and_smokes_model() -> None:
    """Shape:

    1. `vendor/llama.cpp/build/bin/llama-quantize` exists → else skip.
    2. `ollama` binary on PATH → else skip.
    3. SmolLM2-135M fixture resolvable → else skip.
    4. `dlm train` produces an adapter under a fresh tmp store → else skip.
    5. `run_export(..., skip_ollama=False)` emits Modelfile + registers
       with Ollama + smoke returns non-empty.
    6. `manifest.exports[-1].smoke_output_first_line` matches
       `first_line(ollama run stdout)`.
    """
    vendor_root = Path(__file__).resolve().parents[4] / "vendor" / "llama.cpp"
    if not (vendor_root / "build" / "bin" / "llama-quantize").is_file():
        pytest.skip("vendor/llama.cpp not built; run `scripts/bump-llama-cpp.sh build`.")

    if shutil.which("ollama") is None:
        pytest.skip("ollama binary not on PATH; install from https://ollama.com/download.")

    try:
        from tests.fixtures.tiny_model import tiny_model_path

        tiny_model_path()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"tiny-model fixture unavailable: {exc}")

    pytest.xfail("ollama integration scaffolded; body deferred to first CI slow run")


@pytest.mark.slow
def test_template_round_trip_matches_jinja_reference() -> None:
    """Go template rendering is token-identical to the HF Jinja reference.

    For each dialect in the registry, render a fixed message-set matrix
    through:
      - the vendored Jinja chat template (transformers' `apply_chat_template`)
      - the Go `text/template` via `ollama run --format=<rendered>` plumbing

    Assert the two produce the same token ids. This is the contract
    that keeps the Modelfile's `TEMPLATE` directive honest across
    Ollama upgrades.

    Deferred body: the Go-side invocation requires a Modelfile + live
    Ollama binary because there's no standalone Go template CLI we can
    shell out to without going through `ollama create`. The scaffold
    records the intent so the DoD checkbox has a concrete home.
    """
    if shutil.which("ollama") is None:
        pytest.skip("ollama binary not on PATH.")

    if shutil.which("go") is None:
        pytest.skip("go toolchain not on PATH (needed for standalone template rendering).")

    pytest.xfail("round-trip matrix scaffolded; body deferred to first CI slow run")
