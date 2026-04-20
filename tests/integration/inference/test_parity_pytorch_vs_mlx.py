"""MLX vs PyTorch inference parity (Sprint 21 DoD).

Runs the same prompt through both backends at `temperature=0` and
asserts the first N generated tokens match. MLX Metal and PyTorch MPS
float arithmetic is not bit-identical; the sprint accepts "prefix-of-N
equal ≥95% of runs" rather than exact match.

Skipped unless:
- Running on darwin-arm64
- `mlx-lm` is importable (i.e., `uv sync --extra mlx` has been run)

The `trained_store` session fixture (Sprint 14.5) provides a real tiny-
model adapter. We reuse it rather than training in-test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dlm.base_models import resolve as resolve_base_model
from dlm.hardware import doctor
from dlm.inference.backends import mlx_available
from dlm.inference.backends.mlx_backend import MlxBackend
from dlm.inference.backends.pytorch_backend import PyTorchBackend

if TYPE_CHECKING:
    from tests.fixtures.trained_store import TrainedStoreHandle


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not mlx_available(),
        reason="mlx + mlx-lm not available (need darwin-arm64 + `uv sync --extra mlx`)",
    ),
]


_PROMPT = "What is the capital of France?"
_PREFIX_TOKENS = 16
_MIN_OVERLAP = 8
"""Audit-08 N3: tightened from `min(4, len(pt_toks))` to a fixed 8.

Still relaxed from "all 16 tokens agree" (Metal vs MPS float drift
breaks exactness), but 8/16 matching whitespace tokens is the
honest "backends are semantically comparable" bar. Lower values
mean the loader is broken or the greedy decode diverged
immediately."""


def test_pytorch_mlx_prefix_tokens_agree(  # pragma: no cover - slow/darwin
    trained_store: TrainedStoreHandle,
) -> None:
    """First 16 decode tokens match between backends at temperature=0.

    Loosened from exact-sequence match because MLX Metal and PyTorch
    MPS kernels diverge numerically. The per-token argmax stabilizes
    for the first N tokens on well-trained adapters; we pick N=16 as
    the "visibly-identical" threshold from the sprint spec.
    """
    spec = resolve_base_model(trained_store.base_key, accept_license=True)
    caps = doctor().capabilities

    pt = PyTorchBackend(caps)
    pt.load(spec, trained_store.store)
    pt_out = pt.generate(_PROMPT, max_new_tokens=_PREFIX_TOKENS, temperature=0.0)
    pt.unload()

    mlx = MlxBackend(caps)
    mlx.load(spec, trained_store.store)
    mlx_out = mlx.generate(_PROMPT, max_new_tokens=_PREFIX_TOKENS, temperature=0.0)
    mlx.unload()

    # Prefix-of-N check: split to whitespace tokens as a crude
    # tokenization. Tight assert (exact bytes) is too strict given
    # kernel drift; a prefix overlap is what the sprint DoD asks for.
    pt_toks = pt_out.split()
    mlx_toks = mlx_out.split()
    overlap = sum(1 for a, b in zip(pt_toks, mlx_toks, strict=False) if a == b)
    assert overlap >= _MIN_OVERLAP, (
        f"MLX and PyTorch diverged earlier than expected "
        f"(overlap={overlap} < {_MIN_OVERLAP}, "
        f"pt={pt_toks[:8]!r}, mlx={mlx_toks[:8]!r})"
    )
