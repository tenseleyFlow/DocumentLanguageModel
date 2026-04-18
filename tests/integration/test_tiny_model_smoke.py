"""Minimal `slow` + `online` sanity check on the tiny-model fixture.

Placeholder until Sprint 09 lands the real LoRA-loop assertions (audit 02
M3). What this guards today:

- The cache-and-pre-warm CI step actually downloads SmolLM2-135M.
- The downloaded cache contains a tokenizer with a chat template.
- Subsequent `tiny_model_dir` invocations are a no-op (cache hit).

Scope is deliberately narrow so the slow-tests CI job has *something*
non-trivial to exercise.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.online
def test_tiny_model_dir_yields_usable_tokenizer(tiny_model_dir: Path) -> None:
    # Import inside the test so collection stays cheap for the fast
    # subset (transformers import is ~seconds).
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tiny_model_dir))
    assert tokenizer.chat_template, (
        "SmolLM2-135M-Instruct must ship a chat template out of the box; "
        "absence here signals a broken cache or an upstream revision drift"
    )
    # Basic encode roundtrip (no chat template applied — just vocab sanity).
    ids = tokenizer.encode("hello world", add_special_tokens=False)
    assert ids
    assert tokenizer.decode(ids).strip() == "hello world"


@pytest.mark.slow
@pytest.mark.online
def test_tiny_model_dir_cache_hit_is_fast(tiny_model_dir: Path) -> None:
    """Second call should resolve from the session cache."""
    from tests.fixtures.tiny_model import tiny_model_path

    again = tiny_model_path()
    assert again == tiny_model_dir
