"""Bit-identity: ``pretokenize_rows`` output matches TRL's ``_tokenize``.

This is the audit's primary guardrail. Sprint 31's deferred polish
section called out the risk of silent label-mask drift if the
pre-tokenize path diverged from SFTTrainer's own path. This test
closes that risk: we load the real SmolLM2-135M tokenizer and feed
the same ``messages`` and ``text`` rows through both paths, then
assert ``input_ids`` match byte-for-byte.

If TRL 1.x ever changes its tokenize semantics (different EOS
handling, new chat-template kwargs, etc.) this test fails first —
cheap early-warning for a change that would otherwise train a subtly
worse adapter in silence.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.online]


def _trl_tokenize(tokenizer: Any, inp: Any) -> list[int]:
    """Invoke TRL's ``SFTTrainer._tokenize`` as a free function.

    The bound method only references ``self._is_vlm`` and
    ``self.chat_template`` for the non-VLM / default-template case we
    care about. A ``SimpleNamespace`` stand-in satisfies both.
    """
    from trl.trainer.sft_trainer import SFTTrainer

    fake_self = SimpleNamespace(_is_vlm=False, chat_template=None)
    result = SFTTrainer._tokenize(fake_self, tokenizer, inp)  # type: ignore[arg-type]
    return list(result["input_ids"])


def test_pretokenize_matches_trl_on_messages(tiny_model_dir: Path) -> None:
    from transformers import AutoTokenizer

    from dlm.train.tokenization import pretokenize_rows

    tokenizer = AutoTokenizer.from_pretrained(str(tiny_model_dir))
    messages = [
        {"role": "user", "content": "What is a lexer?"},
        {"role": "assistant", "content": "A component that turns text into tokens."},
    ]
    row = {"messages": messages, "_dlm_section_id": "a" * 16}

    out, stats = pretokenize_rows([row], tokenizer=tokenizer, sequence_len=512)
    assert stats.cache_misses == 1

    trl_ids = _trl_tokenize(tokenizer, messages)
    assert out[0]["input_ids"] == trl_ids, (
        "pretokenize_rows diverged from TRL _tokenize on a messages row — "
        "label-mask and loss would differ silently from SFTTrainer's path"
    )


def test_pretokenize_matches_trl_on_text(tiny_model_dir: Path) -> None:
    from transformers import AutoTokenizer

    from dlm.train.tokenization import pretokenize_rows

    tokenizer = AutoTokenizer.from_pretrained(str(tiny_model_dir))
    prose = "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n"
    row = {"text": prose, "_dlm_section_id": "b" * 16}

    out, stats = pretokenize_rows([row], tokenizer=tokenizer, sequence_len=512)
    assert stats.cache_misses == 1

    # TRL's language-modeling branch appends EOS *before* tokenizing
    # (see `_prepare_dataset.add_eos`). Replicate exactly.
    eos = tokenizer.eos_token
    text_with_eos = prose if prose.endswith(eos) else prose + eos
    trl_ids = _trl_tokenize(tokenizer, text_with_eos)
    assert out[0]["input_ids"] == trl_ids, (
        "pretokenize_rows diverged from TRL _tokenize on a text row — "
        "EOS handling or tokenizer kwargs drift"
    )


def test_pretokenize_handles_idempotent_eos(tiny_model_dir: Path) -> None:
    """A text row that already ends in EOS must not double-append."""
    from transformers import AutoTokenizer

    from dlm.train.tokenization import pretokenize_rows

    tokenizer = AutoTokenizer.from_pretrained(str(tiny_model_dir))
    eos = tokenizer.eos_token
    prose_with_eos = f"hello world{eos}"
    row = {"text": prose_with_eos, "_dlm_section_id": "c" * 16}

    out, _stats = pretokenize_rows([row], tokenizer=tokenizer, sequence_len=512)
    trl_ids = _trl_tokenize(tokenizer, prose_with_eos)
    assert out[0]["input_ids"] == trl_ids
    # Sanity: tokenizing ``prose_with_eos`` once vs ``prose + eos + eos``
    # would produce different input_ids — confirm we didn't duplicate.
    double_ids = _trl_tokenize(tokenizer, prose_with_eos + eos)
    assert out[0]["input_ids"] != double_ids
