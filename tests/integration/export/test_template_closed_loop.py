"""Sprint 12.6 — Go↔Jinja token-identity closed-loop integration test.

Two tests live here, both marked `@pytest.mark.slow`:

1. `test_hf_goldens_reproduce` — re-runs `apply_chat_template` against
   the same tokenizer the refresh script targets. If a future HF revision
   drifts the template output, this test fails fast in CI before the
   Ollama-side round trip ever gets its turn.

2. `test_closed_loop_go_vs_jinja_chatml` — the real closed-loop check.
   Requires `ollama` on PATH + a tiny chatml model registered under
   `OLLAMA_NAME`. On CI, the weekly `weekly-template-drift.yml` workflow
   handles registration via the standard export pipeline; for local
   devs, `OLLAMA_NAME` can point at a manually-registered model.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from dlm.export.ollama.verify import verify_token_count

pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHATML_GOLDENS_DIR = _REPO_ROOT / "tests" / "golden" / "chat-templates" / "chatml"


def _load_chatml_goldens() -> list[dict[str, object]]:
    if not _CHATML_GOLDENS_DIR.is_dir():
        return []
    out: list[dict[str, object]] = []
    for path in sorted(_CHATML_GOLDENS_DIR.glob("*.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


@pytest.mark.slow
def test_hf_goldens_reproduce() -> None:
    """Jinja side: HF tokenizer reproduces every chatml golden exactly.

    This is the cheaper half of the closed-loop — it only needs the
    tokenizer + an offline HF cache. Runs in the weekly workflow before
    the Ollama-side test so a template drift upstream fails with a
    clear signal before we burn minutes on `ollama pull` + registration.
    """
    goldens = _load_chatml_goldens()
    if not goldens:
        pytest.skip("no chatml goldens on disk; run refresh-chat-template-goldens.py")

    try:
        from tests.fixtures.tiny_model import tiny_model_path
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"tiny-model fixture unavailable: {exc}")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(tiny_model_path()),
            use_fast=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        pytest.skip(f"could not load tiny-model tokenizer: {exc}")

    for golden in goldens:
        rendered = tokenizer.apply_chat_template(
            golden["messages"],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        actual = len(rendered)
        recorded = golden["expected_hf_token_count"]
        assert actual == recorded, (
            f"chatml/{golden['scenario']}: HF re-render={actual}, "
            f"golden={recorded}. Template drift upstream? Regenerate via "
            "scripts/refresh-chat-template-goldens.py after reviewing."
        )


@pytest.mark.slow
def test_closed_loop_go_vs_jinja_chatml() -> None:
    """Full closed-loop: Ollama `prompt_eval_count` == HF `apply_chat_template` len.

    Expects `OLLAMA_NAME` in the environment to point at a registered
    chatml model. The weekly CI workflow sets this after running
    `dlm export` on the tiny-model fixture.
    """
    if shutil.which("ollama") is None:
        pytest.skip("ollama binary not on PATH.")

    ollama_name = os.environ.get("OLLAMA_NAME")
    if not ollama_name:
        pytest.skip("OLLAMA_NAME not set; weekly workflow or local export registers one.")

    goldens = _load_chatml_goldens()
    if not goldens:
        pytest.skip("no chatml goldens on disk.")

    try:
        from tests.fixtures.tiny_model import tiny_model_path
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(tiny_model_path()),
            use_fast=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        pytest.skip(f"tokenizer setup failed: {exc}")

    for golden in goldens:
        verify_token_count(
            ollama_name=ollama_name,
            hf_tokenizer=tokenizer,
            messages=golden["messages"],
            scenario=f"chatml/{golden['scenario']}",
        )
