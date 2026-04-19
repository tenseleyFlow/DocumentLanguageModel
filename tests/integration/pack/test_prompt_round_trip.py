"""Pack → unpack → `dlm prompt` produces byte-identical output (Sprint 14).

Sprint 14 DoD §1: "pack a tiny-model store → delete original → unpack →
`dlm prompt` works and produces the same output as before pack". The
structural round-trip check lives in `test_round_trip.py`; this test
owns the prompt-output invariant at `temperature=0` (greedy decoding,
deterministic when weights are bit-identical).

Doesn't destroy the shared `trained_store.home` — other session tests
still need it. Instead, unpacks into a fresh home and compares
generations from the two live stores side by side.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

pytestmark = pytest.mark.slow


_PROMPT_QUERY = "hello"
_PROMPT_MAX_TOKENS = 32


def _run_prompt(home: Path, doc: Path) -> str:
    """Invoke `dlm prompt --temp 0` under the given home. Return stdout."""
    from dlm.cli.app import app

    os.environ["DLM_HOME"] = str(home)
    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(
        app,
        [
            "prompt",
            str(doc),
            _PROMPT_QUERY,
            "--temp",
            "0",
            "--max-tokens",
            str(_PROMPT_MAX_TOKENS),
        ],
    )
    assert result.exit_code == 0, f"prompt failed: {result.stderr}"
    return result.stdout


@pytest.mark.slow
def test_pack_unpack_prompt_is_byte_identical(trained_store, tmp_path: Path) -> None:
    """Pre-pack prompt output == post-unpack prompt output (greedy decode)."""
    from dlm.cli.app import app

    offline_vars = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    saved = {k: os.environ.pop(k, None) for k in offline_vars}

    try:
        pre = _run_prompt(trained_store.home, trained_store.doc)

        # Pack under the original home.
        pack_path = tmp_path / "round-trip.pack"
        runner = CliRunner()
        pack_result = runner.invoke(
            app,
            [
                "pack",
                str(trained_store.doc),
                "--out",
                str(pack_path),
            ],
        )
        assert pack_result.exit_code == 0, pack_result.output
        assert pack_path.is_file()

        # Unpack into a fresh home. The original store stays intact so
        # other session-scoped tests aren't affected.
        fresh_home = tmp_path / "restored-home"
        fresh_home.mkdir()
        restored_dir = tmp_path / "restored-doc"
        restored_dir.mkdir()
        os.environ["DLM_HOME"] = str(fresh_home)
        unpack_result = runner.invoke(
            app,
            [
                "unpack",
                str(pack_path),
                "--out",
                str(restored_dir),
            ],
        )
        assert unpack_result.exit_code == 0, unpack_result.output

        restored_doc = restored_dir / trained_store.doc.name
        assert restored_doc.is_file(), f"no restored doc at {restored_doc}"
        post = _run_prompt(fresh_home, restored_doc)

        assert pre == post, (
            f"pack round-trip broke prompt output at temp=0.\npre:  {pre!r}\npost: {post!r}"
        )
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
