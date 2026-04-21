"""`dlm train --no-cache` opt-out for the tokenized-section cache."""

from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app


def _write_minimal_dlm(path: Path) -> None:
    path.write_text(
        "---\n"
        "dlm_id: 01TEST0" + "0" * 19 + "\n"
        "base_model: smollm2-135m\n"
        "training:\n"
        "  seed: 42\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )


def test_no_cache_flag_sets_env_var(tmp_path: Path) -> None:
    """The flag sets ``DLM_DISABLE_TOKENIZED_CACHE=1`` early enough
    that the trainer's pre-tokenize helper sees it. The flag is a
    front-door for an existing escape hatch; this test pins the
    contract so a future refactor doesn't silently drop the plumbing.
    """
    doc = tmp_path / "doc.dlm"
    _write_minimal_dlm(doc)

    # Pre-condition: env var cleared. Sub-test may have left it set.
    os.environ.pop("DLM_DISABLE_TOKENIZED_CACHE", None)

    runner = CliRunner()
    # Run to the point where the trainer would actually fire — that's
    # a heavy path. Instead we run a known-error path that still
    # traverses the CLI arg-parsing + env-setting phase. `--resume`
    # + `--fresh` trips the mutex check AFTER our env-setting logic
    # runs (which is before lock-flag validation)… wait no, the
    # no_cache check is AFTER the resume/fresh mutex. Use the
    # lock-flag mutex instead: --strict-lock + --update-lock trips
    # BEFORE the no_cache branch. So use a different tactic: the
    # `--policy invalid` check trips well before trainer dispatch
    # AND after no_cache. But that check is also before no_cache...
    #
    # Simplest: just read the env var immediately after invoking the
    # CLI on an impossible flag combo that trips earlier than trainer
    # dispatch. Use `--no-cache` with `--resume --fresh` — the mutex
    # check is the first failure, AND it runs AFTER no_cache is
    # parsed (Typer resolves defaults before the function body).
    # But we need the no_cache handler to have run. Confirm by
    # reading the source — no_cache is set AFTER the resume/fresh
    # mutex but BEFORE policy validation. So `--policy bad` is the
    # right trigger.
    result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_path),
            "train",
            str(doc),
            "--no-cache",
            "--policy",
            "invalid",
        ],
    )
    assert result.exit_code == 2, result.output
    assert os.environ.get("DLM_DISABLE_TOKENIZED_CACHE") == "1"

    # Cleanup so later tests in the same session see a clean slate.
    os.environ.pop("DLM_DISABLE_TOKENIZED_CACHE", None)


def test_no_cache_flag_help_text(tmp_path: Path) -> None:
    """The flag appears in `dlm train --help` so users can discover it."""
    runner = CliRunner()
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--no-cache" in result.output
    assert "tokenized" in result.output.lower()


def test_no_cache_absent_leaves_env_unset(tmp_path: Path) -> None:
    """Omitting the flag does not set the env var (no side effect on
    the default cache-on path)."""
    doc = tmp_path / "doc.dlm"
    _write_minimal_dlm(doc)
    os.environ.pop("DLM_DISABLE_TOKENIZED_CACHE", None)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_path),
            "train",
            str(doc),
            "--policy",
            "invalid",
        ],
    )
    assert result.exit_code == 2
    # Env var never became set.
    assert "DLM_DISABLE_TOKENIZED_CACHE" not in os.environ
