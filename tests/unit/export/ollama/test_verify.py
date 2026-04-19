"""Go↔Jinja closed-loop harness (Sprint 12.6) — offline unit tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from dlm.export.ollama.errors import VerificationError
from dlm.export.ollama.verify import (
    parse_prompt_eval_count,
    run_with_telemetry,
    verify_token_count,
)

# --- parse_prompt_eval_count ------------------------------------------------


class TestParsePromptEvalCount:
    def test_compact_json_line(self) -> None:
        stderr = (
            'loading model...\n{"total_duration": 1.2, "prompt_eval_count": 42, "eval_count": 5}\n'
        )
        assert parse_prompt_eval_count(stderr) == 42

    def test_last_json_line_wins_over_earlier(self) -> None:
        """Multiple summary lines — the final one is the real one."""
        stderr = '{"prompt_eval_count": 99}\n{"prompt_eval_count": 42, "eval_count": 5}\n'
        assert parse_prompt_eval_count(stderr) == 42

    def test_space_separated_summary(self) -> None:
        """Older Ollama uses `prompt eval count:  42` label pairs."""
        stderr = (
            "total duration:       1.2s\n"
            "prompt eval count:    42 token(s)\n"
            "eval count:           5 token(s)\n"
        )
        assert parse_prompt_eval_count(stderr) == 42

    def test_malformed_json_skipped_then_space_separated(self) -> None:
        stderr = (
            "{this is not valid JSON but mentions prompt_eval_count\n"
            "prompt eval count: 7 token(s)\n"
        )
        assert parse_prompt_eval_count(stderr) == 7

    def test_no_counter_raises(self) -> None:
        with pytest.raises(VerificationError, match="telemetry parse"):
            parse_prompt_eval_count("random output\nno counter here\n")

    def test_empty_stderr_raises(self) -> None:
        with pytest.raises(VerificationError, match="telemetry parse"):
            parse_prompt_eval_count("")


# --- run_with_telemetry -----------------------------------------------------


def _success_proc(stderr_payload: str) -> Any:
    def runner(_argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=_argv,
            returncode=0,
            stdout="hello, I am an LLM.\n",
            stderr=stderr_payload,
        )

    return runner


class TestRunWithTelemetry:
    def test_success_returns_count(self, tmp_path: Path) -> None:
        # Fake binary on disk so `locate_ollama` isn't hit.
        binary = tmp_path / "ollama-bin"
        binary.write_text("# mock")
        runner = _success_proc('{"prompt_eval_count": 17, "eval_count": 3}\n')
        assert (
            run_with_telemetry(
                ollama_name="demo:latest",
                prompt="hi",
                runner=runner,
                binary=binary,
            )
            == 17
        )

    def test_nonzero_exit_raises(self, tmp_path: Path) -> None:
        binary = tmp_path / "ollama-bin"
        binary.write_text("# mock")

        def runner(_argv: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=_argv,
                returncode=1,
                stdout="",
                stderr="model not found",
            )

        with pytest.raises(VerificationError, match="ollama run exited 1"):
            run_with_telemetry(
                ollama_name="demo:latest",
                prompt="hi",
                runner=runner,
                binary=binary,
            )

    def test_passes_name_and_verbose_to_argv(self, tmp_path: Path) -> None:
        binary = tmp_path / "ollama-bin"
        binary.write_text("# mock")
        captured: dict[str, list[str]] = {}

        def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
            captured["argv"] = argv
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout="",
                stderr='{"prompt_eval_count": 1}',
            )

        run_with_telemetry(
            ollama_name="demo:latest",
            prompt="hello",
            runner=runner,
            binary=binary,
        )
        argv = captured["argv"]
        assert "run" in argv
        assert "demo:latest" in argv
        assert "--verbose" in argv
        assert "hello" in argv


# --- verify_token_count -----------------------------------------------------


class _FakeTokenizer:
    """Stand-in for `transformers.PreTrainedTokenizerBase`."""

    def __init__(self, token_count: int) -> None:
        self._count = token_count
        self.last_messages: list[dict[str, str]] | None = None
        self.last_kwargs: dict[str, object] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> list[int]:
        self.last_messages = messages
        self.last_kwargs = kwargs
        return list(range(self._count))


class TestVerifyTokenCount:
    def test_matching_counts_pass(self, tmp_path: Path) -> None:
        binary = tmp_path / "ollama-bin"
        binary.write_text("# mock")
        runner = _success_proc(json.dumps({"prompt_eval_count": 12}))
        tokenizer = _FakeTokenizer(token_count=12)

        verify_token_count(
            ollama_name="demo:latest",
            hf_tokenizer=tokenizer,
            messages=[{"role": "user", "content": "hi"}],
            runner=runner,
            binary=binary,
        )

    def test_mismatched_counts_raise(self, tmp_path: Path) -> None:
        binary = tmp_path / "ollama-bin"
        binary.write_text("# mock")
        runner = _success_proc(json.dumps({"prompt_eval_count": 11}))
        tokenizer = _FakeTokenizer(token_count=12)

        with pytest.raises(VerificationError, match="template drift") as excinfo:
            verify_token_count(
                ollama_name="demo:latest",
                hf_tokenizer=tokenizer,
                messages=[{"role": "user", "content": "hi"}],
                runner=runner,
                binary=binary,
            )
        assert excinfo.value.hf_count == 12
        assert excinfo.value.go_count == 11

    def test_scenario_appears_in_error(self, tmp_path: Path) -> None:
        binary = tmp_path / "ollama-bin"
        binary.write_text("# mock")
        runner = _success_proc(json.dumps({"prompt_eval_count": 11}))
        tokenizer = _FakeTokenizer(token_count=12)

        with pytest.raises(VerificationError, match="multi-turn") as excinfo:
            verify_token_count(
                ollama_name="demo:latest",
                hf_tokenizer=tokenizer,
                messages=[{"role": "user", "content": "hi"}],
                runner=runner,
                binary=binary,
                scenario="multi-turn",
            )
        assert excinfo.value.scenario == "multi-turn"

    def test_tokenizer_called_with_generation_prompt(self, tmp_path: Path) -> None:
        """HF must be asked with `add_generation_prompt=True` to match Ollama's count."""
        binary = tmp_path / "ollama-bin"
        binary.write_text("# mock")
        runner = _success_proc(json.dumps({"prompt_eval_count": 5}))
        tokenizer = _FakeTokenizer(token_count=5)

        verify_token_count(
            ollama_name="demo:latest",
            hf_tokenizer=tokenizer,
            messages=[{"role": "user", "content": "hi"}],
            runner=runner,
            binary=binary,
        )
        assert tokenizer.last_kwargs is not None
        assert tokenizer.last_kwargs.get("add_generation_prompt") is True
        assert tokenizer.last_kwargs.get("tokenize") is True
        # Pin list-of-ids return type so HF's BatchEncoding default
        # doesn't silently feed `len()` the number of dict keys.
        assert tokenizer.last_kwargs.get("return_dict") is False

    def test_empty_messages_falls_back_to_hello_prompt(self, tmp_path: Path) -> None:
        """Edge case: zero-message scenario shouldn't crash — we still probe."""
        binary = tmp_path / "ollama-bin"
        binary.write_text("# mock")
        captured: dict[str, list[str]] = {}

        def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
            captured["argv"] = argv
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout="",
                stderr='{"prompt_eval_count": 0}',
            )

        tokenizer = _FakeTokenizer(token_count=0)
        verify_token_count(
            ollama_name="demo:latest",
            hf_tokenizer=tokenizer,
            messages=[],
            runner=runner,
            binary=binary,
        )
        # Fallback probe prompt is the literal "hello".
        assert "hello" in captured["argv"]
