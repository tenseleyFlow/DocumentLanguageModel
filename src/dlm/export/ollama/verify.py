"""Go↔Jinja token-identity closed-loop verification (Sprint 12.6).

Sprint 12 shipped our Go template registry + a snapshot-diff against
hand-written goldens. That catches typos but doesn't prove Ollama's
rendering agrees with the base model's Jinja reference — only a
real round trip does. This module closes the loop:

    ollama run <name> --verbose <prompt>
        → parse `prompt_eval_count` from stderr telemetry
    HF apply_chat_template(messages, add_generation_prompt=True)
        → len(tokens)
    assert equal

Subprocess for `ollama run` is injectable; `parse_prompt_eval_count`
is a pure string parser unit-testable without Ollama installed. The
real closed-loop lives in the slow-marked integration test + the
weekly CI workflow (Sprint 14.5 lands the runner; this sprint lands
the harness).
"""

from __future__ import annotations

import json
import logging
import subprocess  # nosec B404
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final

from dlm.export.ollama.binary import locate_ollama
from dlm.export.ollama.errors import VerificationError

if TYPE_CHECKING:
    from pathlib import Path

_LOG = logging.getLogger(__name__)

_DEFAULT_RUN_TIMEOUT_SECONDS: Final[float] = 60.0

# Keys we look for in Ollama's `--verbose` telemetry JSON line.
_REQUIRED_TELEMETRY_KEYS: Final[tuple[str, ...]] = ("prompt_eval_count",)


def parse_prompt_eval_count(stderr: str) -> int:
    """Extract `prompt_eval_count` from `ollama run --verbose` stderr.

    Ollama emits a JSON-ish block at the end of `--verbose` runs that
    includes counters like `prompt_eval_count`, `eval_count`, durations.
    The exact shape has shifted across Ollama releases — some versions
    write a single compact JSON line, some pretty-print over multiple
    lines, some use a `"key":value` space-separated summary. We prefer
    the most specific parse and fall through to lighter-weight ones.

    Raises `VerificationError` when the counter can't be located.
    """
    # Shape 1: single-line JSON. `stderr.splitlines()` in reverse so
    # we prefer the last (summary) line over any earlier debug output.
    for line in reversed(stderr.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        if '"prompt_eval_count"' not in stripped:
            continue
        try:
            blob = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        val = blob.get("prompt_eval_count")
        if isinstance(val, int):
            return val

    # Shape 2: Ollama's space-separated summary lines —
    #   `prompt eval count:  42 token(s)`
    # The whole summary block is tail-of-stderr; scan back-to-front.
    for line in reversed(stderr.splitlines()):
        stripped = line.strip().lower()
        if "prompt eval count" not in stripped:
            continue
        # Extract the first integer after the label.
        after_colon = stripped.split(":", 1)[-1]
        for token in after_colon.split():
            if token.isdigit():
                return int(token)

    raise VerificationError(
        ollama_name="<unknown>",
        hf_count=-1,
        go_count=-1,
        scenario=(
            "telemetry parse: no `prompt_eval_count` found in ollama "
            f"stderr (head: {stderr[:200]!r})"
        ),
    )


def verify_token_count(
    *,
    ollama_name: str,
    hf_tokenizer: Any,
    messages: list[dict[str, str]],
    scenario: str | None = None,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    binary: Path | None = None,
    timeout: float = _DEFAULT_RUN_TIMEOUT_SECONDS,
) -> None:
    """Run the closed-loop check for one message set.

    Renders the prompt through HF's `apply_chat_template` (Jinja side)
    AND through Ollama via `ollama run --verbose` (Go side), then
    compares `len(hf_tokens)` to Ollama's `prompt_eval_count`. Raises
    `VerificationError` on mismatch; returns cleanly on parity.

    `runner` is a test seam — production omits it and the function
    calls `subprocess.run` with the real Ollama binary.

    Note: Ollama's `prompt_eval_count` includes the assistant-turn
    prefix the template emits before generation starts; HF's
    `apply_chat_template(add_generation_prompt=True)` does the same.
    An identical number = identical rendered prompt tokens.
    """
    hf_tokens = hf_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    hf_count = len(hf_tokens)

    # The prompt we send is immaterial — we only care about
    # `prompt_eval_count`. Use the last user turn's content so Ollama
    # has something meaningful to log.
    probe_prompt = messages[-1].get("content", "hello") if messages else "hello"
    telemetry = run_with_telemetry(
        ollama_name=ollama_name,
        prompt=probe_prompt,
        runner=runner,
        binary=binary,
        timeout=timeout,
    )
    go_count = telemetry

    if go_count != hf_count:
        raise VerificationError(
            ollama_name=ollama_name,
            hf_count=hf_count,
            go_count=go_count,
            scenario=scenario,
        )


def run_with_telemetry(
    *,
    ollama_name: str,
    prompt: str,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    binary: Path | None = None,
    timeout: float = _DEFAULT_RUN_TIMEOUT_SECONDS,
) -> int:
    """Run `ollama run <name> --verbose <prompt>`, return `prompt_eval_count`.

    Pipes stderr (where `--verbose` writes its telemetry) into
    `parse_prompt_eval_count`. The subprocess is discarded on success
    — we only care about the counter.
    """
    exe = binary or locate_ollama()
    argv = [str(exe), "run", ollama_name, "--verbose", prompt]

    def _default_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # nosec B603 — caller-controlled argv
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    run = runner or _default_runner
    proc = run(argv)

    if proc.returncode != 0:
        raise VerificationError(
            ollama_name=ollama_name,
            hf_count=-1,
            go_count=-1,
            scenario=(
                f"ollama run exited {proc.returncode}: "
                f"{(proc.stderr or proc.stdout or '').strip()[:200]}"
            ),
        )
    return parse_prompt_eval_count(proc.stderr or "")
