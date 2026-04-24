"""Shared HTTP smoke helpers for OpenAI-compatible local runtimes."""

from __future__ import annotations

import json
import os
import socket
import subprocess  # nosec B404
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import TextIO

from dlm.export.errors import TargetSmokeError

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_STARTUP_TIMEOUT_SECONDS = 30.0
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 5.0
_DEFAULT_POLL_INTERVAL_SECONDS = 0.1
_DEFAULT_PROMPT = "Hello."
_DEFAULT_STARTUP_ATTEMPTS = 2


def reserve_local_port(host: str = _DEFAULT_HOST) -> int:
    """Ask the OS for a free loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def smoke_openai_compat_server(
    command: Sequence[str],
    *,
    host: str = _DEFAULT_HOST,
    port: int | None = None,
    env: Mapping[str, str] | None = None,
    startup_timeout: float = _DEFAULT_STARTUP_TIMEOUT_SECONDS,
    request_timeout: float = _DEFAULT_REQUEST_TIMEOUT_SECONDS,
    poll_interval: float = _DEFAULT_POLL_INTERVAL_SECONDS,
    prompt: str = _DEFAULT_PROMPT,
    startup_attempts: int = _DEFAULT_STARTUP_ATTEMPTS,
) -> str:
    """Start a local OpenAI-compatible server, wait for readiness, then chat."""
    if startup_attempts < 1:
        raise ValueError(f"startup_attempts must be >= 1, got {startup_attempts}")

    last_error: TargetSmokeError | None = None
    for _attempt in range(startup_attempts):
        real_port = port if port is not None else reserve_local_port(host)
        argv = _replace_or_append_flag(list(command), "--host", host)
        argv = _replace_or_append_flag(argv, "--port", str(real_port))

        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log:
            proc = subprocess.Popen(  # nosec B603
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=_merged_env(env),
            )
            try:
                model_id = _wait_for_models(
                    proc,
                    log,
                    host=host,
                    port=real_port,
                    startup_timeout=startup_timeout,
                    request_timeout=request_timeout,
                    poll_interval=poll_interval,
                )
                content = _chat_completion(
                    host=host,
                    port=real_port,
                    model_id=model_id,
                    prompt=prompt,
                    request_timeout=request_timeout,
                )
                first = _first_non_empty_line(content)
                if not first:
                    raise TargetSmokeError(
                        "openai-compatible smoke returned empty assistant content"
                    )
                return first
            except TargetSmokeError as exc:
                last_error = exc
                if port is not None:
                    raise
            finally:
                _stop_process(proc)

    assert last_error is not None
    raise last_error


def _wait_for_models(
    proc: subprocess.Popen[str],
    log: TextIO,
    *,
    host: str,
    port: int,
    startup_timeout: float,
    request_timeout: float,
    poll_interval: float,
) -> str | None:
    deadline = time.monotonic() + startup_timeout
    last_error: str | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise TargetSmokeError(
                f"server exited before readiness (exit {proc.returncode}){_log_tail(log)}"
            )
        try:
            return _fetch_model_id(host=host, port=port, request_timeout=request_timeout)
        except (
            OSError,
            TimeoutError,
            ValueError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(poll_interval)
    suffix = f" last error: {last_error}." if last_error else "."
    raise TargetSmokeError(
        f"server did not become ready on http://{host}:{port}/v1/models within "
        f"{startup_timeout:.1f}s.{suffix}{_log_tail(log)}"
    )


def _fetch_model_id(*, host: str, port: int, request_timeout: float) -> str | None:
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/models",
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=request_timeout) as resp:  # noqa: S310
        payload = json.loads(resp.read())
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    model_id = first.get("id")
    return model_id if isinstance(model_id, str) and model_id.strip() else None


def _chat_completion(
    *,
    host: str,
    port: int,
    model_id: str | None,
    prompt: str,
    request_timeout: float,
) -> str:
    payload = {
        "model": model_id or "dlm-smoke",
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=request_timeout) as resp:  # noqa: S310
        body = json.loads(resp.read())
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TargetSmokeError("chat completion response missing choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise TargetSmokeError("chat completion response has non-object choices[0]")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise TargetSmokeError("chat completion response missing choices[0].message")
    content = _normalize_message_content(message.get("content"))
    if content is None:
        raise TargetSmokeError(
            "chat completion response missing non-empty choices[0].message.content"
        )
    return content


def _normalize_message_content(content: object) -> str | None:
    if isinstance(content, str):
        stripped = content.strip()
        return stripped if stripped else None
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        merged = "\n".join(parts).strip()
        return merged if merged else None
    return None


def _replace_or_append_flag(argv: list[str], flag: str, value: str) -> list[str]:
    updated = list(argv)
    try:
        idx = updated.index(flag)
    except ValueError:
        updated.extend([flag, value])
        return updated
    if idx + 1 >= len(updated):
        updated.append(value)
        return updated
    updated[idx + 1] = value
    return updated


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5.0)


def _log_tail(log: TextIO, *, lines: int = 20) -> str:
    log.seek(0)
    text = log.read().strip()
    if not text:
        return ""
    tail = "\n".join(text.splitlines()[-lines:])
    return f"\n--- server log tail ---\n{tail}"


def _merged_env(env: Mapping[str, str] | None) -> dict[str, str]:
    if env is None:
        return dict(os.environ)
    merged = dict(os.environ)
    merged.update(env)
    return merged
