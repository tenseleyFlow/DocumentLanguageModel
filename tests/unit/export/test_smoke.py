"""Deterministic unit coverage for the shared OpenAI-compatible smoke helper."""

from __future__ import annotations

import io
import json
import subprocess
import urllib.error
from collections.abc import Callable, Iterator

import pytest

from dlm.export import smoke as smoke_mod
from dlm.export.errors import TargetSmokeError


class _FakeProc:
    def __init__(self, *, returncode: int | None = None, kill_times_out: bool = False) -> None:
        self.returncode = returncode
        self.kill_times_out = kill_times_out
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float) -> None:
        self.wait_calls += 1
        if self.kill_times_out and self.wait_calls == 1:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)

    def kill(self) -> None:
        self.killed = True


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _urlopen_with(payload: object) -> Callable[..., _FakeResponse]:
    def _fake_urlopen(*_args: object, **_kwargs: object) -> _FakeResponse:
        return _FakeResponse(payload)

    return _fake_urlopen


class TestSmokeOpenAiCompatServer:
    def test_returns_first_response_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        popen_argv: list[list[str]] = []
        popen_env: list[dict[str, str] | None] = []
        stopped: list[_FakeProc] = []

        def _fake_popen(argv: list[str], **kwargs: object) -> _FakeProc:
            popen_argv.append(list(argv))
            env = kwargs.get("env")
            popen_env.append(env if isinstance(env, dict) else None)
            return _FakeProc()

        monkeypatch.setattr(smoke_mod, "reserve_local_port", lambda host: 43123)
        monkeypatch.setattr(smoke_mod.subprocess, "Popen", _fake_popen)
        monkeypatch.setattr(smoke_mod, "_wait_for_models", lambda *args, **kwargs: "fake-model")
        monkeypatch.setattr(
            smoke_mod,
            "_chat_completion",
            lambda *args, **kwargs: "\n hello from fake server \nsecond line",
        )
        monkeypatch.setattr(smoke_mod, "_stop_process", lambda proc: stopped.append(proc))

        first_line = smoke_mod.smoke_openai_compat_server(
            ["fake-server", "--mode", "ok", "--host", "0.0.0.0", "--port", "8000"],
            env={"FAKE_SMOKE_TOKEN": "ready"},
        )

        assert first_line == "hello from fake server"
        assert popen_argv == [
            ["fake-server", "--mode", "ok", "--host", "127.0.0.1", "--port", "43123"]
        ]
        assert popen_env[0] is not None
        assert popen_env[0]["FAKE_SMOKE_TOKEN"] == "ready"
        assert len(stopped) == 1

    def test_empty_content_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(smoke_mod, "reserve_local_port", lambda host: 42000)
        monkeypatch.setattr(
            smoke_mod.subprocess,
            "Popen",
            lambda argv, **kwargs: _FakeProc(),
        )
        monkeypatch.setattr(smoke_mod, "_wait_for_models", lambda *args, **kwargs: "fake-model")
        monkeypatch.setattr(smoke_mod, "_chat_completion", lambda *args, **kwargs: "  \n  ")
        monkeypatch.setattr(smoke_mod, "_stop_process", lambda proc: None)

        with pytest.raises(TargetSmokeError, match="empty assistant content"):
            smoke_mod.smoke_openai_compat_server(["fake-server"])

    def test_retries_dynamic_port_after_target_smoke_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ports = iter((41001, 41002))
        popen_argv: list[list[str]] = []
        wait_calls = 0

        def _fake_popen(argv: list[str], **kwargs: object) -> _FakeProc:
            popen_argv.append(list(argv))
            return _FakeProc()

        def _fake_wait(*args: object, **kwargs: object) -> str | None:
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls == 1:
                raise TargetSmokeError("port raced")
            return None

        monkeypatch.setattr(smoke_mod, "reserve_local_port", lambda host: next(ports))
        monkeypatch.setattr(smoke_mod.subprocess, "Popen", _fake_popen)
        monkeypatch.setattr(smoke_mod, "_wait_for_models", _fake_wait)
        monkeypatch.setattr(smoke_mod, "_chat_completion", lambda *args, **kwargs: "hello")
        monkeypatch.setattr(smoke_mod, "_stop_process", lambda proc: None)

        first_line = smoke_mod.smoke_openai_compat_server(["fake-server"], startup_attempts=2)

        assert first_line == "hello"
        assert popen_argv == [
            ["fake-server", "--host", "127.0.0.1", "--port", "41001"],
            ["fake-server", "--host", "127.0.0.1", "--port", "41002"],
        ]

    def test_fixed_port_does_not_retry_after_target_smoke_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        popen_argv: list[list[str]] = []

        def _fake_popen(argv: list[str], **kwargs: object) -> _FakeProc:
            popen_argv.append(list(argv))
            return _FakeProc()

        monkeypatch.setattr(smoke_mod.subprocess, "Popen", _fake_popen)
        monkeypatch.setattr(
            smoke_mod,
            "_wait_for_models",
            lambda *args, **kwargs: (_ for _ in ()).throw(TargetSmokeError("boom")),
        )
        monkeypatch.setattr(smoke_mod, "_stop_process", lambda proc: None)

        with pytest.raises(TargetSmokeError, match="boom"):
            smoke_mod.smoke_openai_compat_server(["fake-server"], port=49999, startup_attempts=3)

        assert popen_argv == [["fake-server", "--host", "127.0.0.1", "--port", "49999"]]

    def test_invalid_startup_attempts_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="startup_attempts"):
            smoke_mod.smoke_openai_compat_server(["fake-server"], startup_attempts=0)


class TestWaitForModels:
    def test_returns_model_id_after_retryable_fetch_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proc = _FakeProc()
        seen_sleeps: list[float] = []
        responses: Iterator[object] = iter(
            [
                urllib.error.URLError("warming up"),
                "fake-model",
            ]
        )

        def _fake_fetch(**kwargs: object) -> str | None:
            outcome = next(responses)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(smoke_mod, "_fetch_model_id", _fake_fetch)
        monkeypatch.setattr(smoke_mod.time, "sleep", lambda seconds: seen_sleeps.append(seconds))

        model_id = smoke_mod._wait_for_models(
            proc,
            io.StringIO(""),
            host="127.0.0.1",
            port=41000,
            startup_timeout=1.0,
            request_timeout=0.1,
            poll_interval=0.25,
        )

        assert model_id == "fake-model"
        assert seen_sleeps == [0.25]

    def test_raises_when_process_exits_before_readiness(self) -> None:
        proc = _FakeProc(returncode=3)

        with pytest.raises(TargetSmokeError, match="exited before readiness"):
            smoke_mod._wait_for_models(
                proc,
                io.StringIO("first\nsecond"),
                host="127.0.0.1",
                port=41000,
                startup_timeout=1.0,
                request_timeout=0.1,
                poll_interval=0.1,
            )

    def test_raises_timeout_with_last_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proc = _FakeProc()
        monotonic_values = iter((0.0, 0.05, 0.11))

        monkeypatch.setattr(smoke_mod.time, "monotonic", lambda: next(monotonic_values))
        monkeypatch.setattr(
            smoke_mod,
            "_fetch_model_id",
            lambda **kwargs: (_ for _ in ()).throw(TimeoutError("late reply")),
        )
        monkeypatch.setattr(smoke_mod.time, "sleep", lambda seconds: None)

        with pytest.raises(TargetSmokeError, match="late reply"):
            smoke_mod._wait_for_models(
                proc,
                io.StringIO(""),
                host="127.0.0.1",
                port=41000,
                startup_timeout=0.1,
                request_timeout=0.1,
                poll_interval=0.05,
            )


class TestFetchModelId:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ({"data": [{"id": "model-1"}]}, "model-1"),
            ({"data": []}, None),
            ({"data": ["not-a-dict"]}, None),
            ({"data": [{"id": "   "}]}, None),
        ],
    )
    def test_fetch_model_id_parses_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
        payload: object,
        expected: str | None,
    ) -> None:
        monkeypatch.setattr(
            smoke_mod.urllib.request,
            "urlopen",
            _urlopen_with(payload),
        )

        assert (
            smoke_mod._fetch_model_id(host="127.0.0.1", port=41000, request_timeout=0.1) == expected
        )


class TestChatCompletion:
    def test_returns_string_or_list_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            smoke_mod.urllib.request,
            "urlopen",
            _urlopen_with(
                {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"text": "  first  "},
                                    {"not_text": "ignored"},
                                    {"text": "second"},
                                ]
                            }
                        }
                    ]
                }
            ),
        )

        assert (
            smoke_mod._chat_completion(
                host="127.0.0.1",
                port=41000,
                model_id=None,
                prompt="Hello",
                request_timeout=0.1,
            )
            == "first\nsecond"
        )

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ({}, "missing choices"),
            ({"choices": ["bad"]}, "non-object"),
            ({"choices": [{}]}, "missing choices\\[0\\]\\.message"),
            (
                {"choices": [{"message": {"content": ""}}]},
                "missing non-empty choices\\[0\\]\\.message\\.content",
            ),
        ],
    )
    def test_raises_for_invalid_response_shapes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        payload: object,
        match: str,
    ) -> None:
        monkeypatch.setattr(smoke_mod.urllib.request, "urlopen", _urlopen_with(payload))

        with pytest.raises(TargetSmokeError, match=match):
            smoke_mod._chat_completion(
                host="127.0.0.1",
                port=41000,
                model_id="model-1",
                prompt="Hello",
                request_timeout=0.1,
            )


class TestSmokeHelpers:
    def test_normalize_message_content(self) -> None:
        assert smoke_mod._normalize_message_content("  hello  ") == "hello"
        assert (
            smoke_mod._normalize_message_content(
                [{"text": " first "}, {"skip": True}, {"text": "second"}]
            )
            == "first\nsecond"
        )
        assert smoke_mod._normalize_message_content([{"text": "   "}]) is None
        assert smoke_mod._normalize_message_content(3) is None

    def test_replace_or_append_flag_and_first_non_empty_line(self) -> None:
        assert smoke_mod._replace_or_append_flag(["cmd"], "--host", "127.0.0.1") == [
            "cmd",
            "--host",
            "127.0.0.1",
        ]
        assert smoke_mod._replace_or_append_flag(["cmd", "--port"], "--port", "8000") == [
            "cmd",
            "--port",
            "8000",
        ]
        assert smoke_mod._first_non_empty_line("\n \nhello\nworld\n") == "hello"
        assert smoke_mod._first_non_empty_line(" \n\t") == ""

    def test_stop_process_kills_after_timeout(self) -> None:
        proc = _FakeProc(kill_times_out=True)

        smoke_mod._stop_process(proc)

        assert proc.terminated is True
        assert proc.killed is True

    def test_stop_process_is_noop_when_already_exited(self) -> None:
        proc = _FakeProc(returncode=0)

        smoke_mod._stop_process(proc)

        assert proc.terminated is False
        assert proc.killed is False

    def test_log_tail_and_merged_env(self) -> None:
        log = io.StringIO("line1\nline2\nline3")

        assert "--- server log tail ---" in smoke_mod._log_tail(log, lines=2)
        assert smoke_mod._merged_env({"FAKE_SMOKE_TOKEN": "ready"})["FAKE_SMOKE_TOKEN"] == "ready"
