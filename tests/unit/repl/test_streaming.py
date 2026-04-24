"""Direct unit coverage for REPL streaming helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from dlm.repl.streaming import CaptureStreamer, build_streamer, concatenate_tokens, should_stream


def test_should_stream_tracks_stdout_tty_state() -> None:
    with patch("sys.stdout", new=SimpleNamespace(isatty=lambda: True)):
        assert should_stream() is True
    with patch("sys.stdout", new=SimpleNamespace(isatty=lambda: False)):
        assert should_stream() is False


def test_should_stream_handles_broken_stdout() -> None:
    class MissingIsAtty:
        pass

    class RaisesValueError:
        @staticmethod
        def isatty() -> bool:
            raise ValueError("closed")

    with patch("sys.stdout", new=MissingIsAtty()):
        assert should_stream() is False
    with patch("sys.stdout", new=RaisesValueError()):
        assert should_stream() is False


def test_capture_streamer_is_noop_and_keeps_text_buffer() -> None:
    streamer = CaptureStreamer()
    streamer.put(["ignored"])
    streamer.end()
    assert streamer.text == ""


def test_build_streamer_returns_capture_streamer_when_disabled() -> None:
    assert isinstance(build_streamer(object(), stream_to_stdout=False), CaptureStreamer)


def test_build_streamer_wraps_transformers_text_streamer() -> None:
    calls: list[tuple[object, bool, bool]] = []

    class FakeTextStreamer:
        def __init__(
            self, tokenizer: object, *, skip_prompt: bool, skip_special_tokens: bool
        ) -> None:
            calls.append((tokenizer, skip_prompt, skip_special_tokens))

    fake_transformers = SimpleNamespace(TextStreamer=FakeTextStreamer)
    tokenizer = object()

    with patch.dict("sys.modules", {"transformers": fake_transformers}):
        streamer = build_streamer(tokenizer, stream_to_stdout=True)

    assert isinstance(streamer, FakeTextStreamer)
    assert calls == [(tokenizer, True, True)]


def test_concatenate_tokens_joins_token_pieces() -> None:
    assert concatenate_tokens(["hello", " ", "world"]) == "hello world"
