"""Slash command parser + handler matrix."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dlm.repl.commands import Action, is_command, parse_and_dispatch
from dlm.repl.errors import BadCommandArgumentError, UnknownCommandError
from dlm.repl.session import ReplSession


def _session(**overrides: object) -> ReplSession:
    defaults: dict[str, object] = {
        "backend": MagicMock(name="backend"),
        "tokenizer": MagicMock(name="tokenizer"),
    }
    defaults.update(overrides)
    return ReplSession(**defaults)  # type: ignore[arg-type]


class TestIsCommand:
    def test_slash_leading(self) -> None:
        assert is_command("/exit") is True
        assert is_command("  /exit") is True

    def test_plain_text_not_command(self) -> None:
        assert is_command("hello") is False
        assert is_command("") is False


class TestExitQuit:
    def test_exit_returns_exit_action(self) -> None:
        result = parse_and_dispatch("/exit", _session())
        assert result.action is Action.EXIT

    def test_quit_is_alias_for_exit(self) -> None:
        result = parse_and_dispatch("/quit", _session())
        assert result.action is Action.EXIT


class TestClear:
    def test_clear_wipes_history(self) -> None:
        s = _session()
        s.append_user("x")
        parse_and_dispatch("/clear", s)
        assert s.history == []


class TestSave:
    def test_save_writes_and_returns_message(self, tmp_path: Path) -> None:
        s = _session()
        s.append_user("hi")
        s.append_assistant("hello")
        out = tmp_path / "saved.json"
        result = parse_and_dispatch(f"/save {out}", s)
        assert out.exists()
        assert result.message is not None
        assert "2 messages" in result.message

    def test_save_without_path_raises(self) -> None:
        with pytest.raises(BadCommandArgumentError, match="requires a path"):
            parse_and_dispatch("/save", _session())


class TestAdapter:
    def test_single_adapter_doc_refuses(self) -> None:
        s = _session(declared_adapters=())
        with pytest.raises(BadCommandArgumentError, match="multi-adapter"):
            parse_and_dispatch("/adapter knowledge", s)

    def test_unknown_adapter_refused(self) -> None:
        s = _session(declared_adapters=("knowledge", "tone"))
        with pytest.raises(BadCommandArgumentError, match="not declared"):
            parse_and_dispatch("/adapter ghost", s)

    def test_happy_path_updates_active(self) -> None:
        s = _session(declared_adapters=("knowledge", "tone"))
        parse_and_dispatch("/adapter tone", s)
        assert s.active_adapter == "tone"

    def test_empty_args_raises(self) -> None:
        with pytest.raises(BadCommandArgumentError, match="requires"):
            parse_and_dispatch("/adapter", _session(declared_adapters=("x",)))


class TestParams:
    def test_empty_args_prints_current(self) -> None:
        s = _session()
        result = parse_and_dispatch("/params", s)
        assert result.message is not None
        assert "temperature=0.7" in result.message

    def test_updates_float(self) -> None:
        s = _session()
        parse_and_dispatch("/params temperature=0.5", s)
        assert s.gen_params.temperature == 0.5

    def test_updates_int(self) -> None:
        s = _session()
        parse_and_dispatch("/params max_new_tokens=128", s)
        assert s.gen_params.max_new_tokens == 128

    def test_multiple_assignments(self) -> None:
        s = _session()
        parse_and_dispatch("/params temperature=0.3 top_p=0.9", s)
        assert s.gen_params.temperature == 0.3
        assert s.gen_params.top_p == 0.9

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(BadCommandArgumentError, match="unknown key"):
            parse_and_dispatch("/params flavor=salty", _session())

    def test_bad_float_raises(self) -> None:
        with pytest.raises(BadCommandArgumentError, match="not a number"):
            parse_and_dispatch("/params temperature=hot", _session())

    def test_bad_int_raises(self) -> None:
        with pytest.raises(BadCommandArgumentError, match="not an integer"):
            parse_and_dispatch("/params max_new_tokens=lots", _session())

    def test_missing_equals_raises(self) -> None:
        with pytest.raises(BadCommandArgumentError, match="key=value"):
            parse_and_dispatch("/params temperature", _session())

    def test_partial_failure_leaves_prior_intact(self) -> None:
        """Bad second entry mustn't leave first half applied."""
        s = _session()
        initial_temp = s.gen_params.temperature
        with pytest.raises(BadCommandArgumentError):
            parse_and_dispatch("/params temperature=0.5 top_p=oops", s)
        assert s.gen_params.temperature == initial_temp


class TestModelAndHistory:
    def test_model_mentions_backend_name(self) -> None:
        backend = MagicMock()
        backend.name = "pytorch"
        s = _session(backend=backend)
        result = parse_and_dispatch("/model", s)
        assert result.message is not None
        assert "pytorch" in result.message

    def test_history_empty_message(self) -> None:
        result = parse_and_dispatch("/history", _session())
        assert result.message is not None
        assert "empty" in result.message.lower()

    def test_history_lists_messages(self) -> None:
        s = _session()
        s.append_user("q")
        s.append_assistant("a")
        result = parse_and_dispatch("/history", s)
        assert result.message is not None
        assert "user" in result.message
        assert "assistant" in result.message


class TestHelp:
    def test_help_includes_every_command(self) -> None:
        result = parse_and_dispatch("/help", _session())
        assert result.message is not None
        for cmd in ("/exit", "/clear", "/save", "/adapter", "/params", "/model", "/history"):
            assert cmd in result.message


class TestUnknownCommand:
    def test_unknown_slash_raises(self) -> None:
        with pytest.raises(UnknownCommandError, match="/bogus"):
            parse_and_dispatch("/bogus", _session())

    def test_bare_slash_raises(self) -> None:
        with pytest.raises(UnknownCommandError, match="empty"):
            parse_and_dispatch("/", _session())

    def test_non_slash_line_raises(self) -> None:
        with pytest.raises(UnknownCommandError, match="does not start"):
            parse_and_dispatch("hello", _session())
