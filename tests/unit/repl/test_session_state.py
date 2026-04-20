"""ReplSession + GenerationParams state semantics."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from dlm.repl.session import GenerationParams, ReplSession


def _session(**overrides: object) -> ReplSession:
    defaults: dict[str, object] = {
        "backend": MagicMock(),
        "tokenizer": MagicMock(),
    }
    defaults.update(overrides)
    return ReplSession(**defaults)  # type: ignore[arg-type]


class TestGenerationParamsDefaults:
    def test_defaults_match_prompt_cli(self) -> None:
        p = GenerationParams()
        assert p.temperature == 0.7
        assert p.max_new_tokens == 256
        assert p.top_p is None
        assert p.top_k is None
        assert p.repetition_penalty is None

    def test_to_generate_kwargs_omits_none_knobs(self) -> None:
        p = GenerationParams()
        kwargs = p.to_generate_kwargs()
        assert "temperature" in kwargs
        assert "max_new_tokens" in kwargs
        assert "top_p" not in kwargs  # None values suppressed

    def test_to_generate_kwargs_includes_set_knobs(self) -> None:
        p = GenerationParams(top_p=0.9, top_k=40, repetition_penalty=1.1)
        kwargs = p.to_generate_kwargs()
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 40
        assert kwargs["repetition_penalty"] == 1.1


class TestReplSessionHistory:
    def test_append_user_adds_user_role(self) -> None:
        s = _session()
        s.append_user("hello")
        assert s.history == [{"role": "user", "content": "hello"}]

    def test_append_assistant_adds_assistant_role(self) -> None:
        s = _session()
        s.append_assistant("hi there")
        assert s.history == [{"role": "assistant", "content": "hi there"}]

    def test_append_assistant_cancelled_adds_marker(self) -> None:
        s = _session()
        s.append_assistant("partial ans", cancelled=True)
        assert s.history[-1]["content"].endswith("[cancelled]")

    def test_clear_history_wipes(self) -> None:
        s = _session()
        s.append_user("x")
        s.append_assistant("y")
        s.clear_history()
        assert s.history == []

    def test_save_history_writes_json(self, tmp_path: Path) -> None:
        s = _session()
        s.append_user("q")
        s.append_assistant("a")
        out = tmp_path / "chat.json"
        s.save_history(out)
        loaded = json.loads(out.read_text())
        assert loaded == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]

    def test_save_history_creates_parent_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "dir" / "chat.json"
        s = _session()
        s.append_user("q")
        s.save_history(out)
        assert out.exists()
