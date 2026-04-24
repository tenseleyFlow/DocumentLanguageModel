"""Teacher selector parsing and runtime wrappers for Sprint 43."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import dlm.synth.teachers as teachers_mod
from dlm.synth import (
    AnthropicTeacher,
    HfTeacher,
    InvalidTeacherSpecError,
    OpenAiTeacher,
    SelfTeacher,
    TeacherUnavailableError,
    VllmServerTeacher,
    build_teacher,
    parse_teacher_ref,
)


class TestTeacherSelectorParsing:
    @pytest.mark.parametrize(
        ("raw", "kind", "target"),
        [
            ("self", "self", None),
            ("hf:Qwen/Qwen2.5-1.5B-Instruct", "hf", "Qwen/Qwen2.5-1.5B-Instruct"),
            ("openai:gpt-4o-mini", "openai", "gpt-4o-mini"),
            ("anthropic:claude-3-5-haiku-latest", "anthropic", "claude-3-5-haiku-latest"),
            ("vllm-server:http://127.0.0.1:8000", "vllm-server", "http://127.0.0.1:8000"),
        ],
    )
    def test_parse_teacher_ref(self, raw: str, kind: str, target: str | None) -> None:
        ref = parse_teacher_ref(raw)
        assert ref.kind == kind
        assert ref.target == target

    def test_empty_selector_refused(self) -> None:
        with pytest.raises(InvalidTeacherSpecError, match="must not be empty"):
            parse_teacher_ref("   ")

    def test_unknown_selector_refused(self) -> None:
        with pytest.raises(InvalidTeacherSpecError, match="unknown teacher selector"):
            parse_teacher_ref("mystery:thing")


class TestBuildTeacher:
    def test_self_requires_dlm_path(self) -> None:
        with pytest.raises(TeacherUnavailableError, match="requires the .dlm path context"):
            build_teacher("self")

    def test_build_teacher_dispatches(self, tmp_path: Path) -> None:
        self_teacher = build_teacher("self", dlm_path=tmp_path / "doc.dlm")
        assert isinstance(self_teacher, SelfTeacher)
        assert self_teacher.backend == "pytorch"
        assert isinstance(build_teacher("hf:foo/bar"), HfTeacher)
        assert isinstance(build_teacher("openai:gpt-4o-mini"), OpenAiTeacher)
        assert isinstance(build_teacher("anthropic:claude"), AnthropicTeacher)
        assert isinstance(
            build_teacher("vllm-server:http://127.0.0.1:8000"),
            VllmServerTeacher,
        )


class TestSelfTeacher:
    def test_self_teacher_uses_loader_once_and_forwards_kwargs(self, tmp_path: Path) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []
        loaded_paths: list[tuple[Path, str]] = []

        class _Backend:
            def generate(self, prompt: str, **gen_kwargs: Any) -> str:
                calls.append((prompt, gen_kwargs))
                return "  synthesized answer  "

        def _loader(path: Path, backend: str) -> _Backend:
            loaded_paths.append((path, backend))
            return _Backend()

        teacher = SelfTeacher(tmp_path / "doc.dlm", loader=_loader)
        out1 = teacher.generate(
            "system text",
            "user text",
            max_new_tokens=33,
            temperature=0.7,
            top_p=0.9,
            seed=7,
        )
        out2 = teacher.generate("system text", "user text")

        assert out1 == "synthesized answer"
        assert out2 == "synthesized answer"
        assert loaded_paths == [(tmp_path / "doc.dlm", "auto")]
        assert "system text" in calls[0][0]
        assert "user text" in calls[0][0]
        assert calls[0][1] == {
            "max_new_tokens": 33,
            "temperature": 0.7,
            "top_p": 0.9,
        }


class TestHfTeacher:
    def test_hf_teacher_uses_loader_and_runner(self) -> None:
        seen: dict[str, Any] = {}

        def _loader(hf_id: str, device: str) -> teachers_mod._LoadedHfTeacher:
            seen["loader"] = (hf_id, device)
            return teachers_mod._LoadedHfTeacher(model="model", tokenizer="tok", device=device)

        def _runner(
            model: Any,
            tokenizer: Any,
            prompt: str,
            *,
            max_new_tokens: int,
            temperature: float,
            top_p: float | None,
            seed: int | None,
        ) -> str:
            seen["runner"] = (model, tokenizer, prompt, max_new_tokens, temperature, top_p, seed)
            return " hf output "

        teacher = HfTeacher("Qwen/Qwen2.5-1.5B-Instruct", loader=_loader, runner=_runner)
        out = teacher.generate(
            "system", "user", max_new_tokens=21, temperature=0.5, top_p=0.8, seed=11
        )
        assert out == "hf output"
        assert seen["loader"] == ("Qwen/Qwen2.5-1.5B-Instruct", "cpu")
        assert seen["runner"][3:] == (21, 0.5, 0.8, 11)


class TestOpenAiTeacher:
    def test_missing_api_key_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        teacher = OpenAiTeacher("gpt-4o-mini")
        with pytest.raises(TeacherUnavailableError, match="OPENAI_API_KEY"):
            teacher.generate("system", "user")

    def test_openai_teacher_extracts_message_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "secret")

        captured: dict[str, Any] = {}

        def _create(**kwargs: Any) -> Any:
            captured["payload"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=" generated "))]
            )

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=_create),
            )
        )

        teacher = OpenAiTeacher("gpt-4o-mini", client_factory=lambda api_key: client)
        out = teacher.generate("sys", "usr", max_new_tokens=17, temperature=0.3, top_p=0.7, seed=5)
        assert out == "generated"
        assert captured["payload"]["model"] == "gpt-4o-mini"
        assert captured["payload"]["seed"] == 5


class TestAnthropicTeacher:
    def test_missing_api_key_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        teacher = AnthropicTeacher("claude-3-5-haiku-latest")
        with pytest.raises(TeacherUnavailableError, match="ANTHROPIC_API_KEY"):
            teacher.generate("system", "user")

    def test_anthropic_teacher_extracts_text_blocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
        captured: dict[str, Any] = {}

        class _Messages:
            @staticmethod
            def create(**kwargs: Any) -> Any:
                captured["payload"] = kwargs
                return SimpleNamespace(
                    content=[
                        SimpleNamespace(type="text", text=" first "),
                        SimpleNamespace(type="text", text=" second "),
                    ]
                )

        class _Client:
            messages = _Messages()

        teacher = AnthropicTeacher(
            "claude-3-5-haiku-latest",
            client_factory=lambda api_key: _Client(),
        )
        out = teacher.generate("sys", "usr", max_new_tokens=19, temperature=0.2, top_p=0.6)
        assert out == "first\nsecond"
        assert captured["payload"]["model"] == "claude-3-5-haiku-latest"


class TestVllmServerTeacher:
    def test_invalid_url_refused(self) -> None:
        with pytest.raises(InvalidTeacherSpecError, match="http\\(s\\)"):
            VllmServerTeacher("localhost:8000")

    def test_vllm_teacher_queries_model_and_completion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, Any] = {}

        def _fake_models(base_url: str, *, request_timeout: float) -> str | None:
            calls["models"] = (base_url, request_timeout)
            return "demo-model"

        def _fake_completion(
            base_url: str,
            *,
            model_id: str | None,
            messages: list[dict[str, str]],
            max_new_tokens: int,
            temperature: float,
            top_p: float | None,
            seed: int | None,
            request_timeout: float,
        ) -> str:
            calls["completion"] = (
                base_url,
                model_id,
                messages,
                max_new_tokens,
                temperature,
                top_p,
                seed,
                request_timeout,
            )
            return " served "

        monkeypatch.setattr(teachers_mod, "_fetch_openai_compat_model_id", _fake_models)
        monkeypatch.setattr(teachers_mod, "_request_openai_compat_completion", _fake_completion)

        teacher = VllmServerTeacher("http://127.0.0.1:8000")
        out = teacher.generate("sys", "usr", max_new_tokens=29, temperature=0.4, top_p=0.75, seed=9)

        assert out == "served"
        assert calls["models"] == ("http://127.0.0.1:8000", 30.0)
        assert calls["completion"][1] == "demo-model"
        assert calls["completion"][3:] == (29, 0.4, 0.75, 9, 30.0)
