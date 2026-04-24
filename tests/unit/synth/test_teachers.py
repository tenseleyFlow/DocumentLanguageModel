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
    TeacherInvocationError,
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

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            ("hf:   ", "hf teacher selector must include a model id"),
            ("openai:   ", "openai teacher selector must include a model id"),
            ("anthropic:   ", "anthropic teacher selector must include a model id"),
            ("vllm-server:   ", "vllm-server teacher selector must include a URL"),
        ],
    )
    def test_missing_selector_targets_are_refused(self, raw: str, message: str) -> None:
        with pytest.raises(InvalidTeacherSpecError, match=message):
            parse_teacher_ref(raw)


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
    def test_blank_hf_id_refused(self) -> None:
        with pytest.raises(InvalidTeacherSpecError, match="must include a model id"):
            HfTeacher("   ")

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
        assert seen["loader"] == (
            "Qwen/Qwen2.5-1.5B-Instruct",
            teachers_mod._resolve_generation_device("auto"),
        )
        assert seen["runner"][3:] == (21, 0.5, 0.8, 11)

    def test_hf_teacher_reuses_loaded_bundle(self) -> None:
        loads: list[tuple[str, str]] = []

        def _loader(hf_id: str, device: str) -> teachers_mod._LoadedHfTeacher:
            loads.append((hf_id, device))
            return teachers_mod._LoadedHfTeacher(model="model", tokenizer="tok", device=device)

        teacher = HfTeacher(
            "Qwen/Qwen2.5-1.5B-Instruct",
            loader=_loader,
            runner=lambda *_args, **_kwargs: "ok",
        )

        assert teacher.generate("system", "user") == "ok"
        assert teacher.generate("system", "user") == "ok"
        assert loads == [
            ("Qwen/Qwen2.5-1.5B-Instruct", teachers_mod._resolve_generation_device("auto"))
        ]


class TestOpenAiTeacher:
    def test_blank_model_refused(self) -> None:
        with pytest.raises(InvalidTeacherSpecError, match="must include a model id"):
            OpenAiTeacher("   ")

    def test_missing_api_key_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        teacher = OpenAiTeacher("gpt-4o-mini")
        with pytest.raises(TeacherUnavailableError, match="OPENAI_API_KEY"):
            teacher.generate("system", "user")

    def test_openai_teacher_extracts_message_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "secret")

        payloads: list[dict[str, Any]] = []
        factories: list[str] = []

        def _create(**kwargs: Any) -> Any:
            payloads.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=" generated "))]
            )

        def _factory(api_key: str) -> Any:
            factories.append(api_key)
            return client

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=_create),
            )
        )

        teacher = OpenAiTeacher(
            "gpt-4o-mini",
            client_factory=_factory,
        )
        out = teacher.generate("sys", "usr", max_new_tokens=17, temperature=0.3, top_p=0.7, seed=5)
        second = teacher.generate("sys", "usr")
        assert out == "generated"
        assert second == "generated"
        assert payloads[0]["model"] == "gpt-4o-mini"
        assert payloads[0]["seed"] == 5
        assert factories == ["secret"]

    def test_openai_teacher_wraps_request_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "secret")

        def _create(**_kwargs: Any) -> Any:
            raise RuntimeError("boom")

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=_create),
            )
        )
        teacher = OpenAiTeacher("gpt-4o-mini", client_factory=lambda _api_key: client)

        with pytest.raises(TeacherInvocationError, match="openai:gpt-4o-mini request failed: boom"):
            teacher.generate("sys", "usr")


class TestAnthropicTeacher:
    def test_blank_model_refused(self) -> None:
        with pytest.raises(InvalidTeacherSpecError, match="must include a model id"):
            AnthropicTeacher("   ")

    def test_missing_api_key_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        teacher = AnthropicTeacher("claude-3-5-haiku-latest")
        with pytest.raises(TeacherUnavailableError, match="ANTHROPIC_API_KEY"):
            teacher.generate("system", "user")

    def test_anthropic_teacher_extracts_text_blocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
        captured: dict[str, Any] = {}
        factories: list[str] = []

        class _Messages:
            @staticmethod
            def create(**kwargs: Any) -> Any:
                captured["payload"] = kwargs
                return SimpleNamespace(
                    content=[
                        SimpleNamespace(type="image", text="ignored"),
                        SimpleNamespace(type="text", text=" first "),
                        SimpleNamespace(type="text", text=" second "),
                    ]
                )

        class _Client:
            messages = _Messages()

        def _factory(api_key: str) -> _Client:
            factories.append(api_key)
            return _Client()

        teacher = AnthropicTeacher(
            "claude-3-5-haiku-latest",
            client_factory=_factory,
        )
        out = teacher.generate("sys", "usr", max_new_tokens=19, temperature=0.2, top_p=0.6)
        second = teacher.generate("sys", "usr")
        assert out == "first\nsecond"
        assert second == "first\nsecond"
        assert captured["payload"]["model"] == "claude-3-5-haiku-latest"
        assert factories == ["secret"]

    def test_anthropic_teacher_wraps_request_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")

        class _Messages:
            @staticmethod
            def create(**_kwargs: Any) -> Any:
                raise RuntimeError("boom")

        class _Client:
            messages = _Messages()

        teacher = AnthropicTeacher(
            "claude-3-5-haiku-latest",
            client_factory=lambda _api_key: _Client(),
        )

        with pytest.raises(
            TeacherInvocationError,
            match="anthropic:claude-3-5-haiku-latest request failed: boom",
        ):
            teacher.generate("sys", "usr")


class TestVllmServerTeacher:
    def test_blank_url_refused(self) -> None:
        with pytest.raises(InvalidTeacherSpecError, match="must include a URL"):
            VllmServerTeacher("   ")

    def test_invalid_url_refused(self) -> None:
        with pytest.raises(InvalidTeacherSpecError, match="http\\(s\\)"):
            VllmServerTeacher("localhost:8000")

    def test_vllm_teacher_queries_model_and_completion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model_calls: list[tuple[str, float]] = []
        completion_calls: list[tuple[Any, ...]] = []

        def _fake_models(base_url: str, *, request_timeout: float) -> str | None:
            model_calls.append((base_url, request_timeout))
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
            completion_calls.append(
                (
                    base_url,
                    model_id,
                    messages,
                    max_new_tokens,
                    temperature,
                    top_p,
                    seed,
                    request_timeout,
                )
            )
            return " served "

        monkeypatch.setattr(teachers_mod, "_fetch_openai_compat_model_id", _fake_models)
        monkeypatch.setattr(teachers_mod, "_request_openai_compat_completion", _fake_completion)

        teacher = VllmServerTeacher("http://127.0.0.1:8000")
        out = teacher.generate("sys", "usr", max_new_tokens=29, temperature=0.4, top_p=0.75, seed=9)
        second = teacher.generate("sys", "usr")

        assert out == "served"
        assert second == "served"
        assert model_calls == [("http://127.0.0.1:8000", 30.0)]
        assert completion_calls[0][1] == "demo-model"
        assert completion_calls[0][3:] == (29, 0.4, 0.75, 9, 30.0)


class TestTeacherHelpers:
    def test_flatten_teacher_prompt_handles_partial_inputs(self) -> None:
        assert teachers_mod._flatten_teacher_prompt("system", "user").startswith("System:\n")
        assert teachers_mod._flatten_teacher_prompt("", "user") == "user"
        assert teachers_mod._flatten_teacher_prompt("system", "") == "system"

    def test_require_non_empty_teacher_output_refuses_blank_text(self) -> None:
        with pytest.raises(TeacherInvocationError, match="self returned empty output"):
            teachers_mod._require_non_empty_teacher_output("   ", teacher="self")

    def test_extract_openai_message_text_handles_list_content_and_errors(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"text": " first "},
                            {"text": " second "},
                        ]
                    }
                }
            ]
        }
        assert teachers_mod._extract_openai_message_text(response) == "first\nsecond"

        with pytest.raises(TeacherInvocationError, match="missing choices"):
            teachers_mod._extract_openai_message_text({})

        with pytest.raises(TeacherInvocationError, match="missing choices\\[0\\]\\.message"):
            teachers_mod._extract_openai_message_text({"choices": [{}]})

        with pytest.raises(TeacherInvocationError, match="missing non-empty message content"):
            teachers_mod._extract_openai_message_text({"choices": [{"message": {"content": None}}]})

    def test_extract_anthropic_text_handles_errors(self) -> None:
        with pytest.raises(TeacherInvocationError, match="missing content blocks"):
            teachers_mod._extract_anthropic_text({})

        with pytest.raises(TeacherInvocationError, match="missing non-empty text blocks"):
            teachers_mod._extract_anthropic_text(
                {"content": [{"type": "image", "text": "ignored"}, {"type": "text", "text": "   "}]}
            )

    def test_normalize_chat_content_and_obj_get_helpers(self) -> None:
        assert teachers_mod._normalize_chat_content(" hello ") == "hello"
        assert (
            teachers_mod._normalize_chat_content([{"text": " one "}, {"text": " two "}])
            == "one\ntwo"
        )
        assert teachers_mod._normalize_chat_content([{"text": "   "}]) is None
        assert teachers_mod._normalize_chat_content(123) is None
        assert teachers_mod._obj_get({"name": "value"}, "name") == "value"
        assert teachers_mod._obj_get(SimpleNamespace(name="value"), "name") == "value"

    def test_openai_compat_url_helpers_normalize_suffixes(self) -> None:
        assert (
            teachers_mod._normalize_openai_compat_base_url(
                "http://127.0.0.1:8000/v1/chat/completions"
            )
            == "http://127.0.0.1:8000"
        )
        assert (
            teachers_mod._normalize_openai_compat_base_url("http://127.0.0.1:8000/chat/completions")
            == "http://127.0.0.1:8000"
        )
        assert teachers_mod._openai_compat_models_url("http://127.0.0.1:8000/v1") == (
            "http://127.0.0.1:8000/v1/models"
        )
        assert teachers_mod._openai_compat_models_url("http://127.0.0.1:8000") == (
            "http://127.0.0.1:8000/v1/models"
        )
        assert teachers_mod._openai_compat_chat_url("http://127.0.0.1:8000/v1") == (
            "http://127.0.0.1:8000/v1/chat/completions"
        )
        assert teachers_mod._openai_compat_chat_url("http://127.0.0.1:8000") == (
            "http://127.0.0.1:8000/v1/chat/completions"
        )
