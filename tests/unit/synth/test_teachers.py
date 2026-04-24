"""Teacher selector parsing and runtime wrappers for Sprint 43."""

from __future__ import annotations

import builtins
import json
import sys
import urllib.error
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Literal

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


def _module(name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


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


class TestTeacherRuntimeHelpers:
    def test_resolve_generation_device_prefers_requested_or_detected_backends(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        assert teachers_mod._resolve_generation_device("mps") == "mps"

        monkeypatch.delitem(sys.modules, "torch", raising=False)
        real_import = builtins.__import__

        def _missing_torch(
            name: str,
            globals: dict[str, object] | None = None,
            locals: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _missing_torch)
        assert teachers_mod._resolve_generation_device("auto") == "cpu"

        monkeypatch.setattr(builtins, "__import__", real_import)
        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: True),
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
            ),
        )
        assert teachers_mod._resolve_generation_device("auto") == "cuda"

        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: False),
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
            ),
        )
        assert teachers_mod._resolve_generation_device("auto") == "mps"

        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: False),
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
            ),
        )
        assert teachers_mod._resolve_generation_device("auto") == "cpu"

    def test_default_openai_client_validates_import_surface(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise_import(name: str) -> object:
            raise ImportError(name)

        monkeypatch.setattr("dlm.synth.teachers.importlib.import_module", _raise_import)
        with pytest.raises(TeacherUnavailableError, match="requires the openai package"):
            teachers_mod._default_openai_client("secret")

        monkeypatch.setattr(
            "dlm.synth.teachers.importlib.import_module", lambda _name: SimpleNamespace()
        )
        with pytest.raises(TeacherUnavailableError, match="does not expose OpenAI client"):
            teachers_mod._default_openai_client("secret")

        captured: list[str] = []

        class _OpenAI:
            def __init__(self, *, api_key: str) -> None:
                captured.append(api_key)

        monkeypatch.setattr(
            "dlm.synth.teachers.importlib.import_module",
            lambda _name: SimpleNamespace(OpenAI=_OpenAI),
        )
        client = teachers_mod._default_openai_client("secret")
        assert isinstance(client, _OpenAI)
        assert captured == ["secret"]

    def test_default_anthropic_client_validates_import_surface(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise_import(name: str) -> object:
            raise ImportError(name)

        monkeypatch.setattr("dlm.synth.teachers.importlib.import_module", _raise_import)
        with pytest.raises(TeacherUnavailableError, match="requires the anthropic package"):
            teachers_mod._default_anthropic_client("secret")

        monkeypatch.setattr(
            "dlm.synth.teachers.importlib.import_module", lambda _name: SimpleNamespace()
        )
        with pytest.raises(TeacherUnavailableError, match="does not expose Anthropic client"):
            teachers_mod._default_anthropic_client("secret")

        captured: list[str] = []

        class _Anthropic:
            def __init__(self, *, api_key: str) -> None:
                captured.append(api_key)

        monkeypatch.setattr(
            "dlm.synth.teachers.importlib.import_module",
            lambda _name: SimpleNamespace(Anthropic=_Anthropic),
        )
        client = teachers_mod._default_anthropic_client("secret")
        assert isinstance(client, _Anthropic)
        assert captured == ["secret"]

    def test_fetch_openai_compat_model_id_handles_success_empty_and_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _Response:
            def __init__(self, payload: object) -> None:
                self._payload = payload

            def __enter__(self) -> _Response:
                return self

            def __exit__(self, *_args: object) -> Literal[False]:
                return False

            def read(self) -> bytes:
                return json.dumps(self._payload).encode("utf-8")

        monkeypatch.setattr(
            "dlm.synth.teachers.urllib.request.urlopen",
            lambda *_args, **_kwargs: _Response({"data": [{"id": "demo-model"}]}),
        )
        assert (
            teachers_mod._fetch_openai_compat_model_id(
                "http://127.0.0.1:8000",
                request_timeout=1.0,
            )
            == "demo-model"
        )

        monkeypatch.setattr(
            "dlm.synth.teachers.urllib.request.urlopen",
            lambda *_args, **_kwargs: _Response({"data": []}),
        )
        assert (
            teachers_mod._fetch_openai_compat_model_id(
                "http://127.0.0.1:8000",
                request_timeout=1.0,
            )
            is None
        )

        monkeypatch.setattr(
            "dlm.synth.teachers.urllib.request.urlopen",
            lambda *_args, **_kwargs: _Response({"data": [{"id": "   "}]}),
        )
        assert (
            teachers_mod._fetch_openai_compat_model_id(
                "http://127.0.0.1:8000",
                request_timeout=1.0,
            )
            is None
        )

        def _raise_url_error(*_args: object, **_kwargs: object) -> object:
            raise urllib.error.URLError("boom")

        monkeypatch.setattr("dlm.synth.teachers.urllib.request.urlopen", _raise_url_error)
        with pytest.raises(TeacherUnavailableError, match="could not query models"):
            teachers_mod._fetch_openai_compat_model_id(
                "http://127.0.0.1:8000",
                request_timeout=1.0,
            )

    def test_request_openai_compat_completion_handles_success_and_failures(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _Response:
            def __init__(self, payload: object) -> None:
                self._payload = payload

            def __enter__(self) -> _Response:
                return self

            def __exit__(self, *_args: object) -> Literal[False]:
                return False

            def read(self) -> bytes:
                return json.dumps(self._payload).encode("utf-8")

        monkeypatch.setattr(
            "dlm.synth.teachers.urllib.request.urlopen",
            lambda *_args, **_kwargs: _Response(
                {"choices": [{"message": {"content": [{"text": " served "}]}}]}
            ),
        )
        assert (
            teachers_mod._request_openai_compat_completion(
                "http://127.0.0.1:8000",
                model_id="demo-model",
                messages=[{"role": "user", "content": "hello"}],
                max_new_tokens=11,
                temperature=0.2,
                top_p=0.8,
                seed=5,
                request_timeout=1.0,
            )
            == "served"
        )

        monkeypatch.setattr(
            "dlm.synth.teachers.urllib.request.urlopen",
            lambda *_args, **_kwargs: _Response({"choices": []}),
        )
        with pytest.raises(TeacherInvocationError, match="response missing choices"):
            teachers_mod._request_openai_compat_completion(
                "http://127.0.0.1:8000",
                model_id=None,
                messages=[{"role": "user", "content": "hello"}],
                max_new_tokens=11,
                temperature=0.2,
                top_p=None,
                seed=None,
                request_timeout=1.0,
            )

        monkeypatch.setattr(
            "dlm.synth.teachers.urllib.request.urlopen",
            lambda *_args, **_kwargs: _Response({"choices": [{}]}),
        )
        with pytest.raises(
            TeacherInvocationError, match="response missing choices\\[0\\]\\.message"
        ):
            teachers_mod._request_openai_compat_completion(
                "http://127.0.0.1:8000",
                model_id=None,
                messages=[{"role": "user", "content": "hello"}],
                max_new_tokens=11,
                temperature=0.2,
                top_p=None,
                seed=None,
                request_timeout=1.0,
            )

        monkeypatch.setattr(
            "dlm.synth.teachers.urllib.request.urlopen",
            lambda *_args, **_kwargs: _Response(
                {"choices": [{"message": {"content": [{"text": "   "}]}}]}
            ),
        )
        with pytest.raises(TeacherInvocationError, match="missing non-empty message content"):
            teachers_mod._request_openai_compat_completion(
                "http://127.0.0.1:8000",
                model_id=None,
                messages=[{"role": "user", "content": "hello"}],
                max_new_tokens=11,
                temperature=0.2,
                top_p=None,
                seed=None,
                request_timeout=1.0,
            )

        def _raise_url_error(*_args: object, **_kwargs: object) -> object:
            raise urllib.error.URLError("boom")

        monkeypatch.setattr("dlm.synth.teachers.urllib.request.urlopen", _raise_url_error)
        with pytest.raises(TeacherInvocationError, match="request to http://127.0.0.1:8000 failed"):
            teachers_mod._request_openai_compat_completion(
                "http://127.0.0.1:8000",
                model_id=None,
                messages=[{"role": "user", "content": "hello"}],
                max_new_tokens=11,
                temperature=0.2,
                top_p=None,
                seed=None,
                request_timeout=1.0,
            )


def _install_self_loader_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest_exists: bool = True,
    license_acceptance: object | None = "accepted",
    load_manifest_error: str | None = None,
    resolve_error: str | None = None,
    select_error: str | None = None,
    backend_load_error: str | None = None,
) -> dict[str, object]:
    calls: dict[str, object] = {}
    spec = object()
    caps = object()
    parsed = SimpleNamespace(
        frontmatter=SimpleNamespace(
            dlm_id="01KPQ9X1000000000000000000",
            base_model="smollm2-135m",
        )
    )
    manifest = SimpleNamespace(exists=lambda: manifest_exists)
    store = SimpleNamespace(manifest=manifest)

    class GatedModelError(Exception):
        pass

    class AdapterNotFoundError(Exception):
        pass

    class UnsupportedBackendError(Exception):
        pass

    class ManifestCorruptError(Exception):
        pass

    class _Backend:
        def load(self, spec_arg: object, store_arg: object) -> None:
            calls["load"] = (spec_arg, store_arg)
            if backend_load_error is not None:
                raise AdapterNotFoundError(backend_load_error)

    backend = _Backend()

    def _resolve(base_model: str, *, accept_license: bool) -> object:
        calls["resolve"] = (base_model, accept_license)
        if resolve_error is not None:
            raise GatedModelError(resolve_error)
        return spec

    def _load_manifest(_path: object) -> object:
        calls["load_manifest"] = True
        if load_manifest_error is not None:
            raise ManifestCorruptError(load_manifest_error)
        return SimpleNamespace(license_acceptance=license_acceptance)

    def _select_backend(backend_name: str, capabilities: object) -> str:
        calls["select_backend"] = (backend_name, capabilities)
        if select_error is not None:
            raise UnsupportedBackendError(select_error)
        return "stub-backend"

    def _build_backend(name: str, capabilities: object) -> object:
        calls["build_backend"] = (name, capabilities)
        return backend

    monkeypatch.setitem(
        sys.modules, "dlm.base_models", _module("dlm.base_models", resolve=_resolve)
    )
    monkeypatch.setitem(
        sys.modules,
        "dlm.base_models.errors",
        _module("dlm.base_models.errors", GatedModelError=GatedModelError),
    )
    monkeypatch.setitem(
        sys.modules,
        "dlm.doc.parser",
        _module("dlm.doc.parser", parse_file=lambda _path: parsed),
    )
    monkeypatch.setitem(
        sys.modules,
        "dlm.hardware",
        _module("dlm.hardware", doctor=lambda: SimpleNamespace(capabilities=caps)),
    )
    monkeypatch.setitem(
        sys.modules,
        "dlm.inference",
        _module("dlm.inference", AdapterNotFoundError=AdapterNotFoundError),
    )
    monkeypatch.setitem(
        sys.modules,
        "dlm.inference.backends",
        _module(
            "dlm.inference.backends", build_backend=_build_backend, select_backend=_select_backend
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "dlm.inference.backends.select",
        _module("dlm.inference.backends.select", UnsupportedBackendError=UnsupportedBackendError),
    )
    monkeypatch.setitem(
        sys.modules,
        "dlm.store.errors",
        _module("dlm.store.errors", ManifestCorruptError=ManifestCorruptError),
    )
    monkeypatch.setitem(
        sys.modules,
        "dlm.store.manifest",
        _module("dlm.store.manifest", load_manifest=_load_manifest),
    )
    monkeypatch.setitem(
        sys.modules,
        "dlm.store.paths",
        _module("dlm.store.paths", for_dlm=lambda _dlm_id: store),
    )

    calls["caps"] = caps
    calls["store"] = store
    calls["spec"] = spec
    calls["errors"] = {
        "gated": GatedModelError,
        "adapter": AdapterNotFoundError,
        "unsupported": UnsupportedBackendError,
        "manifest": ManifestCorruptError,
    }
    return calls


class TestTeacherLoaderHelpers:
    def test_load_self_backend_wraps_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def _raise_on_base_models(
            name: str,
            globals: dict[str, object] | None = None,
            locals: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name.startswith("dlm.base_models"):
                raise ImportError("boom")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _raise_on_base_models)
        with pytest.raises(TeacherUnavailableError, match="requires the local inference stack"):
            teachers_mod._load_self_backend(Path("/tmp/doc.dlm"), "auto")

    def test_load_self_backend_uses_recorded_license_acceptance(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = _install_self_loader_modules(monkeypatch, license_acceptance="accepted")

        backend = teachers_mod._load_self_backend(Path("/tmp/doc.dlm"), "auto")

        assert backend is not None
        assert calls["resolve"] == ("smollm2-135m", True)
        assert calls["select_backend"] == ("auto", calls["caps"])
        assert calls["build_backend"] == ("stub-backend", calls["caps"])
        assert calls["load"] == (calls["spec"], calls["store"])

    def test_load_self_backend_tolerates_manifest_read_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = _install_self_loader_modules(
            monkeypatch,
            load_manifest_error="bad manifest",
        )

        teachers_mod._load_self_backend(Path("/tmp/doc.dlm"), "auto")

        assert calls["resolve"] == ("smollm2-135m", False)

    def test_load_self_backend_wraps_gated_backend_and_adapter_failures(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_self_loader_modules(monkeypatch, resolve_error="gated")
        with pytest.raises(TeacherUnavailableError, match="cannot resolve gated base"):
            teachers_mod._load_self_backend(Path("/tmp/doc.dlm"), "auto")

        _install_self_loader_modules(monkeypatch, select_error="unsupported backend")
        with pytest.raises(TeacherUnavailableError, match="unsupported backend"):
            teachers_mod._load_self_backend(Path("/tmp/doc.dlm"), "auto")

        _install_self_loader_modules(monkeypatch, backend_load_error="missing adapter")
        with pytest.raises(TeacherUnavailableError, match="requires a trained adapter"):
            teachers_mod._load_self_backend(Path("/tmp/doc.dlm"), "auto")

    def test_default_hf_loader_wraps_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def _raise_transformers(
            name: str,
            globals: dict[str, object] | None = None,
            locals: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "transformers":
                raise ImportError("boom")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _raise_transformers)
        with pytest.raises(TeacherUnavailableError, match="requires transformers"):
            teachers_mod._default_hf_loader("hf/model", "cpu")

    def test_default_hf_loader_moves_model_and_sets_eval(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: dict[str, object] = {}

        class _Model:
            def to(self, device: str) -> _Model:
                seen["device"] = device
                return self

            def eval(self) -> None:
                seen["eval"] = True

        model = _Model()

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(hf_id: str) -> _Model:
                seen["model_id"] = hf_id
                return model

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(hf_id: str) -> str:
                seen["tokenizer_id"] = hf_id
                return "tok"

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            _module(
                "transformers",
                AutoModelForCausalLM=AutoModelForCausalLM,
                AutoTokenizer=AutoTokenizer,
            ),
        )

        loaded = teachers_mod._default_hf_loader("hf/model", "cuda")

        assert loaded.model is model
        assert loaded.tokenizer == "tok"
        assert loaded.device == "cuda"
        assert seen == {
            "model_id": "hf/model",
            "tokenizer_id": "hf/model",
            "device": "cuda",
            "eval": True,
        }

    def test_default_hf_generate_seeds_torch_and_calls_runner(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manual: list[int] = []
        manual_all: list[int] = []
        calls: dict[str, object] = {}

        def _generate(
            model: object,
            tokenizer: object,
            prompt: str,
            *,
            max_new_tokens: int,
            temperature: float,
            top_p: float | None,
        ) -> str:
            calls["args"] = (model, tokenizer, prompt, max_new_tokens, temperature, top_p)
            return "ok"

        monkeypatch.setitem(
            sys.modules,
            "dlm.inference.generate",
            _module("dlm.inference.generate", generate=_generate),
        )
        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(
                manual_seed=lambda seed: manual.append(seed),
                cuda=SimpleNamespace(
                    is_available=lambda: True,
                    manual_seed_all=lambda seed: manual_all.append(seed),
                ),
            ),
        )

        out = teachers_mod._default_hf_generate(
            "model",
            "tokenizer",
            "prompt",
            max_new_tokens=17,
            temperature=0.3,
            top_p=0.8,
            seed=7,
        )

        assert out == "ok"
        assert manual == [7]
        assert manual_all == [7]
        assert calls["args"] == ("model", "tokenizer", "prompt", 17, 0.3, 0.8)

    def test_default_hf_generate_tolerates_missing_torch_when_seeding(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        real_import = builtins.__import__

        def _generate(
            model: object,
            tokenizer: object,
            prompt: str,
            *,
            max_new_tokens: int,
            temperature: float,
            top_p: float | None,
        ) -> str:
            _ = model, tokenizer, prompt, max_new_tokens, temperature, top_p
            return "ok"

        def _raise_torch(
            name: str,
            globals: dict[str, object] | None = None,
            locals: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setitem(
            sys.modules,
            "dlm.inference.generate",
            _module("dlm.inference.generate", generate=_generate),
        )
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        monkeypatch.setattr(builtins, "__import__", _raise_torch)

        out = teachers_mod._default_hf_generate(
            "model",
            "tokenizer",
            "prompt",
            max_new_tokens=17,
            temperature=0.3,
            top_p=0.8,
            seed=7,
        )

        assert out == "ok"
