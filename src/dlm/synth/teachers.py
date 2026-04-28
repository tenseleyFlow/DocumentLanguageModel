"""Teacher selector parsing and runtime wrappers for synthetic-data generation."""

import importlib
import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable
from urllib.parse import urlparse

from dlm.synth.errors import (
    InvalidTeacherSpecError,
    TeacherInvocationError,
    TeacherUnavailableError,
)

_log = logging.getLogger(__name__)

TeacherKind = Literal["self", "hf", "openai", "anthropic", "vllm-server"]

_DEFAULT_MAX_NEW_TOKENS = 512
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
_ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"


@dataclass
class TeacherUsage:
    """Accumulated token usage from API-backed teachers."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def log_summary(self, teacher_name: str) -> None:
        if self.requests == 0:
            return
        _log.info(
            "teacher %s usage: %d requests, %d prompt tokens, "
            "%d completion tokens, %d total tokens",
            teacher_name,
            self.requests,
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
        )


@dataclass(frozen=True)
class TeacherRef:
    """Parsed `--teacher` selector from the CLI."""

    raw: str
    kind: TeacherKind
    target: str | None = None


@runtime_checkable
class SynthTeacher(Protocol):
    """Runtime teacher contract used by `dlm synth`."""

    @property
    def name(self) -> str:
        """Stable user-facing teacher identifier."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str:
        """Generate one raw teacher response."""


@runtime_checkable
class _GenerateBackend(Protocol):
    """Minimal generation surface for `SelfTeacher`."""

    def generate(self, prompt: str, **gen_kwargs: Any) -> str:
        """Return one completion for `prompt`."""


SelfTeacherLoader = Callable[[Path, str], _GenerateBackend]
HfTeacherLoader = Callable[[str, str], "_LoadedHfTeacher"]
HfGenerateRunner = Callable[..., str]
OpenAiClientFactory = Callable[[str], Any]
AnthropicClientFactory = Callable[[str], Any]


@dataclass(frozen=True)
class SelfTeacher:
    """Current-adapter self-instruct teacher."""

    dlm_path: Path
    backend: Literal["auto", "pytorch", "mlx"] = "auto"
    loader: SelfTeacherLoader | None = field(default=None, repr=False, compare=False)
    name: str = field(default="self", init=False)
    _loaded: _GenerateBackend | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dlm_path", Path(self.dlm_path).expanduser())

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str:
        _ = seed  # The current inference backend wrapper does not accept seeding.
        backend = self._ensure_loaded()
        prompt = _flatten_teacher_prompt(system_prompt, user_prompt)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        return _require_non_empty_teacher_output(
            backend.generate(prompt, **kwargs),
            teacher=self.name,
        )

    def _ensure_loaded(self) -> _GenerateBackend:
        if self._loaded is not None:
            return self._loaded
        loader = self.loader if self.loader is not None else _load_self_backend
        loaded = loader(self.dlm_path, self.backend)
        object.__setattr__(self, "_loaded", loaded)
        return loaded


@dataclass(frozen=True)
class _LoadedHfTeacher:
    """Loaded HF text-generation bundle cached inside `HfTeacher`."""

    model: Any
    tokenizer: Any
    device: str


@dataclass(frozen=True)
class HfTeacher:
    """HF text-generation teacher."""

    hf_id: str
    device: str = "auto"
    loader: HfTeacherLoader | None = field(default=None, repr=False, compare=False)
    runner: HfGenerateRunner | None = field(default=None, repr=False, compare=False)
    name: str = field(init=False)
    _loaded: _LoadedHfTeacher | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        spec = self.hf_id.strip()
        if not spec:
            raise InvalidTeacherSpecError("hf teacher selector must include a model id")
        object.__setattr__(self, "hf_id", spec)
        object.__setattr__(self, "name", f"hf:{spec}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str:
        loaded = self._ensure_loaded()
        prompt = _flatten_teacher_prompt(system_prompt, user_prompt)
        runner = self.runner if self.runner is not None else _default_hf_generate
        text = runner(
            loaded.model,
            loaded.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        return _require_non_empty_teacher_output(text, teacher=self.name)

    def _ensure_loaded(self) -> _LoadedHfTeacher:
        if self._loaded is not None:
            return self._loaded
        device = _resolve_generation_device(self.device)
        loader = self.loader if self.loader is not None else _default_hf_loader
        loaded = loader(self.hf_id, device)
        object.__setattr__(self, "_loaded", loaded)
        return loaded


@dataclass(frozen=True)
class OpenAiTeacher:
    """OpenAI chat-completions teacher."""

    model: str
    client_factory: OpenAiClientFactory | None = field(default=None, repr=False, compare=False)
    api_key_env: str = field(default=_OPENAI_API_KEY_ENV, repr=False, compare=False)
    name: str = field(init=False)
    usage: TeacherUsage = field(default_factory=TeacherUsage, init=False, repr=False, compare=False)
    _client: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        model = self.model.strip()
        if not model:
            raise InvalidTeacherSpecError("openai teacher selector must include a model id")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "name", f"openai:{model}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str:
        client = self._ensure_client()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if seed is not None:
            payload["seed"] = seed
        try:
            response = client.chat.completions.create(**payload)
        except Exception as exc:
            raise TeacherInvocationError(f"{self.name} request failed: {exc}") from exc
        _accumulate_openai_usage(self.usage, response)
        return _require_non_empty_teacher_output(
            _extract_openai_message_text(response),
            teacher=self.name,
        )

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise TeacherUnavailableError(f"{self.name} requires ${self.api_key_env} to be set")
        factory = self.client_factory if self.client_factory is not None else _default_openai_client
        client = factory(api_key)
        object.__setattr__(self, "_client", client)
        return client


@dataclass(frozen=True)
class AnthropicTeacher:
    """Anthropic messages API teacher."""

    model: str
    client_factory: AnthropicClientFactory | None = field(default=None, repr=False, compare=False)
    api_key_env: str = field(default=_ANTHROPIC_API_KEY_ENV, repr=False, compare=False)
    name: str = field(init=False)
    usage: TeacherUsage = field(default_factory=TeacherUsage, init=False, repr=False, compare=False)
    _client: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        model = self.model.strip()
        if not model:
            raise InvalidTeacherSpecError("anthropic teacher selector must include a model id")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "name", f"anthropic:{model}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str:
        _ = seed  # Anthropic messages API does not currently expose seeding here.
        client = self._ensure_client()
        payload: dict[str, Any] = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        try:
            response = client.messages.create(**payload)
        except Exception as exc:
            raise TeacherInvocationError(f"{self.name} request failed: {exc}") from exc
        _accumulate_anthropic_usage(self.usage, response)
        return _require_non_empty_teacher_output(
            _extract_anthropic_text(response),
            teacher=self.name,
        )

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise TeacherUnavailableError(f"{self.name} requires ${self.api_key_env} to be set")
        factory = (
            self.client_factory if self.client_factory is not None else _default_anthropic_client
        )
        client = factory(api_key)
        object.__setattr__(self, "_client", client)
        return client


@dataclass(frozen=True)
class VllmServerTeacher:
    """OpenAI-compatible teacher backed by a vLLM server endpoint."""

    url: str
    request_timeout: float = _DEFAULT_REQUEST_TIMEOUT_SECONDS
    name: str = field(init=False)
    _base_url: str = field(init=False, repr=False)
    _model_id: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        spec = self.url.strip()
        if not spec:
            raise InvalidTeacherSpecError("vllm-server teacher selector must include a URL")
        base = _normalize_openai_compat_base_url(spec)
        object.__setattr__(self, "url", spec)
        object.__setattr__(self, "_base_url", base)
        object.__setattr__(self, "name", f"vllm-server:{spec}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str:
        model_id = self._ensure_model_id()
        text = _request_openai_compat_completion(
            self._base_url,
            model_id=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            request_timeout=self.request_timeout,
        )
        return _require_non_empty_teacher_output(text, teacher=self.name)

    def _ensure_model_id(self) -> str | None:
        if self._model_id is not None:
            return self._model_id
        model_id = _fetch_openai_compat_model_id(
            self._base_url,
            request_timeout=self.request_timeout,
        )
        object.__setattr__(self, "_model_id", model_id)
        return model_id


def parse_teacher_ref(raw: str) -> TeacherRef:
    """Parse `self`, `hf:<model>`, `openai:<model>`, `anthropic:<model>`, or `vllm-server:<url>`."""
    spec = raw.strip()
    if not spec:
        raise InvalidTeacherSpecError("teacher selector must not be empty")
    if spec == "self":
        return TeacherRef(raw=spec, kind="self", target=None)
    if spec.startswith("hf:"):
        target = spec.removeprefix("hf:").strip()
        if not target:
            raise InvalidTeacherSpecError("hf teacher selector must include a model id")
        # Resolve dlm-registry aliases (e.g. `hf:smollm2-135m`) to their
        # canonical HuggingFace id. Keeps `--teacher hf:<key>` consistent
        # with `dlm init --base <key>`. A literal HF id (`org/repo`)
        # passes through unchanged.
        from dlm.base_models.registry import BASE_MODELS

        spec_entry = BASE_MODELS.get(target)
        if spec_entry is not None:
            target = spec_entry.hf_id
        return TeacherRef(raw=spec, kind="hf", target=target)
    if spec.startswith("openai:"):
        target = spec.removeprefix("openai:").strip()
        if not target:
            raise InvalidTeacherSpecError("openai teacher selector must include a model id")
        return TeacherRef(raw=spec, kind="openai", target=target)
    if spec.startswith("anthropic:"):
        target = spec.removeprefix("anthropic:").strip()
        if not target:
            raise InvalidTeacherSpecError("anthropic teacher selector must include a model id")
        return TeacherRef(raw=spec, kind="anthropic", target=target)
    if spec.startswith("vllm-server:"):
        target = spec.removeprefix("vllm-server:").strip()
        if not target:
            raise InvalidTeacherSpecError("vllm-server teacher selector must include a URL")
        return TeacherRef(raw=spec, kind="vllm-server", target=target)
    raise InvalidTeacherSpecError(
        "unknown teacher selector "
        f"{raw!r}; expected 'self', 'hf:<model>', 'openai:<model>', "
        "'anthropic:<model>', or 'vllm-server:<url>'"
    )


def build_teacher(raw: str | TeacherRef, *, dlm_path: Path | None = None) -> SynthTeacher:
    """Instantiate the concrete teacher for `raw`."""
    ref = parse_teacher_ref(raw) if isinstance(raw, str) else raw
    if ref.kind == "self":
        if dlm_path is None:
            raise TeacherUnavailableError("self teacher requires the .dlm path context")
        # The synth CLI wants the most conservative backend choice for the
        # adapter-under-test path. PyTorch is slower than MLX on Apple
        # Silicon, but it avoids bringing a second runtime stack into a
        # write-heavy loop whose main requirement is stable, portable
        # generation rather than lowest-latency prompting.
        return SelfTeacher(dlm_path, backend="pytorch")
    if ref.kind == "hf":
        assert ref.target is not None
        return HfTeacher(ref.target)
    if ref.kind == "openai":
        assert ref.target is not None
        return OpenAiTeacher(ref.target)
    if ref.kind == "anthropic":
        assert ref.target is not None
        return AnthropicTeacher(ref.target)
    assert ref.target is not None
    return VllmServerTeacher(ref.target)


def _load_self_backend(dlm_path: Path, backend: str) -> _GenerateBackend:
    try:
        from dlm.base_models import resolve as resolve_base_model
        from dlm.base_models.errors import GatedModelError
        from dlm.doc.parser import parse_file
        from dlm.hardware import doctor
        from dlm.inference import AdapterNotFoundError
        from dlm.inference.backends import build_backend, select_backend
        from dlm.inference.backends.select import UnsupportedBackendError
        from dlm.store.errors import ManifestCorruptError
        from dlm.store.manifest import load_manifest
        from dlm.store.paths import for_dlm
    except ImportError as exc:
        raise TeacherUnavailableError("self teacher requires the local inference stack") from exc

    parsed = parse_file(dlm_path)
    store = for_dlm(parsed.frontmatter.dlm_id)
    accepted = False
    if store.manifest.exists():
        try:
            accepted = load_manifest(store.manifest).license_acceptance is not None
        except (ManifestCorruptError, OSError):
            accepted = False

    try:
        spec = resolve_base_model(parsed.frontmatter.base_model, accept_license=accepted)
    except GatedModelError as exc:
        raise TeacherUnavailableError(
            f"self teacher cannot resolve gated base {parsed.frontmatter.base_model!r} "
            "without a recorded license acceptance"
        ) from exc

    caps = doctor().capabilities
    try:
        backend_name = select_backend(backend, caps)  # type: ignore[arg-type]
    except UnsupportedBackendError as exc:
        raise TeacherUnavailableError(str(exc)) from exc
    backend_obj = build_backend(backend_name, caps)
    try:
        backend_obj.load(spec, store)
    except AdapterNotFoundError as exc:
        raise TeacherUnavailableError(
            f"self teacher requires a trained adapter for {dlm_path}"
        ) from exc
    return cast(_GenerateBackend, backend_obj)


def _default_hf_loader(hf_id: str, device: str) -> _LoadedHfTeacher:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise TeacherUnavailableError("hf teacher requires transformers to be installed") from exc

    model: Any = AutoModelForCausalLM.from_pretrained(hf_id)
    tokenizer: Any = AutoTokenizer.from_pretrained(hf_id)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return _LoadedHfTeacher(model=model, tokenizer=tokenizer, device=device)


def _default_hf_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
    seed: int | None,
) -> str:
    from dlm.inference.generate import generate as hf_generate

    if seed is not None:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    return hf_generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def _resolve_generation_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_openai_client(api_key: str) -> Any:
    try:
        openai_mod = importlib.import_module("openai")
    except ImportError as exc:
        raise TeacherUnavailableError(
            "openai teacher requires the openai package to be installed"
        ) from exc
    openai_cls = getattr(openai_mod, "OpenAI", None)
    if openai_cls is None:
        raise TeacherUnavailableError("openai package does not expose OpenAI client")
    return openai_cls(api_key=api_key)


def _default_anthropic_client(api_key: str) -> Any:
    try:
        anthropic_mod = importlib.import_module("anthropic")
    except ImportError as exc:
        raise TeacherUnavailableError(
            "anthropic teacher requires the anthropic package to be installed"
        ) from exc
    anthropic_cls = getattr(anthropic_mod, "Anthropic", None)
    if anthropic_cls is None:
        raise TeacherUnavailableError("anthropic package does not expose Anthropic client")
    return anthropic_cls(api_key=api_key)


def _flatten_teacher_prompt(system_prompt: str, user_prompt: str) -> str:
    system = system_prompt.strip()
    user = user_prompt.strip()
    if system and user:
        return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"
    if user:
        return user
    return system


def _require_non_empty_teacher_output(output: str, *, teacher: str) -> str:
    text = output.strip()
    if not text:
        raise TeacherInvocationError(f"{teacher} returned empty output")
    return text


def _extract_openai_message_text(response: Any) -> str:
    choices = _obj_get(response, "choices")
    if not isinstance(choices, list) or not choices:
        raise TeacherInvocationError("openai teacher response missing choices")
    message = _obj_get(choices[0], "message")
    if message is None:
        raise TeacherInvocationError("openai teacher response missing choices[0].message")
    content = _obj_get(message, "content")
    text = _normalize_chat_content(content)
    if text is None:
        raise TeacherInvocationError("openai teacher response missing non-empty message content")
    return text


def _extract_anthropic_text(response: Any) -> str:
    content = _obj_get(response, "content")
    if not isinstance(content, list) or not content:
        raise TeacherInvocationError("anthropic teacher response missing content blocks")
    parts: list[str] = []
    for block in content:
        if _obj_get(block, "type") != "text":
            continue
        text = _obj_get(block, "text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    merged = "\n".join(parts).strip()
    if not merged:
        raise TeacherInvocationError("anthropic teacher response missing non-empty text blocks")
    return merged


def _normalize_chat_content(content: object) -> str | None:
    if isinstance(content, str):
        stripped = content.strip()
        return stripped if stripped else None
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _obj_get(item, "text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        merged = "\n".join(parts).strip()
        return merged if merged else None
    return None


def _obj_get(obj: object, name: str) -> object:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _accumulate_openai_usage(usage: TeacherUsage, response: Any) -> None:
    usage.requests += 1
    u = _obj_get(response, "usage")
    if u is None:
        return
    pt = _obj_get(u, "prompt_tokens")
    ct = _obj_get(u, "completion_tokens")
    if isinstance(pt, int):
        usage.prompt_tokens += pt
    if isinstance(ct, int):
        usage.completion_tokens += ct


def _accumulate_anthropic_usage(usage: TeacherUsage, response: Any) -> None:
    usage.requests += 1
    u = _obj_get(response, "usage")
    if u is None:
        return
    pt = _obj_get(u, "input_tokens")
    ct = _obj_get(u, "output_tokens")
    if isinstance(pt, int):
        usage.prompt_tokens += pt
    if isinstance(ct, int):
        usage.completion_tokens += ct


def _normalize_openai_compat_base_url(url: str) -> str:
    stripped = url.rstrip("/")
    if stripped.endswith("/v1/chat/completions"):
        stripped = stripped[: -len("/v1/chat/completions")]
    elif stripped.endswith("/chat/completions"):
        stripped = stripped[: -len("/chat/completions")]
    parsed = urlparse(stripped)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise InvalidTeacherSpecError(
            f"vllm-server teacher URL must be http(s) with a host, got {url!r}"
        )
    return stripped


def _fetch_openai_compat_model_id(base_url: str, *, request_timeout: float) -> str | None:
    req = urllib.request.Request(
        _openai_compat_models_url(base_url),
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=request_timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read())
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise TeacherUnavailableError(
            f"vllm-server teacher could not query models from {base_url}: {exc}"
        ) from exc
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    model_id = _obj_get(data[0], "id")
    return model_id if isinstance(model_id, str) and model_id.strip() else None


def _request_openai_compat_completion(
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
    payload: dict[str, Any] = {
        "model": model_id or "dlm-synth",
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if seed is not None:
        payload["seed"] = seed

    req = urllib.request.Request(
        _openai_compat_chat_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=request_timeout) as resp:  # noqa: S310
            body = json.loads(resp.read())
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise TeacherInvocationError(
            f"vllm-server teacher request to {base_url} failed: {exc}"
        ) from exc

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TeacherInvocationError("vllm-server teacher response missing choices")
    message = _obj_get(choices[0], "message")
    if message is None:
        raise TeacherInvocationError("vllm-server teacher response missing choices[0].message")
    content = _normalize_chat_content(_obj_get(message, "content"))
    if content is None:
        raise TeacherInvocationError(
            "vllm-server teacher response missing non-empty message content"
        )
    return content


def _openai_compat_models_url(base_url: str) -> str:
    if base_url.endswith("/v1"):
        return base_url + "/models"
    return base_url + "/v1/models"


def _openai_compat_chat_url(base_url: str) -> str:
    if base_url.endswith("/v1"):
        return base_url + "/chat/completions"
    return base_url + "/v1/chat/completions"
