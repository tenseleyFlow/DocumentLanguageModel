"""Typed judge protocol, selector parsing, and concrete judge runtimes."""

from __future__ import annotations

import json
import math
import shlex
import subprocess  # nosec B404
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from dlm.preference.errors import (
    InvalidJudgeSpecError,
    JudgeInvocationError,
    JudgeUnavailableError,
)

JudgeKind = Literal["sway", "hf", "cli"]

_DEFAULT_SWAY_THRESHOLD = 0.1
_DEFAULT_CLI_THRESHOLD = 0.1
_DEFAULT_CLI_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class PairScore:
    """Judge output for a two-candidate comparison."""

    score_a: float
    score_b: float
    reasoning: str | None = None
    margin: float = field(init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.score_a) or not math.isfinite(self.score_b):
            raise ValueError("judge scores must be finite floats")
        object.__setattr__(self, "margin", self.score_a - self.score_b)

    @property
    def preferred(self) -> Literal["a", "b", "tie"]:
        if self.score_a > self.score_b:
            return "a"
        if self.score_b > self.score_a:
            return "b"
        return "tie"


@dataclass(frozen=True)
class JudgeRef:
    """Parsed `--judge` selector from the CLI."""

    raw: str
    kind: JudgeKind
    target: str | None = None


@runtime_checkable
class PreferenceJudge(Protocol):
    """Runtime judge contract used by the mine loop."""

    @property
    def name(self) -> str:
        """Stable user-facing judge identifier."""

    @property
    def suggested_threshold(self) -> float:
        """Default minimum margin on this judge's native scale."""

    def score_pair(self, prompt: str, candidate_a: str, candidate_b: str) -> PairScore:
        """Score candidate A vs candidate B for one prompt."""


@dataclass(frozen=True)
class _CliInvocationResult:
    """Minimal subprocess result surface for CLI judge execution."""

    returncode: int
    stdout: str
    stderr: str


CliJudgeRunner = Callable[[list[str], str, float], _CliInvocationResult]
SwayBackendFactory = Callable[[Path], Any]


@dataclass(frozen=True)
class CliJudge:
    """External command-backed preference judge.

    The command is parsed with `shlex.split()` and invoked once per
    candidate with a JSON payload on stdin:

        {"prompt": "...", "candidate": "..."}

    The command must emit a JSON object on stdout:

        {"score": 0.73, "reasoning": "optional note"}
    """

    command: str
    timeout: float = _DEFAULT_CLI_TIMEOUT_SECONDS
    runner: CliJudgeRunner | None = field(default=None, repr=False, compare=False)
    name: str = field(init=False)
    suggested_threshold: float = field(default=_DEFAULT_CLI_THRESHOLD, init=False)
    _argv: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        spec = self.command.strip()
        if not spec:
            raise InvalidJudgeSpecError("cli judge selector must include a command")
        argv = tuple(shlex.split(spec))
        if not argv:
            raise InvalidJudgeSpecError("cli judge selector must include a command")
        if self.timeout <= 0.0:
            raise ValueError(f"cli judge timeout must be > 0, got {self.timeout}")
        object.__setattr__(self, "name", f"cli:{spec}")
        object.__setattr__(self, "_argv", argv)

    def score_pair(self, prompt: str, candidate_a: str, candidate_b: str) -> PairScore:
        score_a = self._score_candidate(prompt, candidate_a)
        score_b = self._score_candidate(prompt, candidate_b)
        return PairScore(
            score_a=score_a.score,
            score_b=score_b.score,
            reasoning=_combine_reasoning(score_a.reasoning, score_b.reasoning),
        )

    def _score_candidate(self, prompt: str, candidate: str) -> _CandidateScore:
        payload = json.dumps(
            {"prompt": prompt, "candidate": candidate},
            ensure_ascii=False,
        )
        runner = self.runner if self.runner is not None else _default_cli_runner
        try:
            result = runner(list(self._argv), payload, self.timeout)
        except FileNotFoundError as exc:
            raise JudgeUnavailableError(
                f"cli judge binary {self._argv[0]!r} is not available on PATH"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise JudgeInvocationError(
                f"cli judge {self.command!r} timed out after {self.timeout}s"
            ) from exc
        except OSError as exc:
            raise JudgeUnavailableError(
                f"cli judge {self.command!r} could not start: {exc}"
            ) from exc

        if result.returncode != 0:
            tail = (result.stderr or result.stdout).strip() or "(no output)"
            raise JudgeInvocationError(
                f"cli judge {self.command!r} exited {result.returncode}: {tail}"
            )
        return _parse_cli_candidate_score(result.stdout)


@dataclass(frozen=True)
class SwayJudge:
    """sway-backed preference judge over the current document adapter.

    The default bootstrap path uses the fine-tuned adapter itself as a
    scorer. Scores are normalized by an approximate completion-token
    count so the judge does not trivially prefer the shortest sample.
    """

    dlm_path: Path
    backend_factory: SwayBackendFactory | None = field(default=None, repr=False, compare=False)
    name: str = field(default="sway:preference_judge", init=False)
    suggested_threshold: float = field(default=_DEFAULT_SWAY_THRESHOLD, init=False)
    _backend: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dlm_path", Path(self.dlm_path).expanduser())

    def score_pair(self, prompt: str, candidate_a: str, candidate_b: str) -> PairScore:
        backend = self._ensure_backend()
        try:
            with backend.as_finetuned() as ft_view:
                score_a = _normalized_sway_score(ft_view, prompt, candidate_a)
                score_b = _normalized_sway_score(ft_view, prompt, candidate_b)
        except Exception as exc:
            raise JudgeInvocationError(f"sway judge failed to score candidates: {exc}") from exc
        return PairScore(
            score_a=score_a,
            score_b=score_b,
            reasoning=f"ft mean-logprob delta={score_a - score_b:+.3f}",
        )

    def _ensure_backend(self) -> Any:
        cached = self._backend
        if cached is not None:
            return cached
        factory = self.backend_factory if self.backend_factory is not None else _build_sway_backend
        backend = factory(self.dlm_path)
        object.__setattr__(self, "_backend", backend)
        return backend


def parse_judge_ref(raw: str) -> JudgeRef:
    """Parse `sway`, `hf:<model>`, or `cli:<cmd>` judge selectors."""
    spec = raw.strip()
    if not spec:
        raise InvalidJudgeSpecError("judge selector must not be empty")
    if spec == "sway":
        return JudgeRef(raw=spec, kind="sway", target=None)
    if spec.startswith("hf:"):
        target = spec.removeprefix("hf:").strip()
        if not target:
            raise InvalidJudgeSpecError("hf judge selector must include a model id")
        return JudgeRef(raw=spec, kind="hf", target=target)
    if spec.startswith("cli:"):
        target = spec.removeprefix("cli:").strip()
        if not target:
            raise InvalidJudgeSpecError("cli judge selector must include a command")
        return JudgeRef(raw=spec, kind="cli", target=target)
    raise InvalidJudgeSpecError(
        f"unknown judge selector {raw!r}; expected 'sway', 'hf:<model>', or 'cli:<cmd>'"
    )


def build_judge(raw: str | JudgeRef, *, dlm_path: Path | None = None) -> PreferenceJudge:
    """Instantiate the concrete judge for `raw`.

    `sway` needs the source `.dlm` path so it can resolve the current
    adapter through sway's dlm bridge.
    """
    ref = parse_judge_ref(raw) if isinstance(raw, str) else raw
    if ref.kind == "cli":
        assert ref.target is not None
        return CliJudge(ref.target)
    if ref.kind == "hf":
        assert ref.target is not None
        return HfRewardModelJudge(ref.target)
    if dlm_path is None:
        raise JudgeUnavailableError("sway preference judge requires the .dlm path context")
    return SwayJudge(dlm_path)


@dataclass(frozen=True)
class _CandidateScore:
    """One CLI judge response normalized to score + optional reasoning."""

    score: float
    reasoning: str | None = None


def _default_cli_runner(argv: list[str], payload: str, timeout: float) -> _CliInvocationResult:
    proc = subprocess.run(  # nosec B603
        argv,
        input=payload,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return _CliInvocationResult(
        returncode=int(proc.returncode),
        stdout=str(proc.stdout),
        stderr=str(proc.stderr),
    )


def _parse_cli_candidate_score(stdout: str) -> _CandidateScore:
    blob = stdout.strip()
    if not blob:
        raise JudgeInvocationError("cli judge returned empty stdout")
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise JudgeInvocationError(f"cli judge returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise JudgeInvocationError("cli judge must return a JSON object")

    raw_score = payload.get("score")
    if isinstance(raw_score, bool) or not isinstance(raw_score, int | float):
        raise JudgeInvocationError("cli judge JSON must include numeric `score`")
    score = float(raw_score)
    if not math.isfinite(score):
        raise JudgeInvocationError("cli judge `score` must be finite")

    raw_reasoning = payload.get("reasoning")
    if raw_reasoning is not None and not isinstance(raw_reasoning, str):
        raise JudgeInvocationError("cli judge `reasoning` must be a string when present")

    return _CandidateScore(score=score, reasoning=raw_reasoning)


def _combine_reasoning(left: str | None, right: str | None) -> str | None:
    parts: list[str] = []
    if left:
        parts.append(f"a: {left}")
    if right:
        parts.append(f"b: {right}")
    return " | ".join(parts) if parts else None


def _normalized_sway_score(view: Any, prompt: str, completion: str) -> float:
    raw = float(view.logprob_of(prompt, completion))
    return raw / max(_token_estimate(completion), 1)


def _token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


RewardModelLoader = Callable[[str, str], "_LoadedRewardJudge"]


@dataclass(frozen=True)
class _LoadedRewardJudge:
    """Loaded HF reward-model bundle cached inside the judge."""

    model: Any
    tokenizer: Any
    device: str


class HfRewardModelJudge:
    """HF reward-model backed preference judge."""

    suggested_threshold = 1.0

    def __init__(
        self,
        hf_id: str,
        *,
        device: str = "auto",
        loader: RewardModelLoader | None = None,
    ) -> None:
        spec = hf_id.strip()
        if not spec:
            raise InvalidJudgeSpecError("hf judge selector must include a model id")
        self.hf_id = spec
        self.name = f"hf:{spec}"
        self._requested_device = device
        self._loader = loader
        self._loaded: _LoadedRewardJudge | None = None

    def score_pair(self, prompt: str, candidate_a: str, candidate_b: str) -> PairScore:
        score_a = self._score_candidate(prompt, candidate_a)
        score_b = self._score_candidate(prompt, candidate_b)
        return PairScore(score_a=score_a, score_b=score_b)

    def _score_candidate(self, prompt: str, candidate: str) -> float:
        loaded = self._ensure_loaded()
        try:
            import torch
        except ImportError as exc:
            raise JudgeUnavailableError("hf reward-model judge requires torch") from exc

        encoded = _encode_reward_input(loaded.tokenizer, prompt, candidate)
        moved = _move_to_device(encoded, loaded.device)
        with torch.inference_mode():
            output = loaded.model(**moved)
        logits = getattr(output, "logits", None)
        if logits is None:
            raise JudgeInvocationError("hf reward model returned no `.logits`")
        return _extract_reward_scalar(logits)

    def _ensure_loaded(self) -> _LoadedRewardJudge:
        if self._loaded is not None:
            return self._loaded
        device = _resolve_reward_device(self._requested_device)
        if self._loader is not None:
            loaded = self._loader(self.hf_id, device)
            self._loaded = loaded
            return loaded
        loaded = _default_reward_loader(self.hf_id, device)
        self._loaded = loaded
        return loaded


def _default_reward_loader(hf_id: str, device: str) -> _LoadedRewardJudge:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise JudgeUnavailableError(
            "hf reward-model judge requires transformers to be installed"
        ) from exc

    model = AutoModelForSequenceClassification.from_pretrained(hf_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return _LoadedRewardJudge(model=model, tokenizer=tokenizer, device=device)


def _resolve_reward_device(requested: str) -> str:
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


def _encode_reward_input(tokenizer: Any, prompt: str, candidate: str) -> Any:
    if getattr(tokenizer, "chat_template", None):
        try:
            rendered = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": candidate},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            if isinstance(rendered, str):
                return tokenizer(rendered, return_tensors="pt", truncation=True)
        except Exception:
            pass
    return tokenizer(prompt, text_pair=candidate, return_tensors="pt", truncation=True)


def _move_to_device(encoded: Any, device: str) -> Any:
    if hasattr(encoded, "to"):
        return encoded.to(device)
    if isinstance(encoded, dict):
        moved: dict[str, Any] = {}
        for key, value in encoded.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved
    return encoded


def _extract_reward_scalar(logits: Any) -> float:
    if hasattr(logits, "numel") and callable(logits.numel):
        numel = int(logits.numel())
        if numel != 1:
            raise JudgeInvocationError(
                f"hf reward model must return a single scalar logit, got {numel} values"
            )
        if hasattr(logits, "reshape"):
            flat = logits.reshape(-1)
            if hasattr(flat, "__getitem__"):
                value = flat[0]
                if hasattr(value, "item"):
                    scalar = float(value.item())
                    if math.isfinite(scalar):
                        return scalar
        if hasattr(logits, "item"):
            scalar = float(logits.item())
            if math.isfinite(scalar):
                return scalar
    raise JudgeInvocationError("hf reward model returned an unreadable scalar logit")


def _build_sway_backend(dlm_path: Path) -> Any:
    try:
        resolve_dlm, build_backend, model_spec_cls, sway_error_cls = _import_sway_bridge()
    except ImportError as exc:
        raise JudgeUnavailableError(
            "sway preference judge requires the sway bridge to be importable"
        ) from exc

    try:
        handle = resolve_dlm(dlm_path)
    except sway_error_cls as exc:
        raise JudgeUnavailableError(f"sway judge could not resolve {dlm_path}: {exc}") from exc
    except Exception as exc:
        raise JudgeUnavailableError(f"sway judge could not resolve {dlm_path}: {exc}") from exc

    adapter_path = handle.adapter_path
    if adapter_path is None:
        raise JudgeUnavailableError(
            f"sway preference judge requires a trained adapter for {dlm_path}"
        )

    try:
        base_spec = model_spec_cls(
            kind="hf",
            base=handle.base_model,
            adapter=adapter_path,
            trust_remote_code=_resolve_sway_trust_remote_code(dlm_path),
        )
        return build_backend(base_spec, adapter_path=adapter_path)
    except Exception as exc:
        raise JudgeUnavailableError(
            f"sway judge could not load backend for {dlm_path}: {exc}"
        ) from exc


def _import_sway_bridge() -> tuple[Any, Any, Any, Any]:
    def _load() -> tuple[Any, Any, Any, Any]:
        import importlib

        build_backend = importlib.import_module("dlm_sway.backends").build
        sway_error = importlib.import_module("dlm_sway.core.errors").SwayError
        model_spec = importlib.import_module("dlm_sway.core.model").ModelSpec
        resolve_dlm = importlib.import_module("dlm_sway.integrations.dlm.resolver").resolve_dlm
        return resolve_dlm, build_backend, model_spec, sway_error

    try:
        return _load()
    except ImportError:
        sway_src = Path(__file__).resolve().parents[3] / "sway" / "src"
        sway_src_str = str(sway_src)
        if sway_src.exists() and sway_src_str not in sys.path:
            sys.path.insert(0, sway_src_str)
        return _load()


def _resolve_sway_trust_remote_code(dlm_path: Path) -> bool:
    try:
        from dlm.base_models import resolve as resolve_base_model
        from dlm.doc.parser import parse_file
    except ImportError:
        return False

    try:
        parsed = parse_file(dlm_path)
    except Exception:
        return False

    base_model = parsed.frontmatter.base_model.strip()
    if not base_model or base_model.startswith("hf:"):
        return False

    try:
        spec = resolve_base_model(base_model, accept_license=True)
    except Exception:
        return False
    return bool(spec.trust_remote_code)
