"""Typed judge protocol, selector parsing, and concrete judge runtimes."""

from __future__ import annotations

import json
import math
import shlex
import subprocess  # nosec B404
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from dlm.preference.errors import (
    InvalidJudgeSpecError,
    JudgeInvocationError,
    JudgeUnavailableError,
)

JudgeKind = Literal["sway", "hf", "cli"]

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


def build_judge(raw: str | JudgeRef) -> PreferenceJudge:
    """Instantiate the concrete judge for `raw`.

    This slice only lands the external CLI-backed path. HF reward-model
    and sway-backed judges follow in later Sprint 42 chunks.
    """
    ref = parse_judge_ref(raw) if isinstance(raw, str) else raw
    if ref.kind == "cli":
        assert ref.target is not None
        return CliJudge(ref.target)
    if ref.kind == "hf":
        raise JudgeUnavailableError(
            f"hf reward-model judge {ref.target!r} is not wired yet in this slice"
        )
    raise JudgeUnavailableError("sway preference judge is not wired yet in this slice")


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
