"""Typed judge protocol and selector parsing for Sprint 42."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from dlm.preference.errors import InvalidJudgeSpecError

JudgeKind = Literal["sway", "hf", "cli"]


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

    name: str
    suggested_threshold: float

    def score_pair(self, prompt: str, candidate_a: str, candidate_b: str) -> PairScore:
        """Score candidate A vs candidate B for one prompt."""


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
