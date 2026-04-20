"""Dispatch `training.preference.method` → phase runtime function.

The orchestrator consults this registry to decide which preference
runtime to call. Two methods ship today: `"dpo"` (via `dpo_phase.run`)
and `"orpo"` (via `orpo_phase.run`). A method name that isn't
registered raises `UnknownMethodError` so silent typos in frontmatter
fail loudly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final

from dlm.train.preference.errors import DpoPhaseError

if TYPE_CHECKING:
    from dlm.train.trainer import TrainingRunResult


class UnknownMethodError(DpoPhaseError):
    """`training.preference.method` is set to a name the registry
    doesn't know. Distinct from TRL API errors so the CLI reporter
    can render a clean "did you mean dpo or orpo?" surface."""

    def __init__(self, name: str) -> None:
        known = ", ".join(sorted(METHODS))
        super().__init__(f"unknown preference method {name!r}; known: {known}")
        self.name = name


MethodRunner = Callable[..., "TrainingRunResult"]

_REGISTRY: dict[str, MethodRunner] = {}


def register(name: str, runner: MethodRunner) -> None:
    """Register (or replace) a phase runner by method name."""
    _REGISTRY[name] = runner


def resolve(name: str) -> MethodRunner:
    """Look up the runner for `name` — raises `UnknownMethodError` if
    unregistered."""
    if name not in _REGISTRY:
        raise UnknownMethodError(name)
    return _REGISTRY[name]


# Side-effect registrations run below so `resolve` is usable at import
# time without the orchestrator needing to prime the registry itself.
# Lazy-import the phase modules to keep the registry cheap to import
# from test code that mocks runners anyway.


def _lazy_dpo_runner() -> MethodRunner:  # pragma: no cover
    from dlm.train.preference.dpo_phase import run as dpo_run

    return dpo_run


def _lazy_orpo_runner() -> MethodRunner:  # pragma: no cover
    from dlm.train.preference.orpo_phase import run as orpo_run

    return orpo_run


def _dpo_proxy(*args: object, **kwargs: object) -> TrainingRunResult:  # pragma: no cover
    return _lazy_dpo_runner()(*args, **kwargs)


def _orpo_proxy(*args: object, **kwargs: object) -> TrainingRunResult:  # pragma: no cover
    return _lazy_orpo_runner()(*args, **kwargs)


METHODS: Final[set[str]] = {"dpo", "orpo"}

register("dpo", _dpo_proxy)
register("orpo", _orpo_proxy)
