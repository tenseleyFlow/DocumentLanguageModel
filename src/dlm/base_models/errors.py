"""Typed errors raised by `dlm.base_models`."""

from __future__ import annotations

from dataclasses import dataclass, field


class BaseModelError(Exception):
    """Base class for every `dlm.base_models` error."""


class UnknownBaseModelError(BaseModelError):
    """Spec didn't resolve — not in the registry and not a valid `hf:` escape.

    Carries a short list of the registry keys we do know so the caller
    can render a helpful diagnostic.
    """

    def __init__(self, spec: str, known_keys: tuple[str, ...]) -> None:
        self.spec = spec
        self.known_keys = known_keys
        preview = ", ".join(known_keys[:5])
        tail = ""
        if len(known_keys) > 5:
            tail = f", … ({len(known_keys) - 5} more)"
        super().__init__(
            f"unknown base model {spec!r}. Known keys: {preview}{tail}. "
            "Use `hf:org/name` for models outside the registry."
        )


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of a single compatibility probe.

    `skipped=True` signals the probe couldn't run (e.g., vendored
    llama.cpp not yet installed) but we're choosing not to block — the
    aggregate verdict treats skipped as pass so offline dev stays
    unblocked. `detail` carries the human-readable reason.
    """

    name: str
    passed: bool
    detail: str
    skipped: bool = False


class ProbeFailedError(BaseModelError):
    """One or more compatibility probes failed for a resolved spec.

    The error carries every probe's result (pass and fail) so the CLI
    can render a complete picture, not just the first failure.
    """

    def __init__(self, hf_id: str, results: list[ProbeResult]) -> None:
        self.hf_id = hf_id
        self.results = tuple(results)
        failed = [r for r in results if not r.passed]
        failed_summary = "; ".join(f"{r.name}: {r.detail}" for r in failed)
        super().__init__(
            f"{hf_id}: {len(failed)} of {len(results)} probes failed: {failed_summary}"
        )


class GatedModelError(BaseModelError):
    """Model requires HuggingFace license acceptance and the user hasn't accepted.

    Lives here because registry probes catch it first; the acceptance
    record is written elsewhere, but the error shape is owned here.
    """

    def __init__(self, hf_id: str, license_url: str | None) -> None:
        self.hf_id = hf_id
        self.license_url = license_url
        where = f" License: {license_url}" if license_url else ""
        super().__init__(
            f"{hf_id} is a gated HuggingFace model. Accept the license and "
            f"pass --i-accept-license (or via `dlm init`).{where}"
        )


@dataclass(frozen=True)
class ProbeReport:
    """Aggregate of probe results; useful for `dlm doctor <base>` reporting."""

    hf_id: str
    results: tuple[ProbeResult, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """All non-skipped probes passed. Skipped probes don't block."""
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> tuple[ProbeResult, ...]:
        return tuple(r for r in self.results if not r.passed)

    @property
    def skipped(self) -> tuple[ProbeResult, ...]:
        return tuple(r for r in self.results if r.skipped)
