"""Mismatch severity table for `dlm.lock` validation (Sprint 15).

Decides for each field how loudly a drift between the recorded lock
and the current runtime should surface:

- `ALLOW` — drift is fine; `dlm_sha256` is the obvious one (editing
  the `.dlm` is the point).
- `WARN` — drift deserves a printed message but doesn't block the
  run. `--strict-lock` upgrades all WARNs to ERROR.
- `ERROR` — drift blocks the run unless the operator explicitly
  accepts via `--update-lock` (overwrite + continue) or
  `--ignore-lock` (continue without touching the lock).

Each rule is a pure function from the prior & current `DlmLock` to
`(severity, message) | None`. The validator drives them all; adding
a new field only requires adding a rule here.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Final

from packaging.version import InvalidVersion, Version

from dlm.lock.schema import DlmLock


class Severity(Enum):
    """Three-state mismatch outcome."""

    ALLOW = "allow"
    WARN = "warn"
    ERROR = "error"


# Rule signature: takes (prior, current), returns a mismatch description
# or `None` when the field hasn't drifted.
Rule = Callable[[DlmLock, DlmLock], tuple[Severity, str] | None]


# --- version parsing helper -------------------------------------------------


def _major(version_str: str) -> int | None:
    """Extract the major version; return None on unparseable input.

    Falling back to `None` is deliberate — non-semver packages (like
    llama.cpp's `b8816` tag) should treat any change as a minor drift,
    not a major-version error that blocks the run.
    """
    try:
        return Version(version_str).major
    except InvalidVersion:
        return None


# --- per-field rules --------------------------------------------------------


def _rule_dlm_sha(prior: DlmLock, current: DlmLock) -> tuple[Severity, str] | None:
    if prior.dlm_sha256 == current.dlm_sha256:
        return None
    # Editing the .dlm is the whole point of `dlm train` — never block.
    return (
        Severity.ALLOW,
        f"dlm_sha256 changed ({prior.dlm_sha256[:12]}… → {current.dlm_sha256[:12]}…)",
    )


def _rule_base_revision(prior: DlmLock, current: DlmLock) -> tuple[Severity, str] | None:
    if prior.base_model_revision == current.base_model_revision:
        return None
    return (
        Severity.ERROR,
        f"base_model_revision changed ({prior.base_model_revision} → "
        f"{current.base_model_revision}); re-run with --update-lock to accept",
    )


def _rule_hardware_tier(prior: DlmLock, current: DlmLock) -> tuple[Severity, str] | None:
    if prior.hardware_tier == current.hardware_tier:
        return None
    return (
        Severity.WARN,
        f"hardware_tier changed ({prior.hardware_tier} → {current.hardware_tier}); "
        "re-plan recommended",
    )


def _rule_determinism_class(
    prior: DlmLock, current: DlmLock
) -> tuple[Severity, str] | None:
    if prior.determinism_class == current.determinism_class:
        return None
    return (
        Severity.WARN,
        f"determinism_class changed ({prior.determinism_class} → "
        f"{current.determinism_class})",
    )


def _rule_determinism_flags(
    prior: DlmLock, current: DlmLock
) -> tuple[Severity, str] | None:
    if prior.determinism_flags == current.determinism_flags:
        return None
    return (Severity.WARN, "determinism_flags changed")


def _rule_torch_version(prior: DlmLock, current: DlmLock) -> tuple[Severity, str] | None:
    prior_v = prior.pinned_versions.get("torch")
    current_v = current.pinned_versions.get("torch")
    if prior_v is None or current_v is None or prior_v == current_v:
        return None
    prior_major = _major(prior_v)
    current_major = _major(current_v)
    if prior_major is not None and current_major is not None and prior_major != current_major:
        return (
            Severity.ERROR,
            f"torch major-version mismatch ({prior_v} → {current_v})",
        )
    return (Severity.WARN, f"torch minor-version drift ({prior_v} → {current_v})")


def _rule_bitsandbytes_any(
    prior: DlmLock, current: DlmLock
) -> tuple[Severity, str] | None:
    prior_v = prior.pinned_versions.get("bitsandbytes")
    current_v = current.pinned_versions.get("bitsandbytes")
    if prior_v == current_v:
        return None
    # Any bnb drift is a strong warning — QLoRA correctness is unusually
    # sensitive to bnb kernels.
    return (
        Severity.WARN,
        f"bitsandbytes changed ({prior_v!r} → {current_v!r}); QLoRA kernels are "
        "version-sensitive",
    )


def _rule_minor_peers(prior: DlmLock, current: DlmLock) -> list[tuple[Severity, str]]:
    """WARN on drift for transformers / peft / trl / accelerate / llama_cpp."""
    keys = ("transformers", "peft", "trl", "accelerate", "llama_cpp")
    mismatches: list[tuple[Severity, str]] = []
    for key in keys:
        prior_v = prior.pinned_versions.get(key)
        current_v = current.pinned_versions.get(key)
        if prior_v == current_v:
            continue
        mismatches.append(
            (Severity.WARN, f"{key} changed ({prior_v!r} → {current_v!r})"),
        )
    return mismatches


# --- driver -----------------------------------------------------------------


DEFAULT_RULES: Final[tuple[Rule, ...]] = (
    _rule_dlm_sha,
    _rule_base_revision,
    _rule_hardware_tier,
    _rule_determinism_class,
    _rule_determinism_flags,
    _rule_torch_version,
    _rule_bitsandbytes_any,
)


def classify_mismatches(
    prior: DlmLock,
    current: DlmLock,
    *,
    strict: bool = False,
) -> list[tuple[Severity, str]]:
    """Return every field-level mismatch between `prior` and `current`.

    `strict=True` upgrades `WARN` results to `ERROR`. `ALLOW` is never
    upgraded — allowed drift is allowed, even under strict mode
    (otherwise `--strict-lock` would block every retrain).
    """
    results: list[tuple[Severity, str]] = []
    for rule in DEFAULT_RULES:
        outcome = rule(prior, current)
        if outcome is not None:
            results.append(outcome)
    results.extend(_rule_minor_peers(prior, current))
    if strict:
        results = [
            (Severity.ERROR if sev is Severity.WARN else sev, msg)
            for sev, msg in results
        ]
    return results
