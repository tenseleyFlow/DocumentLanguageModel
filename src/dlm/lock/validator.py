"""Validate the current run against a prior `dlm.lock`.

The three `dlm train` flags map to `LockMode`:

- `default` — validate; abort on ERROR, warn on WARN, proceed otherwise.
- `strict` — `--strict-lock`; upgrade every WARN to ERROR.
- `update` — `--update-lock`; bypass validation, overwrite on success.
- `ignore` — `--ignore-lock`; bypass validation, don't write on success.

Returning a `LockDecision` rather than raising keeps the caller in
charge of how to render failures; the trainer converts `abort` into a
typed `LockValidationError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dlm.lock.policy import Severity, classify_mismatches
from dlm.lock.schema import DlmLock

LockMode = Literal["default", "strict", "update", "ignore"]


@dataclass(frozen=True)
class LockDecision:
    """Result of `validate_lock`.

    `action`:
      - `"proceed"` — no mismatches (or mode bypassed validation).
      - `"proceed_with_warnings"` — only WARN-level drift.
      - `"abort"` — at least one ERROR-level mismatch; caller must
        refuse to run (or re-invoke with `--update-lock` / `--ignore-lock`).

    `should_write_lock`:
      - True in `default` / `strict` / `update` modes.
      - False in `ignore` mode — the operator is deliberately
        experimenting without rewriting the recorded contract.

    `mismatches`: full list `[(severity, message), …]` suitable for
    rendering in the CLI reporter.
    """

    action: Literal["proceed", "proceed_with_warnings", "abort"]
    mismatches: list[tuple[Severity, str]]
    should_write_lock: bool


def validate_lock(
    prior: DlmLock | None,
    current: DlmLock,
    *,
    mode: LockMode = "default",
) -> LockDecision:
    """Compare `current` against `prior`, return a `LockDecision`."""
    if mode == "update":
        return LockDecision(action="proceed", mismatches=[], should_write_lock=True)
    if mode == "ignore":
        return LockDecision(action="proceed", mismatches=[], should_write_lock=False)
    if prior is None:
        # Fresh store: nothing to validate against; write the baseline.
        return LockDecision(action="proceed", mismatches=[], should_write_lock=True)

    strict = mode == "strict"
    mismatches = classify_mismatches(prior, current, strict=strict)
    has_error = any(sev is Severity.ERROR for sev, _ in mismatches)
    has_warn = any(sev is Severity.WARN for sev, _ in mismatches)

    if has_error:
        return LockDecision(action="abort", mismatches=mismatches, should_write_lock=False)
    if has_warn:
        return LockDecision(
            action="proceed_with_warnings",
            mismatches=mismatches,
            should_write_lock=True,
        )
    return LockDecision(action="proceed", mismatches=mismatches, should_write_lock=True)
