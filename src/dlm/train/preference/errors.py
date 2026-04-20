"""Typed errors for the DPO (preference) phase.

Mirrors the pattern in `dlm.train.errors`: every failure mode the CLI
reworders into a user-actionable message has its own class so we don't
stringly-match upstream TRL / HF exceptions.
"""

from __future__ import annotations


class DpoPhaseError(Exception):
    """Base for `dlm.train.preference` errors."""


class NoPreferenceContentError(DpoPhaseError):
    """`--phase dpo` or `dpo.enabled=True` but the document has zero
    `::preference::` triples. Spec calls for a warn-and-skip when DPO
    is *implicit*; this error fires when DPO was *explicitly* requested
    and the content isn't there."""


class PriorAdapterRequiredError(DpoPhaseError):
    """DPO phase was invoked (via `--phase dpo`) without a prior SFT
    adapter version on disk.

    The reference-adapter mode needs an adapter to freeze as the
    reference; the base-model reference mode needs one to continue
    training from. Either way, DPO-only without prior SFT is not
    supported."""


class DpoReferenceLoadError(DpoPhaseError):
    """Failed to materialize the frozen reference model.

    Wraps the specific PEFT/transformers load failure with the
    adapter-version path that couldn't be opened."""

    def __init__(self, *, adapter_path: str, cause: str) -> None:
        super().__init__(
            f"could not load DPO reference model from {adapter_path}: {cause}"
        )
        self.adapter_path = adapter_path
        self.cause = cause
