"""Modality dispatch base class — predicate flags + method hooks.

Callers that used to branch on ``spec.modality == "vision-language"``
or ``"audio-language"`` now read from a registered
:class:`ModalityDispatch` instance. Three concrete subclasses live
under the ``dlm.modality`` package — one per supported modality —
registered in :data:`MODALITIES` and resolved via
:func:`modality_for`. The split keeps the "does this spec accept
images?" predicate next to the "route the export through the VL
path" method: both are modality-specific concerns. Text-family tags
(`"text"` and `"text-moe"`) intentionally share the same dispatch
behavior.

Each instance carries:

- ``modality`` (string tag — the only place a `"vision-language"`
  string literal appears outside the base-model schema);
- predicate flags (``requires_processor``, ``accepts_images``,
  ``accepts_audio``) callers read instead of comparing the tag;
- dispatch hooks (``dispatch_export``, ``dispatch_prompt``) that
  forward to the modality-specific pipeline.

A pregate grep-gate refuses new ``spec.modality ==`` comparisons
outside this package so next-modality work lands here rather than
scattering another set of branches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dlm.modality.errors import UnknownModalityError

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.export.dispatch import DispatchResult


class ModalityDispatch:
    """Base class — subclasses override per-modality predicates + hooks.

    The base implementation defaults to the text-path semantics
    (nothing to probe, nothing to dispatch). Subclasses narrow the
    predicates and override the dispatch hooks.
    """

    modality: str = "text"
    """The modality tag. The only place modality string literals
    should appear outside this package."""

    requires_processor: bool = False
    """True for media modalities that ship a feature extractor /
    processor alongside the tokenizer. Text-only bases set this
    False — the trainer skips the BlobStore + preprocess pass."""

    accepts_images: bool = False
    """True for vision-language bases. Drives the ``dlm prompt
    --image`` guardrail."""

    accepts_audio: bool = False
    """True for audio-language bases. Drives the ``dlm prompt
    --audio`` guardrail."""

    def load_processor(self, spec: BaseModelSpec) -> Any | None:
        """Load the HF processor if this modality needs one. Text → None."""
        return None

    def dispatch_export(
        self,
        *,
        store: Any,
        spec: BaseModelSpec,
        adapter_name: str | None,
        quant: str | None,
        merged: bool,
        adapter_mix_raw: str | None,
        gguf_emission_context: dict[str, Any] | None = None,
    ) -> DispatchResult | None:
        """Route an export through the modality-specific path.

        Returns ``None`` on the text path — the caller falls back to
        the GGUF `run_export` pipeline, which has a different result
        shape (`run_export` returns `RunResult`, not `DispatchResult`,
        and the text path prints its own banner inline).
        """
        return None


class TextModality(ModalityDispatch):
    """Text-family base — defaults carry the whole contract."""

    modality = "text"


def _unknown(mod: str) -> UnknownModalityError:
    return UnknownModalityError(
        f"modality={mod!r} has no registered dispatcher. "
        "Register a ModalityDispatch subclass in dlm.modality and "
        "add it to MODALITIES."
    )


def modality_for(spec: BaseModelSpec) -> ModalityDispatch:
    """Resolve a spec's ``ModalityDispatch``, raising if unregistered."""
    from dlm.modality import MODALITIES  # late import to avoid cycle

    try:
        return MODALITIES[spec.modality]
    except KeyError as exc:
        raise _unknown(spec.modality) from exc
