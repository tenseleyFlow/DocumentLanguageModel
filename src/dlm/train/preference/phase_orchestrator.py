"""Sequence SFT → DPO within a single `dlm train` invocation.

The existing SFT trainer (`dlm.train.trainer.run`) owns one phase end
to end. This module picks up from that contract and layers on the
optional DPO phase:

- `--phase sft` runs only SFT (existing behavior, no DPO even if
  preference content exists and `dpo.enabled=True`)
- `--phase dpo` runs only DPO, requiring a prior SFT adapter on disk
- `--phase all` (default) runs SFT first if SFT content exists, then
  DPO if `dpo.enabled=True` and preference content exists

Content detection is a pure function of `parsed.sections`; the
dispatcher is a pure function over the runners. The heavy ML path
lives in `dpo_phase.run`, which accepts the same test-seam shape as
`trainer.run` (a runner callable you can mock in unit tests).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

from dlm.doc.sections import Section, SectionType
from dlm.train.preference.errors import (
    NoPreferenceContentError,
    PriorAdapterRequiredError,
)

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.doc.parser import ParsedDlm
    from dlm.hardware.capabilities import Capabilities
    from dlm.hardware.plan import TrainingPlan
    from dlm.lock import LockMode
    from dlm.store.paths import StorePath
    from dlm.train.trainer import TrainingRunResult

Phase = Literal["sft", "dpo", "all"]

SftRunner = Callable[..., "TrainingRunResult"]
DpoRunner = Callable[..., "TrainingRunResult"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhaseResult:
    """Per-phase training outcome — the orchestrator returns one per
    phase that actually ran."""

    phase: Literal["sft", "dpo"]
    result: TrainingRunResult


def has_sft_content(sections: list[Section]) -> bool:
    """True iff at least one non-empty PROSE or INSTRUCTION section
    exists. Empty PROSE bodies are excluded to match `sections_to_rows`'
    silent-drop behavior."""
    for section in sections:
        if section.type is SectionType.PROSE and section.content.strip():
            return True
        if section.type is SectionType.INSTRUCTION:
            return True
    return False


def has_preference_content(sections: list[Section]) -> bool:
    """True iff at least one PREFERENCE section exists.

    The body-level parser guarantees any PREFERENCE section has ≥1
    triple, so section presence is sufficient — no need to inspect
    content."""
    return any(s.type is SectionType.PREFERENCE for s in sections)


def run_phases(
    store: StorePath,
    parsed: ParsedDlm,
    spec: BaseModelSpec,
    plan: TrainingPlan,
    *,
    phase: Phase = "all",
    seed: int | None = None,
    max_steps: int | None = None,
    lock_mode: LockMode = "default",
    capabilities: Capabilities | None = None,
    sft_runner: SftRunner | None = None,
    dpo_runner: DpoRunner | None = None,
) -> list[PhaseResult]:
    """Run the requested phase(s) in SFT → DPO order.

    `sft_runner` / `dpo_runner` are test seams — production callers
    leave them `None` and the orchestrator binds the real functions.

    Semantics by `phase`:

    - `"sft"`: run SFT only if SFT content exists; if not, return [].
      Never triggers DPO even when preference content is present.
    - `"dpo"`: run DPO only. Requires a prior adapter version on disk
      (via the manifest) — raises `PriorAdapterRequiredError` if
      there isn't one. Raises `NoPreferenceContentError` if the
      document has no `::preference::` sections (explicit request
      without content is a typo, not a warn).
    - `"all"`: run SFT if SFT content exists, then DPO if
      `dpo.enabled` and preference content exists. When DPO is
      auto-enabled (by presence of preference sections) but the
      document has no preference content and the user didn't
      explicitly request DPO, skip with a warning instead of raising.
    """
    sections = list(parsed.sections)
    dpo_cfg = parsed.frontmatter.training.dpo
    results: list[PhaseResult] = []

    sft_fn = sft_runner or _real_sft_runner()
    dpo_fn = dpo_runner or _real_dpo_runner()

    if phase in ("sft", "all") and has_sft_content(sections):
        sft_result = sft_fn(
            store,
            parsed,
            spec,
            plan,
            seed=seed,
            max_steps=max_steps,
            lock_mode=lock_mode,
            capabilities=capabilities,
        )
        results.append(PhaseResult(phase="sft", result=sft_result))

    should_run_dpo = phase == "dpo" or (phase == "all" and dpo_cfg.enabled)
    if should_run_dpo:
        if not has_preference_content(sections):
            if phase == "dpo":
                raise NoPreferenceContentError(
                    "--phase dpo was requested but the document has no "
                    "::preference:: sections"
                )
            log.warning(
                "dpo.enabled=True but the document has no ::preference:: "
                "sections; skipping DPO phase"
            )
            return results

        # Determine which adapter version to use as reference.
        # When SFT just ran this invocation, it's the result we just
        # captured. When `--phase dpo` alone, we read from disk.
        sft_adapter_version = _resolve_reference_adapter_version(
            store, prior_sft_result=_latest_sft(results)
        )

        dpo_result = dpo_fn(
            store,
            parsed,
            spec,
            plan,
            reference_adapter_version=sft_adapter_version,
            seed=seed,
            max_steps=max_steps,
            lock_mode=lock_mode,
            capabilities=capabilities,
        )
        results.append(PhaseResult(phase="dpo", result=dpo_result))

    return results


def _latest_sft(results: list[PhaseResult]) -> TrainingRunResult | None:
    for r in reversed(results):
        if r.phase == "sft":
            return r.result
    return None


def _resolve_reference_adapter_version(
    store: StorePath,
    *,
    prior_sft_result: TrainingRunResult | None,
) -> int:
    """Pick the adapter version DPO should load as reference.

    Priority: SFT result from this invocation, then the
    manifest-recorded `adapter_current` version. Raises
    `PriorAdapterRequiredError` if neither exists — DPO can't fabricate
    a reference from thin air."""
    if prior_sft_result is not None:
        return prior_sft_result.adapter_version

    from dlm.store.manifest import load_manifest

    manifest = load_manifest(store.manifest)
    if manifest.adapter_version == 0:
        raise PriorAdapterRequiredError(
            "--phase dpo requires a prior SFT adapter version on disk; "
            "run `dlm train --phase sft` first"
        )
    return manifest.adapter_version


def _real_sft_runner() -> SftRunner:  # pragma: no cover
    """Lazy binding so we don't import the heavy trainer module at
    orchestrator import time (tests that mock both runners never need
    to load HF)."""
    from dlm.train.trainer import run as sft_run

    return sft_run


def _real_dpo_runner() -> DpoRunner:  # pragma: no cover
    from dlm.train.preference.dpo_phase import run as dpo_run

    return dpo_run
