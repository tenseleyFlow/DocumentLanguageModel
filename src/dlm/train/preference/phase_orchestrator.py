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
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from dlm.doc.sections import Section, SectionType
from dlm.train.preference.auto_enable import resolve_preference_enabled
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
    from dlm.train.trainer import Mode, TrainingRunResult

Phase = Literal["sft", "preference", "all"]

SftRunner = Callable[..., "TrainingRunResult"]
PreferenceRunner = Callable[..., "TrainingRunResult"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhaseResult:
    """Per-phase training outcome — the orchestrator returns one per
    phase that actually ran. `phase` is the high-level name ("sft" or
    "preference"); the specific preference method ("dpo" / "orpo") is
    recorded in the training-run summary, not here."""

    phase: Literal["sft", "preference"]
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


def _filter_preference_sections(
    sections: list[Section],
    *,
    include_auto_mined: bool,
) -> list[Section]:
    if include_auto_mined:
        return sections
    return [
        section
        for section in sections
        if section.type is not SectionType.PREFERENCE or not section.auto_mined
    ]


def run_phases(
    store: StorePath,
    parsed: ParsedDlm,
    spec: BaseModelSpec,
    plan: TrainingPlan,
    *,
    phase: Phase = "all",
    mode: Mode = "fresh",
    seed: int | None = None,
    max_steps: int | None = None,
    lock_mode: LockMode = "default",
    capabilities: Capabilities | None = None,
    world_size: int | None = None,
    strict_metrics: bool = False,
    include_auto_mined: bool = True,
    sft_runner: SftRunner | None = None,
    dpo_runner: PreferenceRunner | None = None,
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
    sections = _filter_preference_sections(
        list(parsed.sections),
        include_auto_mined=include_auto_mined,
    )
    pref_cfg = resolve_preference_enabled(parsed.frontmatter.training.preference, sections)
    results: list[PhaseResult] = []

    sft_fn = sft_runner or _real_sft_runner()
    pref_fn = dpo_runner or _method_runner(pref_cfg.method)

    if phase in ("sft", "all") and has_sft_content(sections):
        sft_kwargs: dict[str, object] = {
            "mode": mode,
            "seed": seed,
            "max_steps": max_steps,
            "lock_mode": lock_mode,
            "capabilities": capabilities,
            "strict_metrics": strict_metrics,
        }
        if world_size is not None:
            sft_kwargs["world_size"] = world_size
        sft_result = sft_fn(store, parsed, spec, plan, **sft_kwargs)
        results.append(PhaseResult(phase="sft", result=sft_result))

    should_run_pref = phase == "preference" or (phase == "all" and pref_cfg.enabled)
    if should_run_pref:
        if not has_preference_content(sections):
            if phase == "preference":
                raise NoPreferenceContentError(
                    "--phase preference was requested but the document has no "
                    "::preference:: sections"
                )
            log.warning(
                "preference.enabled=True but the document has no "
                "::preference:: sections; skipping preference phase"
            )
            return results

        # Determine which adapter version to load as the pre-preference
        # checkpoint. When SFT just ran this invocation, it's the
        # result we captured; otherwise, read from the manifest.
        sft_adapter_version = _resolve_reference_adapter_version(
            store, prior_sft_result=_latest_sft(results)
        )

        pref_result = pref_fn(
            store,
            parsed,
            spec,
            plan,
            reference_adapter_version=sft_adapter_version,
            seed=seed,
            max_steps=max_steps,
            lock_mode=lock_mode,
            capabilities=capabilities,
            include_auto_mined=include_auto_mined,
        )
        results.append(PhaseResult(phase="preference", result=pref_result))

    return results


def _method_runner(method: str) -> PreferenceRunner:
    """Resolve the preference phase runner by `method` name via the
    registry. Raises `UnknownMethodError` if unregistered."""
    from dlm.train.preference.method_registry import resolve as resolve_method

    return resolve_method(method)


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
