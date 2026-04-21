"""Parse a sway JSON report into harvest candidates.

Sway emits reports with this shape (see
``sway/src/dlm_sway/suite/report.py``):

.. code-block:: json

    {
      "schema_version": 1,
      "sway_version": "...",
      "base_model_id": "...",
      "adapter_id": "...",
      "probes": [
        {
          "name": "...",
          "kind": "...",
          "verdict": "pass" | "fail" | "warn" | "skip" | "error",
          "score": 0.0,
          "evidence": {...},
          "message": "...",
          ...
        }
      ]
    }

The harvest pull path filters for ``verdict == "fail"`` and lifts
out ``evidence.prompt`` + ``evidence.reference`` as the Q/A pair for
the next retrain. Probes without both fields are skipped with a
:class:`NoReferenceError` under strict mode (default) or a log line
under ``strict=False``.

``evidence.confidence`` (optional, 0-1) gates candidates via the
caller's ``--min-confidence``. Absent confidence is treated as 1.0
— the probe itself already failed, which is our signal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from dlm.harvest.errors import MalformedSwayReportError, NoReferenceError

_LOG = logging.getLogger(__name__)

# Sway's JSON schema version we know how to parse. A higher version
# in a report triggers a refusal with a clear pointer — sway's schema
# is stable but not fixed forever.
_SUPPORTED_SWAY_SCHEMA: Final[int] = 1


@dataclass(frozen=True)
class HarvestCandidate:
    """One failing probe ready to become a `!probe`-tagged section.

    Attributes
    ----------
    prompt:
        The question text. Becomes the `### Q` body.
    reference:
        The expected answer. Becomes the `### A` body.
    confidence:
        0-1 weight sway assigned to this probe's reference, when
        present. Defaults to 1.0 when the report doesn't carry it.
    probe_name:
        Human-readable probe name from the sway spec. Used for the
        harvest tag so users can trace a synthesized section back to
        its probe origin.
    probe_kind:
        Probe discriminator (``section_internalization`` etc.).
    source_adapter_version:
        The adapter revision sway was scoring when it failed, if
        `adapter_id` carries one. Informational; the harvest
        itself doesn't need it.
    """

    prompt: str
    reference: str
    confidence: float
    probe_name: str
    probe_kind: str
    source_adapter_version: str | None


def read_sway_report(
    path: Path | str,
    *,
    strict: bool = True,
    min_confidence: float = 0.0,
) -> list[HarvestCandidate]:
    """Parse a sway JSON report at `path` into harvest candidates.

    Parameters
    ----------
    path:
        Path to the sway JSON report.
    strict:
        If True (default), raise :class:`NoReferenceError` when a
        failing probe lacks a ``prompt`` / ``reference`` pair. If
        False, log a warning and skip the probe.
    min_confidence:
        Minimum ``evidence.confidence`` for a candidate to survive.
        Default 0.0 accepts all.

    Raises
    ------
    MalformedSwayReportError:
        File unreadable, not JSON, missing required keys, or carries
        a newer ``schema_version`` than this reader supports.
    NoReferenceError:
        Strict mode + at least one failing probe lacks a reference.
    """
    report_path = Path(path)
    try:
        raw = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MalformedSwayReportError(f"cannot read sway report at {report_path}: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MalformedSwayReportError(
            f"sway report at {report_path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise MalformedSwayReportError(
            f"sway report at {report_path} must be a JSON object; got {type(payload).__name__}"
        )

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int):
        raise MalformedSwayReportError(
            f"sway report at {report_path} missing integer `schema_version`"
        )
    if schema_version > _SUPPORTED_SWAY_SCHEMA:
        raise MalformedSwayReportError(
            f"sway report schema_version={schema_version} is newer than this "
            f"reader supports ({_SUPPORTED_SWAY_SCHEMA}); bump the sway pin "
            "in `dlm.lock` after verifying harvest still round-trips"
        )

    probes = payload.get("probes")
    if not isinstance(probes, list):
        raise MalformedSwayReportError(f"sway report at {report_path} missing `probes` array")

    adapter_id = payload.get("adapter_id")
    source_adapter_version: str | None = None
    if isinstance(adapter_id, str) and adapter_id:
        source_adapter_version = adapter_id

    candidates: list[HarvestCandidate] = []
    for idx, probe in enumerate(probes):
        if not isinstance(probe, dict):
            _LOG.warning(
                "sway report %s: probe index %d is not an object; skipping",
                report_path,
                idx,
            )
            continue
        if probe.get("verdict") != "fail":
            continue
        try:
            candidate = _probe_to_candidate(
                probe,
                source_adapter_version=source_adapter_version,
            )
        except NoReferenceError:
            if strict:
                raise
            _LOG.warning(
                "sway report %s: probe %r failed but carries no "
                "reference; skipping (use --strict to fail)",
                report_path,
                probe.get("name", "<unnamed>"),
            )
            continue
        if candidate.confidence < min_confidence:
            _LOG.info(
                "harvest: skipping %r (confidence=%.2f < %.2f)",
                candidate.probe_name,
                candidate.confidence,
                min_confidence,
            )
            continue
        candidates.append(candidate)

    return candidates


def _probe_to_candidate(
    probe: dict[str, Any],
    *,
    source_adapter_version: str | None,
) -> HarvestCandidate:
    """Lift one failing probe into a `HarvestCandidate`.

    Raises :class:`NoReferenceError` when the evidence doesn't
    carry both a prompt and a reference — that probe cannot be
    round-tripped into a supervised Q/A row.
    """
    name = str(probe.get("name") or "<unnamed>")
    kind = str(probe.get("kind") or "")
    evidence = probe.get("evidence") or {}
    if not isinstance(evidence, dict):
        raise NoReferenceError(f"probe {name!r}: evidence is not an object; cannot harvest")

    prompt_raw = evidence.get("prompt")
    reference_raw = evidence.get("reference")
    if not isinstance(prompt_raw, str) or not prompt_raw.strip():
        raise NoReferenceError(f"probe {name!r}: evidence.prompt missing or non-string")
    if not isinstance(reference_raw, str) or not reference_raw.strip():
        raise NoReferenceError(f"probe {name!r}: evidence.reference missing or non-string")

    confidence_raw = evidence.get("confidence", 1.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))

    return HarvestCandidate(
        prompt=prompt_raw.strip(),
        reference=reference_raw.strip(),
        confidence=confidence,
        probe_name=name,
        probe_kind=kind,
        source_adapter_version=source_adapter_version,
    )
