"""Persist staged auto-synth instruction sections between CLI steps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlm.doc.sections import Section, SectionType
from dlm.io.atomic import write_text as atomic_write_text
from dlm.synth.errors import SynthError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dlm.store.paths import StorePath


class PendingSynthPlanError(SynthError):
    """Raised when the staged synth plan cannot be read or validated."""


@dataclass(frozen=True)
class PendingSynthPlan:
    """One staged synth plan for a store."""

    source_path: Path
    created_at: str
    sections: tuple[Section, ...]


def pending_plan_path(store: StorePath) -> Path:
    """Path to the staged synth payload for `store`."""
    return store.root / "synth" / "pending.json"


def save_pending_plan(
    store: StorePath,
    *,
    source_path: Path,
    sections: Sequence[Section],
) -> PendingSynthPlan:
    """Persist `sections` as the staged synth plan for `store`."""
    plan = PendingSynthPlan(
        source_path=source_path.resolve(),
        created_at=_utcnow(),
        sections=tuple(sections),
    )
    path = pending_plan_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_path": str(plan.source_path),
        "created_at": plan.created_at,
        "sections": [_section_to_payload(section) for section in plan.sections],
    }
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return plan


def load_pending_plan(store: StorePath) -> PendingSynthPlan | None:
    """Return the staged synth plan for `store`, or None when absent."""
    path = pending_plan_path(store)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PendingSynthPlanError(f"could not read staged synth plan: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PendingSynthPlanError(f"staged synth plan is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise PendingSynthPlanError("staged synth plan must be a JSON object")
    if raw.get("schema_version") != 1:
        raise PendingSynthPlanError(
            f"unsupported staged synth plan schema_version={raw.get('schema_version')!r}"
        )

    source_path = raw.get("source_path")
    created_at = raw.get("created_at")
    sections_raw = raw.get("sections")
    if not isinstance(source_path, str) or not source_path:
        raise PendingSynthPlanError("staged synth plan is missing source_path")
    if not isinstance(created_at, str) or not created_at:
        raise PendingSynthPlanError("staged synth plan is missing created_at")
    if not isinstance(sections_raw, list):
        raise PendingSynthPlanError("staged synth plan is missing sections")

    sections: list[Section] = []
    for idx, entry in enumerate(sections_raw):
        try:
            sections.append(_section_from_payload(entry))
        except (TypeError, ValueError, KeyError) as exc:
            raise PendingSynthPlanError(f"invalid section payload at index {idx}: {exc}") from exc

    return PendingSynthPlan(
        source_path=Path(source_path),
        created_at=created_at,
        sections=tuple(sections),
    )


def clear_pending_plan(store: StorePath) -> bool:
    """Delete the staged synth plan for `store`. Returns True iff it existed."""
    path = pending_plan_path(store)
    if not path.exists():
        return False
    path.unlink()
    return True


def _utcnow() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _section_to_payload(section: Section) -> dict[str, Any]:
    return {
        "type": section.type.value,
        "content": section.content,
        "start_line": section.start_line,
        "adapter": section.adapter,
        "tags": dict(section.tags),
        "auto_harvest": section.auto_harvest,
        "harvest_source": section.harvest_source,
        "auto_mined": section.auto_mined,
        "judge_name": section.judge_name,
        "judge_score_chosen": section.judge_score_chosen,
        "judge_score_rejected": section.judge_score_rejected,
        "mined_at": section.mined_at,
        "mined_run_id": section.mined_run_id,
        "auto_synth": section.auto_synth,
        "synth_teacher": section.synth_teacher,
        "synth_strategy": section.synth_strategy,
        "synth_at": section.synth_at,
        "source_section_id": section.source_section_id,
        "media_path": section.media_path,
        "media_alt": section.media_alt,
        "media_blob_sha": section.media_blob_sha,
        "media_transcript": section.media_transcript,
    }


def _section_from_payload(raw: object) -> Section:
    if not isinstance(raw, dict):
        raise TypeError(f"expected object, got {type(raw).__name__}")
    section_type = SectionType(str(raw["type"]))
    tags = raw.get("tags", {})
    if not isinstance(tags, dict):
        raise TypeError("tags must be an object")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in tags.items()):
        raise TypeError("tags keys and values must be strings")
    return Section(
        type=section_type,
        content=str(raw["content"]),
        start_line=int(raw.get("start_line", 0)),
        adapter=_optional_str(raw.get("adapter")),
        tags=dict(tags),
        auto_harvest=bool(raw.get("auto_harvest", False)),
        harvest_source=_optional_str(raw.get("harvest_source")),
        auto_mined=bool(raw.get("auto_mined", False)),
        judge_name=_optional_str(raw.get("judge_name")),
        judge_score_chosen=_optional_float(raw.get("judge_score_chosen")),
        judge_score_rejected=_optional_float(raw.get("judge_score_rejected")),
        mined_at=_optional_str(raw.get("mined_at")),
        mined_run_id=_optional_int(raw.get("mined_run_id")),
        auto_synth=bool(raw.get("auto_synth", False)),
        synth_teacher=_optional_str(raw.get("synth_teacher")),
        synth_strategy=_optional_str(raw.get("synth_strategy")),
        synth_at=_optional_str(raw.get("synth_at")),
        source_section_id=_optional_str(raw.get("source_section_id")),
        media_path=_optional_str(raw.get("media_path")),
        media_alt=_optional_str(raw.get("media_alt")),
        media_blob_sha=_optional_str(raw.get("media_blob_sha")),
        media_transcript=_optional_str(raw.get("media_transcript")),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"expected string or null, got {type(value).__name__}")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected float or null, got {type(value).__name__}")
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected int or null, got {type(value).__name__}")
    return value
