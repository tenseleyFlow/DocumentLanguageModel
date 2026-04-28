"""Shared substrate for staged "pending plan" payloads under a store.

Both `dlm preference mine`/`apply` and `dlm synth instructions`/`apply`
need to stage a list of `Section` payloads on disk between two CLI
invocations, then read them back in the apply step. Each domain stores
the payload under a different store subdirectory and wraps validation
errors in its own typed exception, but the I/O shape is identical.

This module owns the I/O. The two domain modules
(`dlm.preference.pending`, `dlm.synth.pending`) supply their own
`PendingPlan` dataclass and error class via the small set of
parameterized functions below.

The on-disk format records the full optional-field surface of
``Section`` so a domain that grows new optional fields tomorrow does
not need to bump ``schema_version``: load-time defaults absorb the
addition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlm.doc.sections import Section, SectionType
from dlm.io.atomic import write_text as atomic_write_text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dlm.store.paths import StorePath


_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PendingSectionPlan:
    """Generic staged plan payload — domain modules subclass this for typing."""

    source_path: Path
    created_at: str
    sections: tuple[Section, ...]


def pending_plan_path(store: StorePath, *, subdir: str) -> Path:
    """Path to the staged payload for `store` under the given subdir."""
    return store.root / subdir / "pending.json"


def save_pending_plan(
    store: StorePath,
    *,
    source_path: Path,
    sections: Sequence[Section],
    subdir: str,
    plan_cls: type[PendingSectionPlan],
) -> PendingSectionPlan:
    plan = plan_cls(
        source_path=source_path.resolve(),
        created_at=_utcnow(),
        sections=tuple(sections),
    )
    path = pending_plan_path(store, subdir=subdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "source_path": str(plan.source_path),
        "created_at": plan.created_at,
        "sections": [_section_to_payload(section) for section in plan.sections],
    }
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return plan


def load_pending_plan(
    store: StorePath,
    *,
    subdir: str,
    plan_cls: type[PendingSectionPlan],
    error_cls: type[Exception],
    label: str,
) -> PendingSectionPlan | None:
    """Return the staged plan, or None when absent. Raises `error_cls` on corruption.

    `label` names the domain in error messages ("preference plan", "synth plan").
    """
    path = pending_plan_path(store, subdir=subdir)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise error_cls(f"could not read staged {label}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise error_cls(f"staged {label} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise error_cls(f"staged {label} must be a JSON object")
    if raw.get("schema_version") != _SCHEMA_VERSION:
        raise error_cls(f"unsupported staged {label} schema_version={raw.get('schema_version')!r}")

    source_path = raw.get("source_path")
    created_at = raw.get("created_at")
    sections_raw = raw.get("sections")
    if not isinstance(source_path, str) or not source_path:
        raise error_cls(f"staged {label} is missing source_path")
    if not isinstance(created_at, str) or not created_at:
        raise error_cls(f"staged {label} is missing created_at")
    if not isinstance(sections_raw, list):
        raise error_cls(f"staged {label} is missing sections")

    sections: list[Section] = []
    for idx, entry in enumerate(sections_raw):
        try:
            sections.append(_section_from_payload(entry))
        except (TypeError, ValueError, KeyError) as exc:
            raise error_cls(f"invalid section payload at index {idx}: {exc}") from exc

    return plan_cls(
        source_path=Path(source_path),
        created_at=created_at,
        sections=tuple(sections),
    )


def clear_pending_plan(store: StorePath, *, subdir: str) -> bool:
    """Delete the staged plan; True iff it existed."""
    path = pending_plan_path(store, subdir=subdir)
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
