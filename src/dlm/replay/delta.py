"""Content-delta against a previous training run's manifest.

`ChangeSet` classifies every section in the current parsed `.dlm` as
`new` or `unchanged`, and every `section_id` recorded in the previous
`manifest.content_hashes` that's missing from the current document as
`removed`.

No `changed` field
------------------

Sections are identified purely by their content hash (`doc.sections`
computes `sha256(type || content)[:16]`). There is no stable
cross-edit identity — a section whose content changes gets a different
`section_id`, so it appears as `new` (plus the previous id in
`removed`). Distinguishing "edited" from "replaced" requires an
explicit per-section anchor. If Sprint 20 introduces anchor-based
identity, it can re-add `ChangeSet.changed` with a real implementation
at that point; carrying a reserved-but-always-empty field today just
invites consumers to write code against it that never fires.

The sampler in Sprint 08 needs only `new` (freshly-append-to-replay),
`unchanged` (already-present training signal), and `removed` (for
forgetting bookkeeping) — all three are non-lossy under this design.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dlm.doc.sections import Section
from dlm.store.manifest import Manifest


@dataclass(frozen=True)
class ChangeSet:
    """Result of a content-delta against a previous manifest."""

    new: list[Section] = field(default_factory=list)
    unchanged: list[Section] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def diff_against_manifest(sections: list[Section], manifest: Manifest) -> ChangeSet:
    """Classify each section vs `manifest.content_hashes`.

    `manifest.content_hashes` is `section_id → section_id` under the
    current design (the "hash" IS the section id). A section whose id
    is already in the manifest is `unchanged`; otherwise `new`. Any
    manifest id missing from the current document is `removed`.
    """
    prior_ids = set(manifest.content_hashes.keys())
    current_ids: set[str] = set()
    new: list[Section] = []
    unchanged: list[Section] = []

    for section in sections:
        sid = section.section_id
        if sid in current_ids:
            # Duplicate content in the same document — keep order but
            # only classify once; duplicates hit the same training row.
            continue
        current_ids.add(sid)
        if sid in prior_ids:
            unchanged.append(section)
        else:
            new.append(section)

    removed = sorted(prior_ids - current_ids)
    return ChangeSet(new=new, unchanged=unchanged, removed=removed)
