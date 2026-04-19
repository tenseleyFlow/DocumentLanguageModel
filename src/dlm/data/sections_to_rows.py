"""Turn `doc.sections.Section` objects into ready-to-train dict rows.

Per Sprint 07's shape table:

| Section type | Row shape |
|---|---|
| PROSE       | `{"text": <raw prose>}` |
| INSTRUCTION | one `{"messages": [{"role":"user","content":Q},{"role":"assistant","content":A}]}` per Q/A pair |
| PREFERENCE  | one `{"prompt":P,"chosen":C,"rejected":R}` per triple |

Every row carries `_dlm_section_id` so `splitter.split()` can key
deterministically on (seed, section_id) rather than row index. This is
what makes adding sections to a document *not* reshuffle the existing
train/val assignments.
"""

from __future__ import annotations

from typing import Any

from dlm.data.instruction_parser import parse_instruction_body
from dlm.data.preference_parser import parse_preference_body
from dlm.doc.sections import Section, SectionType

Row = dict[str, Any]


def sections_to_rows(sections: list[Section]) -> list[Row]:
    """Flatten every section into its row shape(s), preserving order.

    PROSE sections with empty content are dropped silently — blank
    regions of a document shouldn't create empty training rows. Empty
    INSTRUCTION / PREFERENCE bodies are parse errors (handled by the
    respective section parsers).
    """
    rows: list[Row] = []
    for section in sections:
        rows.extend(_section_to_rows(section))
    return rows


def _section_to_rows(section: Section) -> list[Row]:
    sid = section.section_id
    if section.type is SectionType.PROSE:
        text = section.content.strip()
        if not text:
            return []
        return [{"text": text, "_dlm_section_id": sid}]

    if section.type is SectionType.INSTRUCTION:
        pairs = parse_instruction_body(section.content, section_id=sid)
        return [
            {
                "messages": [
                    {"role": "user", "content": p.question},
                    {"role": "assistant", "content": p.answer},
                ],
                "_dlm_section_id": sid,
            }
            for p in pairs
        ]

    if section.type is SectionType.PREFERENCE:
        triples = parse_preference_body(section.content, section_id=sid)
        return [
            {
                "prompt": t.prompt,
                "chosen": t.chosen,
                "rejected": t.rejected,
                "_dlm_section_id": sid,
            }
            for t in triples
        ]

    raise ValueError(f"unknown section type: {section.type!r}")  # pragma: no cover
