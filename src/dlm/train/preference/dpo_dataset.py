"""Assemble a DPO-shaped dataset from `::preference::` sections.

DPOTrainer's canonical row format is three strings:

    {"prompt": "...", "chosen": "...", "rejected": "..."}

We pre-pend `_dlm_section_id` + `_dlm_sub_index` so the deterministic
splitter can key on document-stable coordinates rather than row order —
same pattern the SFT row builders use.

The section-level parse (`parse_preference_body`) has already
validated that every triple has non-empty prompt/chosen/rejected
fields; this module only routes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dlm.data.preference_parser import PreferenceTriple, parse_preference_body
from dlm.doc.sections import Section, SectionType

if TYPE_CHECKING:
    from datasets import Dataset

DpoRow = dict[str, Any]


def extract_preference_triples(
    sections: list[Section],
) -> list[tuple[Section, PreferenceTriple]]:
    """Parse every `::preference::` body into triples, paired with the
    owning section so callers know where each triple came from.

    Non-preference sections are skipped silently; there's no error for
    "document has no preferences" here — the caller decides whether
    zero preferences is fatal (explicit `--phase dpo`) or warn-and-skip
    (auto-enabled).
    """
    pairs: list[tuple[Section, PreferenceTriple]] = []
    for section in sections:
        if section.type is not SectionType.PREFERENCE:
            continue
        triples = parse_preference_body(section.content, section_id=section.section_id)
        for triple in triples:
            pairs.append((section, triple))
    return pairs


def preference_rows(sections: list[Section]) -> list[DpoRow]:
    """Flatten preference sections into DPOTrainer-shaped rows.

    Each row carries `_dlm_section_id` + `_dlm_sub_index` so the
    deterministic splitter (shared with SFT) can key on document
    coordinates instead of row position. Empty document → empty list,
    never raises on its own.
    """
    rows: list[DpoRow] = []
    sub_indices: dict[str, int] = {}
    for section, triple in extract_preference_triples(sections):
        sid = section.section_id
        idx = sub_indices.get(sid, 0)
        sub_indices[sid] = idx + 1
        rows.append(
            {
                "prompt": triple.prompt,
                "chosen": triple.chosen,
                "rejected": triple.rejected,
                "_dlm_section_id": sid,
                "_dlm_sub_index": idx,
            }
        )
    return rows


def build_dpo_dataset(sections: list[Section]) -> Dataset:
    """Wrap `preference_rows()` output in a HF `Dataset`.

    Imported lazily: `datasets` is a ~20 MB import and a lot of non-DPO
    code paths have no business pulling it in.
    """
    from datasets import Dataset

    rows = preference_rows(sections)
    return Dataset.from_list(rows)
