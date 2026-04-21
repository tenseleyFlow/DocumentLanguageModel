"""Turn `doc.sections.Section` objects into ready-to-train dict rows.

Per Sprint 07's shape table (extended by Sprint 35 v1 for media):

| Section type | Row shape |
|---|---|
| PROSE       | `{"text": <raw prose>}` |
| INSTRUCTION | one `{"messages": [{"role":"user","content":Q},{"role":"assistant","content":A}]}` per Q/A pair |
| PREFERENCE  | one `{"prompt":P,"chosen":C,"rejected":R}` per triple |
| IMAGE       | `{"images": [PIL.Image], "text": "<image>\\n<caption>"}` — matches TRL 1.2's `DataCollatorForVisionLanguageModeling` standard-LM contract |

IMAGE emission requires a `BlobStore` (to resolve `media_blob_sha`
into bytes) and the base's `image_token` placeholder (from
`VlPreprocessorPlan.image_token`). Callers that leave `blob_store=None`
with IMAGE sections in the input raise `ValueError` — the row shape
isn't viable without the actual bytes.

Every row carries `_dlm_section_id` so `splitter.split()` can key
deterministically on (seed, section_id) rather than row index. This is
what makes adding sections to a document *not* reshuffle the existing
train/val assignments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dlm.data.instruction_parser import parse_instruction_body
from dlm.data.preference_parser import parse_preference_body
from dlm.doc.sections import Section, SectionType

if TYPE_CHECKING:
    from dlm.store.blobs import BlobStore

_PROBE_MARKER = "!probe"
_PROBE_HEADER = f"### Q {_PROBE_MARKER}"


def _normalize_probe_markers(body: str) -> str:
    """Rewrite `### Q !probe` → `### Q` so the strict parser accepts it.

    Mirrors `dlm.eval.probes._normalize_probe_markers` (kept local to
    avoid a data → eval import). Probe-marked Q/A pairs still train
    exactly like plain pairs; the marker is only load-bearing for probe
    extraction. We drop it silently here rather than leak `!probe:` into
    the training question text.
    """
    if _PROBE_HEADER not in body:
        return body
    lines = body.splitlines()
    rewritten = [("### Q" if line.strip() == _PROBE_HEADER else line) for line in lines]
    return "\n".join(rewritten)


Row = dict[str, Any]

_DEFAULT_IMAGE_TOKEN = "<image>"


def sections_to_rows(
    sections: list[Section],
    *,
    blob_store: BlobStore | None = None,
    image_token: str = _DEFAULT_IMAGE_TOKEN,
) -> list[Row]:
    """Flatten every section into its row shape(s), preserving order.

    PROSE sections with empty content are dropped silently — blank
    regions of a document shouldn't create empty training rows. Empty
    INSTRUCTION / PREFERENCE bodies are parse errors (handled by the
    respective section parsers).

    IMAGE sections require `blob_store` (to resolve `media_blob_sha`
    into bytes) and use `image_token` as the textual placeholder — the
    base model's processor expands that placeholder into its fixed
    `num_image_tokens` slots at collate time. Passing `blob_store=None`
    with IMAGE sections in the input raises `ValueError`.
    """
    rows: list[Row] = []
    for section in sections:
        rows.extend(
            _section_to_rows(
                section,
                blob_store=blob_store,
                image_token=image_token,
            ),
        )
    return rows


def _section_to_rows(
    section: Section,
    *,
    blob_store: BlobStore | None,
    image_token: str,
) -> list[Row]:
    sid = section.section_id
    tags = dict(section.tags)
    if section.type is SectionType.PROSE:
        text = section.content.strip()
        if not text:
            return []
        return [{"text": text, "_dlm_section_id": sid, "_dlm_row_tags": tags}]

    if section.type is SectionType.INSTRUCTION:
        body = _normalize_probe_markers(section.content)
        pairs = parse_instruction_body(body, section_id=sid)
        return [
            {
                "messages": [
                    {"role": "user", "content": p.question},
                    {"role": "assistant", "content": p.answer},
                ],
                "_dlm_section_id": sid,
                "_dlm_row_tags": tags,
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
                "_dlm_row_tags": tags,
            }
            for t in triples
        ]

    if section.type is SectionType.IMAGE:
        return [_image_section_to_row(section, blob_store, image_token, sid, tags)]

    raise ValueError(f"unknown section type: {section.type!r}")  # pragma: no cover


def _image_section_to_row(
    section: Section,
    blob_store: BlobStore | None,
    image_token: str,
    sid: str,
    tags: dict[str, str],
) -> Row:
    """Emit a TRL 1.2-shaped VL row: `{images: [PIL], text: "<image>\\n..."}`.

    The caller (dataset_builder / trainer) owns the BlobStore; sections
    with `media_blob_sha is None` reach this path only via tests that
    construct Sections manually without ingest — we refuse them
    explicitly rather than silently dropping them.
    """
    if blob_store is None:
        raise ValueError(
            "sections_to_rows: IMAGE section requires a blob_store; "
            f"section {sid} has media_path={section.media_path!r}",
        )
    if section.media_blob_sha is None:
        raise ValueError(
            "sections_to_rows: IMAGE section has no media_blob_sha "
            f"(section {sid} hasn't been ingested through the blob store)",
        )

    from PIL import Image

    blob_path = blob_store.get(section.media_blob_sha)
    with Image.open(blob_path) as pil:
        pil.load()
        image = pil.convert("RGB")

    caption = section.content.strip()
    text = f"{image_token}\n{caption}" if caption else image_token
    return {
        "images": [image],
        "text": text,
        "_dlm_section_id": sid,
        "_dlm_row_tags": tags,
    }
