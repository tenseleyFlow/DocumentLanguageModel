"""Turn `doc.sections.Section` objects into ready-to-train dict rows.

Current shape table:

| Section type | Row shape |
|---|---|
| PROSE       | `{"text": <raw prose>}` |
| INSTRUCTION | one `{"messages": [{"role":"user","content":Q},{"role":"assistant","content":A}]}` per Q/A pair |
| PREFERENCE  | one `{"prompt":P,"chosen":C,"rejected":R}` per triple |
| IMAGE       | `{"images": [PIL.Image], "text": "<image>\\n<caption>"}` — matches TRL 1.2's `DataCollatorForVisionLanguageModeling` standard-LM contract |
| AUDIO       | `{"audio_blob_sha": sha, "audio_path": str, "text": "<|AUDIO|>\\n<transcript>"}` — path-based (TRL has no audio auto-dispatch; a custom collator resolves the blob and drives `preprocess_audio` at collate time) |

IMAGE / AUDIO emission requires a `BlobStore` (to resolve
`media_blob_sha` into bytes) and the base's placeholder token.
Callers that leave `blob_store=None` with media sections in the
input raise `ValueError` — the row shape isn't viable without the
actual bytes. Audio rows hold only the path + sha, not the decoded
waveform; the audio cache is the right place to hold preprocessed
features across epochs, and loading lazily at collate time keeps
dataset rows small.

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
_DEFAULT_AUDIO_TOKEN = "<|AUDIO|>"


def sections_to_rows(
    sections: list[Section],
    *,
    blob_store: BlobStore | None = None,
    image_token: str = _DEFAULT_IMAGE_TOKEN,
    audio_token: str = _DEFAULT_AUDIO_TOKEN,
) -> list[Row]:
    """Flatten every section into its row shape(s), preserving order.

    PROSE sections with empty content are dropped silently — blank
    regions of a document shouldn't create empty training rows. Empty
    INSTRUCTION / PREFERENCE bodies are parse errors (handled by the
    respective section parsers).

    IMAGE / AUDIO sections require `blob_store` (to resolve
    `media_blob_sha` into bytes) and use `image_token` / `audio_token`
    as the textual placeholder — the base model's processor expands
    that placeholder into its fixed token window at collate time.
    Passing `blob_store=None` with media sections in the input raises
    `ValueError`.
    """
    rows: list[Row] = []
    for section in sections:
        rows.extend(
            _section_to_rows(
                section,
                blob_store=blob_store,
                image_token=image_token,
                audio_token=audio_token,
            ),
        )
    return rows


def _section_to_rows(
    section: Section,
    *,
    blob_store: BlobStore | None,
    image_token: str,
    audio_token: str,
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

    if section.type is SectionType.AUDIO:
        return [_audio_section_to_row(section, blob_store, audio_token, sid, tags)]

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


def _audio_section_to_row(
    section: Section,
    blob_store: BlobStore | None,
    audio_token: str,
    sid: str,
    tags: dict[str, str],
) -> Row:
    """Emit an audio row: path + sha + transcript-prefixed text.

    Audio rows carry the blob path + sha rather than a decoded
    waveform so the custom collator (T8) can drive `preprocess_audio`
    with the content-addressed cache — decoding + feature-extraction
    is expensive and repeats across epochs. The transcript from the
    sibling `<stem>.txt` goes into the row's `text` after the
    placeholder; the custom collator replaces the placeholder with
    the feature-extractor's `num_audio_tokens` slots before the
    trainer sees it.

    Audio sections with an empty transcript aren't useful for SFT
    (no target text to predict) — refuse loudly rather than emit a
    placeholder-only row that would train the model to produce an
    empty response.
    """
    if blob_store is None:
        raise ValueError(
            "sections_to_rows: AUDIO section requires a blob_store; "
            f"section {sid} has media_path={section.media_path!r}",
        )
    if section.media_blob_sha is None:
        raise ValueError(
            "sections_to_rows: AUDIO section has no media_blob_sha "
            f"(section {sid} hasn't been ingested through the blob store)",
        )
    transcript = (section.media_transcript or "").strip()
    if not transcript:
        raise ValueError(
            "sections_to_rows: AUDIO section has empty transcript "
            f"(section {sid}, media_path={section.media_path!r}); "
            "transcript sibling `<stem>.txt` must be non-empty",
        )

    blob_path = blob_store.get(section.media_blob_sha)
    text = f"{audio_token}\n{transcript}"
    return {
        "audio_blob_sha": section.media_blob_sha,
        "audio_path": str(blob_path),
        "text": text,
        "_dlm_section_id": sid,
        "_dlm_row_tags": tags,
    }
