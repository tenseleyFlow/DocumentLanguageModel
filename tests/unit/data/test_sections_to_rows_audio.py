"""Sprint 35.2 — AUDIO section row emission.

Mirrors `test_sections_to_rows_image.py`. Each AUDIO section becomes
`{"audio_blob_sha", "audio_path", "text": "<|AUDIO|>\\n<transcript>"}`
plus the standard `_dlm_section_id` + `_dlm_row_tags` bookkeeping.
The row shape is path-based (not waveform-based) because TRL has no
audio auto-dispatch; a custom collator (T8) resolves the blob and
drives `preprocess_audio` with the content-addressed cache.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType

import pytest

from dlm.data.sections_to_rows import sections_to_rows
from dlm.doc.sections import Section, SectionType
from dlm.store.blobs import BlobStore


@pytest.fixture
def blob_store(tmp_path: Path) -> BlobStore:
    return BlobStore(tmp_path / "blobs")


@pytest.fixture
def audio_section(blob_store: BlobStore, tmp_path: Path) -> tuple[Section, bytes, str]:
    """Ingested AUDIO section + the raw bytes + the blob sha."""
    data = b"fake-wav-bytes"
    src = tmp_path / "hello.wav"
    src.write_bytes(data)
    handle = blob_store.put(src)
    section = Section(
        type=SectionType.AUDIO,
        content="",
        media_path="hello.wav",
        media_blob_sha=handle.sha,
        media_transcript="Hello there.",
    )
    return section, data, handle.sha


class TestAudioRowShape:
    def test_emits_expected_keys(
        self, blob_store: BlobStore, audio_section: tuple[Section, bytes, str]
    ) -> None:
        section, _, sha = audio_section
        rows = sections_to_rows([section], blob_store=blob_store)
        assert len(rows) == 1
        row = rows[0]
        assert set(row.keys()) == {
            "audio_blob_sha",
            "audio_path",
            "text",
            "_dlm_section_id",
            "_dlm_row_tags",
        }
        assert row["audio_blob_sha"] == sha
        assert row["text"] == "<|AUDIO|>\nHello there."

    def test_audio_path_resolves_to_blob(
        self, blob_store: BlobStore, audio_section: tuple[Section, bytes, str]
    ) -> None:
        section, data, _ = audio_section
        row = sections_to_rows([section], blob_store=blob_store)[0]
        path = Path(row["audio_path"])
        assert path.is_file()
        assert path.read_bytes() == data

    def test_custom_audio_token(
        self, blob_store: BlobStore, audio_section: tuple[Section, bytes, str]
    ) -> None:
        section, _, _ = audio_section
        row = sections_to_rows([section], blob_store=blob_store, audio_token="<|custom|>")[0]
        assert row["text"] == "<|custom|>\nHello there."

    def test_section_id_flows_through(
        self, blob_store: BlobStore, audio_section: tuple[Section, bytes, str]
    ) -> None:
        section, _, _ = audio_section
        row = sections_to_rows([section], blob_store=blob_store)[0]
        assert row["_dlm_section_id"] == section.section_id


class TestAudioTagsFlowThrough:
    def test_tags_propagate_into_row_tags(self, blob_store: BlobStore, tmp_path: Path) -> None:
        data = b"audio"
        src = tmp_path / "x.wav"
        src.write_bytes(data)
        handle = blob_store.put(src)
        section = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="x.wav",
            media_blob_sha=handle.sha,
            media_transcript="A clip.",
            tags=MappingProxyType({"source": "podcast", "lang": "en"}),
        )
        row = sections_to_rows([section], blob_store=blob_store)[0]
        assert row["_dlm_row_tags"] == {"source": "podcast", "lang": "en"}


class TestAudioRefusals:
    def test_missing_blob_store_refused(self, tmp_path: Path) -> None:
        section = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="x.wav",
            media_blob_sha="a" * 64,
            media_transcript="Hi.",
        )
        with pytest.raises(ValueError, match="requires a blob_store"):
            sections_to_rows([section], blob_store=None)

    def test_unset_blob_sha_refused(self, blob_store: BlobStore) -> None:
        section = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="x.wav",
            media_blob_sha=None,  # not ingested
            media_transcript="Hi.",
        )
        with pytest.raises(ValueError, match="media_blob_sha"):
            sections_to_rows([section], blob_store=blob_store)

    def test_empty_transcript_refused(self, blob_store: BlobStore, tmp_path: Path) -> None:
        data = b"bytes"
        src = tmp_path / "x.wav"
        src.write_bytes(data)
        handle = blob_store.put(src)
        section = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="x.wav",
            media_blob_sha=handle.sha,
            media_transcript="   ",  # whitespace-only after strip
        )
        with pytest.raises(ValueError, match="empty transcript"):
            sections_to_rows([section], blob_store=blob_store)

    def test_none_transcript_refused(self, blob_store: BlobStore, tmp_path: Path) -> None:
        data = b"bytes"
        src = tmp_path / "x.wav"
        src.write_bytes(data)
        handle = blob_store.put(src)
        section = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="x.wav",
            media_blob_sha=handle.sha,
            media_transcript=None,
        )
        with pytest.raises(ValueError, match="empty transcript"):
            sections_to_rows([section], blob_store=blob_store)


class TestAudioMixedWithText:
    """Emit audio alongside prose + instruction without cross-contamination."""

    def test_mixed_corpus_orders_preserved(self, blob_store: BlobStore, tmp_path: Path) -> None:
        data = b"audio"
        src = tmp_path / "clip.wav"
        src.write_bytes(data)
        handle = blob_store.put(src)
        prose = Section(type=SectionType.PROSE, content="Notes on X.")
        audio = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="clip.wav",
            media_blob_sha=handle.sha,
            media_transcript="Speaking.",
        )
        rows = sections_to_rows([prose, audio, prose], blob_store=blob_store)
        # Prose rows carry `text` only; audio row carries `audio_path`.
        assert len(rows) == 3
        assert "audio_path" not in rows[0]
        assert "audio_path" in rows[1]
        assert "audio_path" not in rows[2]


class TestAudioSectionIdStableAcrossTranscriptEdits:
    """Transcript changes do NOT change section_id (metadata not content)."""

    def test_transcript_edit_preserves_identity(
        self, blob_store: BlobStore, tmp_path: Path
    ) -> None:
        data = b"audio-bytes"
        src = tmp_path / "same.wav"
        src.write_bytes(data)
        handle = blob_store.put(src)
        a = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="same.wav",
            media_blob_sha=handle.sha,
            media_transcript="First version.",
        )
        b = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="same.wav",
            media_blob_sha=handle.sha,
            media_transcript="Corrected version.",
        )
        assert a.section_id == b.section_id
        row_a = sections_to_rows([a], blob_store=blob_store)[0]
        row_b = sections_to_rows([b], blob_store=blob_store)[0]
        # Same section id, different text (transcript change is a training
        # pair change, not a section identity change).
        assert row_a["_dlm_section_id"] == row_b["_dlm_section_id"]
        assert row_a["text"] != row_b["text"]


class TestAudioHashMatchesBlobSha:
    """Sanity: the emitted audio_blob_sha matches sha256 of the ingested bytes."""

    def test_sha_round_trip(self, blob_store: BlobStore, tmp_path: Path) -> None:
        data = b"deterministic bytes for hash check"
        src = tmp_path / "clip.wav"
        src.write_bytes(data)
        handle = blob_store.put(src)
        expected = hashlib.sha256(data).hexdigest()
        assert handle.sha == expected

        section = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="clip.wav",
            media_blob_sha=handle.sha,
            media_transcript="X.",
        )
        row = sections_to_rows([section], blob_store=blob_store)[0]
        assert row["audio_blob_sha"] == expected
