"""Directive walker audio-extension dispatch (Sprint 35.2 T4).

Parallel to `test_blob_ingest.py` (image dispatch). The walker sees a
`.wav` / `.flac` / `.ogg` file, looks up a sibling `<stem>.txt` for
the transcript, hands the bytes to the BlobStore, and emits a
`Section(type=AUDIO, ...)` with `media_transcript` populated.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from dlm.directives.expand import expand_sources
from dlm.doc.parser import parse_text
from dlm.doc.sections import SectionType
from dlm.store.blobs import BlobStore


def _dlm(body: str = "") -> str:
    return (
        "---\n"
        "dlm_id: 01KPMGSTNGSTTSTTSTTSTTSTVA\n"
        "dlm_version: 11\n"
        "base_model: smollm2-135m\n"
        "training:\n"
        "  sources:\n"
        f"{body}"
        "---\n"
    )


def _parse(body: str) -> object:
    return parse_text(_dlm(body))


def _write_pair(
    corpus: Path, stem: str, audio_bytes: bytes, transcript: str, ext: str = ".wav"
) -> None:
    """Create a matched audio + transcript pair under `corpus/`."""
    (corpus / f"{stem}{ext}").write_bytes(audio_bytes)
    (corpus / f"{stem}.txt").write_text(transcript, encoding="utf-8")


class TestAudioExtensionDispatch:
    def test_wav_with_transcript_ingested(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        _write_pair(corpus, "hello", b"RIFF....fake wav", "Hello there.")
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.wav"]\n'
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)

        assert len(result.sections) == 1
        section = result.sections[0]
        assert section.type == SectionType.AUDIO
        assert section.media_path == "hello.wav"
        assert section.media_transcript == "Hello there."
        expected_sha = hashlib.sha256(b"RIFF....fake wav").hexdigest()
        assert section.media_blob_sha == expected_sha

    def test_flac_also_supported(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        _write_pair(corpus, "clip", b"fLaC....fake", "Clip one.", ext=".flac")
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.flac"]\n'
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)
        assert len(result.sections) == 1
        assert result.sections[0].type == SectionType.AUDIO

    def test_missing_transcript_skipped_not_ingested(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        # Audio without matching .txt sidecar.
        (corpus / "orphan.wav").write_bytes(b"RIFF....orphan")
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.wav"]\n'
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)
        assert result.sections == ()
        [prov] = result.provenance
        assert prov.skipped_audio_no_transcript == 1
        assert prov.audio_count == 0

    def test_missing_blob_store_counts_skip(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        _write_pair(corpus, "hello", b"audio bytes", "Hello.")
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.wav"]\n'
        )
        result = expand_sources(parsed, base_path=tmp_path, blob_store=None)
        assert result.sections == ()
        [prov] = result.provenance
        assert prov.skipped_audio_no_store == 1

    def test_mixed_text_audio_image_in_one_directive(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        _write_pair(corpus, "clip", b"wav bytes", "A clip.")
        (corpus / "fig.png").write_bytes(b"png bytes")
        (corpus / "notes.md").write_text("Notes.\n", encoding="utf-8")
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.wav", "**/*.png", "**/*.md"]\n'
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)
        kinds = {s.type for s in result.sections}
        assert SectionType.AUDIO in kinds
        assert SectionType.IMAGE in kinds
        assert SectionType.PROSE in kinds
        [prov] = result.provenance
        assert prov.audio_count == 1
        assert prov.image_count == 1
        assert prov.file_count == 1  # prose count only

    def test_transcript_stripped_of_whitespace(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        # Sidecar with surrounding whitespace + trailing newline.
        _write_pair(corpus, "x", b"bytes", "\n\n  Actual text.  \n")
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.wav"]\n'
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)
        assert result.sections[0].media_transcript == "Actual text."

    def test_extension_case_insensitive(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "LOUD.WAV").write_bytes(b"bytes")
        (corpus / "LOUD.txt").write_text("Transcript.", encoding="utf-8")
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.WAV"]\n'
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)
        assert len(result.sections) == 1
        assert result.sections[0].type == SectionType.AUDIO


class TestAudioProvenance:
    def test_audio_count_and_bytes(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        _write_pair(corpus, "a", b"a" * 100, "First.")
        _write_pair(corpus, "b", b"b" * 250, "Second.")
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.wav"]\n'
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)
        [prov] = result.provenance
        assert prov.audio_count == 2
        assert prov.audio_bytes == 350

    def test_max_files_cap_includes_audio(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for i in range(5):
            _write_pair(corpus, f"clip-{i}", f"payload {i}".encode(), f"Clip {i}.")
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.wav"]\n'
            "      max_files: 3\n"
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)
        assert len(result.sections) == 3

    def test_partial_corpus_reports_both_success_and_skip(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        _write_pair(corpus, "good", b"wav one", "Transcript one.")
        (corpus / "bad.wav").write_bytes(b"wav two")  # no sidecar
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.wav"]\n'
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)
        assert len(result.sections) == 1
        [prov] = result.provenance
        assert prov.audio_count == 1
        assert prov.skipped_audio_no_transcript == 1


class TestAudioSectionIdentity:
    def test_same_bytes_different_paths_distinct_section_ids(
        self, tmp_path: Path
    ) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        body = b"same-bytes"
        _write_pair(corpus, "a", body, "First path.")
        _write_pair(corpus, "b", body, "Second path.")
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.wav"]\n'
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)
        assert len(result.sections) == 2
        a, b = result.sections
        assert a.media_blob_sha == b.media_blob_sha  # same bytes → same blob
        assert a.section_id != b.section_id  # different paths → different IDs
