"""Directive walker image-extension dispatch (Sprint 35 v1).

The walker sees a `.png` / `.jpg` / `.webp` file, hands its bytes to
the `BlobStore`, and synthesizes a `Section(type=IMAGE, ...)` with
`media_path` = relpath and `media_blob_sha` = the hash. Without a
`blob_store` argument, image files are tallied under
`skipped_image_no_store` so `dlm show` can surface "would ingest N
images on next train" without touching disk.
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
        "dlm_version: 10\n"
        "base_model: smollm2-135m\n"
        "training:\n"
        "  sources:\n"
        f"{body}"
        "---\n"
    )


def _parse(body: str) -> object:
    return parse_text(_dlm(body))


class TestImageExtensionDispatch:
    def test_png_ingested_as_image(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "arch.png").write_bytes(b"\x89PNG\r\n\x1a\nbody")
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.png"]\n'
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)

        assert len(result.sections) == 1
        section = result.sections[0]
        assert section.type == SectionType.IMAGE
        assert section.media_path == "arch.png"
        assert section.media_alt == "arch"
        expected_sha = hashlib.sha256(b"\x89PNG\r\n\x1a\nbody").hexdigest()
        assert section.media_blob_sha == expected_sha

    def test_missing_blob_store_counts_skip(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "hero.jpg").write_bytes(b"jpeg body")
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.jpg"]\n'
        )
        result = expand_sources(parsed, base_path=tmp_path, blob_store=None)
        assert result.sections == ()
        [prov] = result.provenance
        assert prov.skipped_image_no_store == 1
        assert prov.image_count == 0

    def test_text_and_image_mix_in_one_directive(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "hero.png").write_bytes(b"png bytes")
        (corpus / "notes.md").write_text("Notes.\n", encoding="utf-8")
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.png", "**/*.md"]\n'
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)
        kinds = [s.type for s in result.sections]
        assert SectionType.IMAGE in kinds
        assert SectionType.PROSE in kinds
        [prov] = result.provenance
        assert prov.image_count == 1
        assert prov.file_count == 1  # prose count only

    def test_identical_bytes_different_paths_distinct_section_ids(
        self, tmp_path: Path
    ) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        body = b"same-bytes-shared"
        (corpus / "a.png").write_bytes(body)
        (corpus / "b.png").write_bytes(body)
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.png"]\n'
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)
        assert len(result.sections) == 2
        a, b = result.sections
        assert a.media_blob_sha == b.media_blob_sha
        assert a.section_id != b.section_id

    def test_image_file_bypasses_binary_skip(self, tmp_path: Path) -> None:
        # A PNG starts with a NUL-free signature but most JPEGs contain
        # NUL in the first KiB. The text-read path's binary heuristic
        # would skip that; the image dispatch must take precedence.
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        payload = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00rest"
        (corpus / "photo.jpg").write_bytes(payload)
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.jpg"]\n'
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)
        assert len(result.sections) == 1
        assert result.sections[0].type == SectionType.IMAGE

    def test_extension_is_case_insensitive(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "FIG.PNG").write_bytes(b"uppercase extension")
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.PNG"]\n'
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)
        assert len(result.sections) == 1
        assert result.sections[0].type == SectionType.IMAGE


class TestImageAltDefaults:
    def test_alt_defaults_to_stem(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "pipeline-v2.png").write_bytes(b"bytes")
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.png"]\n'
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)
        assert result.sections[0].media_alt == "pipeline-v2"


class TestImageProvenance:
    def test_image_count_and_bytes(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.png").write_bytes(b"a" * 100)
        (corpus / "b.png").write_bytes(b"b" * 200)
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.png"]\n'
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)
        [prov] = result.provenance
        assert prov.image_count == 2
        assert prov.image_bytes == 300

    def test_max_files_cap_includes_images(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for i in range(5):
            (corpus / f"{i}.png").write_bytes(f"payload {i}".encode())
        parsed = _parse(
            f"    - path: {corpus}\n"
            '      include: ["**/*.png"]\n'
            "      max_files: 3\n"
        )
        blob_store = BlobStore(tmp_path / "blobs")
        result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)
        assert len(result.sections) == 3
