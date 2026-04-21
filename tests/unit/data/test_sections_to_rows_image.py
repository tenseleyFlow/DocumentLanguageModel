"""Sprint 35 v1 — IMAGE section row emission.

Covers the TRL 1.2 `DataCollatorForVisionLanguageModeling` contract:
each IMAGE section becomes `{"images": [PIL.Image], "text": "<image>\\n<caption>"}`
plus the standard `_dlm_section_id` + `_dlm_row_tags` bookkeeping.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType

import pytest
from PIL import Image

from dlm.data.sections_to_rows import sections_to_rows
from dlm.doc.sections import Section, SectionType
from dlm.store.blobs import BlobStore


def _make_image_bytes(color: tuple[int, int, int]) -> bytes:
    """Render a 4×4 PNG to bytes so we can compute a real sha + ingest it."""
    import io

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def blob_store(tmp_path: Path) -> BlobStore:
    return BlobStore(tmp_path / "blobs")


@pytest.fixture
def red_image(blob_store: BlobStore, tmp_path: Path) -> tuple[Section, bytes]:
    """Section + blob bytes for a red 4×4 image already ingested."""
    data = _make_image_bytes((255, 0, 0))
    src = tmp_path / "hero.png"
    src.write_bytes(data)
    handle = blob_store.put(src)
    section = Section(
        type=SectionType.IMAGE,
        content="",
        media_path="hero.png",
        media_alt="hero",
        media_blob_sha=handle.sha,
    )
    return section, data


class TestImageRowShape:
    def test_emits_images_and_text_keys(
        self, blob_store: BlobStore, red_image: tuple[Section, bytes]
    ) -> None:
        section, _ = red_image
        rows = sections_to_rows([section], blob_store=blob_store)
        assert len(rows) == 1
        row = rows[0]
        assert set(row.keys()) == {"images", "text", "_dlm_section_id", "_dlm_row_tags"}

    def test_images_is_list_of_pil(
        self, blob_store: BlobStore, red_image: tuple[Section, bytes]
    ) -> None:
        section, _ = red_image
        rows = sections_to_rows([section], blob_store=blob_store)
        images = rows[0]["images"]
        assert isinstance(images, list)
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
        assert images[0].mode == "RGB"
        assert images[0].size == (4, 4)

    def test_text_defaults_to_image_token_alone(
        self, blob_store: BlobStore, red_image: tuple[Section, bytes]
    ) -> None:
        section, _ = red_image
        rows = sections_to_rows([section], blob_store=blob_store)
        assert rows[0]["text"] == "<image>"

    def test_caption_appended_after_token(self, blob_store: BlobStore, tmp_path: Path) -> None:
        data = _make_image_bytes((0, 255, 0))
        (tmp_path / "fig.png").write_bytes(data)
        handle = blob_store.put(tmp_path / "fig.png")
        section = Section(
            type=SectionType.IMAGE,
            content="Figure 1: the architecture diagram.",
            media_path="fig.png",
            media_blob_sha=handle.sha,
        )
        rows = sections_to_rows([section], blob_store=blob_store)
        assert rows[0]["text"] == "<image>\nFigure 1: the architecture diagram."

    def test_image_token_is_configurable(
        self, blob_store: BlobStore, red_image: tuple[Section, bytes]
    ) -> None:
        section, _ = red_image
        rows = sections_to_rows([section], blob_store=blob_store, image_token="<|image|>")
        assert rows[0]["text"] == "<|image|>"

    def test_bytes_round_trip_match_blob(
        self, blob_store: BlobStore, red_image: tuple[Section, bytes]
    ) -> None:
        # The PIL object we emit was loaded from the blob — re-encode
        # and confirm the bytes are the same image we stored.
        section, data = red_image
        rows = sections_to_rows([section], blob_store=blob_store)
        pil = rows[0]["images"][0]
        assert (
            pil.tobytes()
            == Image.open(blob_store.get(section.media_blob_sha or "")).convert("RGB").tobytes()
        )


class TestImageRowProvenance:
    def test_section_id_preserved(
        self, blob_store: BlobStore, red_image: tuple[Section, bytes]
    ) -> None:
        section, _ = red_image
        rows = sections_to_rows([section], blob_store=blob_store)
        assert rows[0]["_dlm_section_id"] == section.section_id

    def test_tags_copied(self, blob_store: BlobStore, tmp_path: Path) -> None:
        data = _make_image_bytes((0, 0, 255))
        (tmp_path / "x.png").write_bytes(data)
        handle = blob_store.put(tmp_path / "x.png")
        section = Section(
            type=SectionType.IMAGE,
            content="",
            media_path="x.png",
            media_blob_sha=handle.sha,
            tags=MappingProxyType({"modality": "image", "domain": "arch"}),
        )
        rows = sections_to_rows([section], blob_store=blob_store)
        assert rows[0]["_dlm_row_tags"] == {"modality": "image", "domain": "arch"}


class TestImageRowMixedBatch:
    def test_image_row_alongside_prose(
        self, blob_store: BlobStore, red_image: tuple[Section, bytes]
    ) -> None:
        prose = Section(type=SectionType.PROSE, content="intro line")
        img, _ = red_image
        rows = sections_to_rows([prose, img], blob_store=blob_store)
        assert len(rows) == 2
        assert "text" in rows[0]
        assert "images" not in rows[0]
        assert rows[1]["text"] == "<image>"
        assert "images" in rows[1]


class TestImageRowRefusals:
    def test_refuses_without_blob_store(self) -> None:
        section = Section(
            type=SectionType.IMAGE,
            content="",
            media_path="x.png",
            media_blob_sha="a" * 64,
        )
        with pytest.raises(ValueError, match="requires a blob_store"):
            sections_to_rows([section])

    def test_refuses_without_blob_sha(self, blob_store: BlobStore) -> None:
        section = Section(
            type=SectionType.IMAGE,
            content="",
            media_path="x.png",
            media_blob_sha=None,
        )
        with pytest.raises(ValueError, match="hasn't been ingested"):
            sections_to_rows([section], blob_store=blob_store)


class TestNonImageRowsStableWithBlobStore:
    """Passing a blob_store must not perturb PROSE/INSTRUCTION/PREFERENCE emission.

    The image path is opt-in per section type; everything else reaches
    its existing code path unchanged.
    """

    def test_prose_row_unchanged(self, blob_store: BlobStore) -> None:
        section = Section(type=SectionType.PROSE, content="hello world")
        rows = sections_to_rows([section], blob_store=blob_store)
        assert rows == [
            {
                "text": "hello world",
                "_dlm_section_id": section.section_id,
                "_dlm_row_tags": {},
            }
        ]


def test_sha_still_matches_blob(blob_store: BlobStore, tmp_path: Path) -> None:
    """Meta: our fixture's handle.sha equals sha256(bytes)."""
    data = _make_image_bytes((128, 128, 128))
    (tmp_path / "z.png").write_bytes(data)
    handle = blob_store.put(tmp_path / "z.png")
    assert handle.sha == hashlib.sha256(data).hexdigest()
