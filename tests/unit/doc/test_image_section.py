"""Schema v10 — `SectionType.IMAGE` parse / serialize / round-trip.

Covers:

- Fence attribute grammar (required `path`, optional `alt`).
- Section identity pre- and post-blob-ingest.
- Serializer emits the attribute fence back.
- Rejections: missing `path`, unknown attribute, repeated attribute,
  non-ASCII value, newline in value, adapter-suffix form on IMAGE.
"""

from __future__ import annotations

import pytest

from dlm.doc.errors import FenceError
from dlm.doc.parser import parse_text
from dlm.doc.sections import Section, SectionType
from dlm.doc.serializer import serialize

_FRONTMATTER = """\
---
dlm_id: 01JZZZZZZZZZZZZZZZZZZZZZZZ
dlm_version: 10
base_model: smollm2-135m
---
"""


def _doc(body: str) -> str:
    return _FRONTMATTER + "\n" + body


class TestImageFenceGrammar:
    def test_path_only(self) -> None:
        text = _doc('::image path="figures/fig1.png"::\n')
        parsed = parse_text(text)
        assert len(parsed.sections) == 1
        section = parsed.sections[0]
        assert section.type == SectionType.IMAGE
        assert section.media_path == "figures/fig1.png"
        assert section.media_alt is None

    def test_path_and_alt(self) -> None:
        text = _doc('::image path="arch.png" alt="diagram of the pipeline"::\n')
        parsed = parse_text(text)
        section = parsed.sections[0]
        assert section.media_path == "arch.png"
        assert section.media_alt == "diagram of the pipeline"

    def test_image_with_caption_body(self) -> None:
        text = _doc('::image path="arch.png"::\nCaption for the image.\n')
        parsed = parse_text(text)
        section = parsed.sections[0]
        assert section.type == SectionType.IMAGE
        assert section.content.strip() == "Caption for the image."

    def test_bare_image_fence_rejected(self) -> None:
        # `::image::` without attributes — the type requires at least `path`.
        text = _doc("::image::\n")
        with pytest.raises(FenceError, match="requires attributes"):
            parse_text(text)

    def test_missing_path(self) -> None:
        text = _doc('::image alt="oops no path"::\n')
        with pytest.raises(FenceError, match="missing required attribute"):
            parse_text(text)

    def test_unknown_attribute(self) -> None:
        text = _doc('::image path="a.png" caption="oops"::\n')
        with pytest.raises(FenceError, match="unknown attribute"):
            parse_text(text)

    def test_repeated_attribute(self) -> None:
        text = _doc('::image path="a.png" path="b.png"::\n')
        with pytest.raises(FenceError, match="repeats attribute"):
            parse_text(text)

    def test_non_ascii_attribute_value(self) -> None:
        text = _doc('::image path="résumé.png"::\n')
        with pytest.raises(FenceError, match="non-ASCII"):
            parse_text(text)

    def test_newline_in_attribute_value_breaks_fence_match(self) -> None:
        # A literal newline splits the fence across lines; neither the
        # attribute-fence regex nor the bare-fence regex then matches
        # either fragment. Both lines fall through as prose content so
        # we never synthesize a malformed IMAGE section.
        text = _doc('::image path="a\npng"::\n')
        parsed = parse_text(text)
        types = [s.type for s in parsed.sections]
        assert SectionType.IMAGE not in types

    def test_adapter_suffix_on_image_not_recognized(self) -> None:
        # IMAGE does not route through the adapter-suffix grammar; the
        # `#foo` prefix + attribute-form together don't match either
        # fence shape, so the line falls through as prose content.
        text = _doc('::image#foo path="a.png"::\n')
        parsed = parse_text(text)
        assert not any(s.type == SectionType.IMAGE for s in parsed.sections)

    def test_bare_adapter_suffixed_image_fence_rejected(self) -> None:
        # `::image#foo::` (attribute-free) is a valid bare-fence shape,
        # but IMAGE requires attributes — the resolver must reject
        # rather than silently creating a zero-media section.
        text = _doc("::image#foo::\n")
        with pytest.raises(FenceError, match="requires attributes"):
            parse_text(text)


class TestImageSectionIdentity:
    def test_preingest_identity_is_path_only(self) -> None:
        a = Section(type=SectionType.IMAGE, content="", media_path="x.png")
        b = Section(type=SectionType.IMAGE, content="", media_path="x.png")
        assert a.section_id == b.section_id

    def test_preingest_different_paths_differ(self) -> None:
        a = Section(type=SectionType.IMAGE, content="", media_path="x.png")
        b = Section(type=SectionType.IMAGE, content="", media_path="y.png")
        assert a.section_id != b.section_id

    def test_blob_sha_changes_identity(self) -> None:
        a = Section(
            type=SectionType.IMAGE, content="", media_path="x.png", media_blob_sha="aa" * 32
        )
        b = Section(
            type=SectionType.IMAGE, content="", media_path="x.png", media_blob_sha="bb" * 32
        )
        assert a.section_id != b.section_id

    def test_same_blob_different_paths_differ(self) -> None:
        a = Section(
            type=SectionType.IMAGE, content="", media_path="a.png", media_blob_sha="cc" * 32
        )
        b = Section(
            type=SectionType.IMAGE, content="", media_path="b.png", media_blob_sha="cc" * 32
        )
        assert a.section_id != b.section_id

    def test_caption_content_does_not_affect_identity(self) -> None:
        # Caption body is a routing metadata field, not part of image
        # content identity — matches how path+blob identity works for
        # replay corpus determinism.
        a = Section(
            type=SectionType.IMAGE,
            content="caption one",
            media_path="x.png",
            media_blob_sha="ee" * 32,
        )
        b = Section(
            type=SectionType.IMAGE,
            content="caption two",
            media_path="x.png",
            media_blob_sha="ee" * 32,
        )
        assert a.section_id == b.section_id


class TestImageSerializer:
    def test_roundtrip_path_only(self) -> None:
        text = _doc('::image path="a.png"::\n')
        parsed = parse_text(text)
        assert serialize(parsed) == text

    def test_roundtrip_with_alt(self) -> None:
        text = _doc('::image path="a.png" alt="hero"::\n')
        parsed = parse_text(text)
        assert serialize(parsed) == text

    def test_roundtrip_with_caption(self) -> None:
        text = _doc('::image path="a.png"::\nCaption line.\n')
        parsed = parse_text(text)
        assert serialize(parsed) == text

    def test_roundtrip_idempotent(self) -> None:
        text = _doc('::image path="a.png" alt="hero"::\nCaption.\n')
        first = serialize(parse_text(text))
        second = serialize(parse_text(first))
        assert first == second


class TestImageInMixedBody:
    def test_prose_then_image(self) -> None:
        # Sections continue until the next fence, so prose after an
        # image gets folded into the image's caption body — the same
        # rule that applies to INSTRUCTION/PREFERENCE sections today.
        body = (
            "Opening prose.\n"
            "\n"
            '::image path="fig1.png" alt="first figure"::\n'
            "Figure one caption.\n"
        )
        parsed = parse_text(_doc(body))
        types = [s.type for s in parsed.sections]
        assert types == [SectionType.PROSE, SectionType.IMAGE]
        assert parsed.sections[1].media_path == "fig1.png"
        assert parsed.sections[1].media_alt == "first figure"

    def test_image_followed_by_instruction(self) -> None:
        body = (
            '::image path="x.png"::\n'
            "\n"
            "::instruction::\n"
            "### Q\n"
            "What is in the image?\n"
            "### A\n"
            "A figure.\n"
        )
        parsed = parse_text(_doc(body))
        types = [s.type for s in parsed.sections]
        assert types == [SectionType.IMAGE, SectionType.INSTRUCTION]
