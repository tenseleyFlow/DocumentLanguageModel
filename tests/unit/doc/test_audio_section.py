"""Schema v11 — `SectionType.AUDIO` parse / serialize / round-trip.

Covers:

- Fence attribute grammar (both `path` + `transcript` required).
- Section identity — media path + blob sha, transcript metadata only.
- Round-trip through serializer.
- Rejections: missing `path`, missing `transcript`, unknown attribute,
  double-quote in transcript (serializer refuses).
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
dlm_version: 12
base_model: smollm2-135m
---
"""


def _doc(body: str) -> str:
    return _FRONTMATTER + "\n" + body


class TestAudioFenceGrammar:
    def test_path_and_transcript(self) -> None:
        text = _doc('::audio path="clips/hello.wav" transcript="Hello there."::\n')
        parsed = parse_text(text)
        assert len(parsed.sections) == 1
        section = parsed.sections[0]
        assert section.type == SectionType.AUDIO
        assert section.media_path == "clips/hello.wav"
        assert section.media_transcript == "Hello there."

    def test_audio_with_caption_body(self) -> None:
        text = _doc(
            '::audio path="clips/hello.wav" transcript="Hello there."::\n'
            "Recorded in a quiet room.\n"
        )
        parsed = parse_text(text)
        section = parsed.sections[0]
        assert section.type == SectionType.AUDIO
        assert section.content.strip() == "Recorded in a quiet room."

    def test_missing_path(self) -> None:
        text = _doc('::audio transcript="no path"::\n')
        with pytest.raises(FenceError, match="missing required attribute"):
            parse_text(text)

    def test_missing_transcript(self) -> None:
        text = _doc('::audio path="clips/x.wav"::\n')
        with pytest.raises(FenceError, match="missing required attribute"):
            parse_text(text)

    def test_bare_audio_fence_rejected(self) -> None:
        text = _doc("::audio::\n")
        with pytest.raises(FenceError, match="requires attributes"):
            parse_text(text)

    def test_unknown_attribute(self) -> None:
        text = _doc('::audio path="x.wav" transcript="t" sample_rate="16000"::\n')
        with pytest.raises(FenceError, match="unknown attribute"):
            parse_text(text)

    def test_transcript_with_punctuation(self) -> None:
        text = _doc('::audio path="clips/q.wav" transcript="Question: is this clear?"::\n')
        parsed = parse_text(text)
        assert parsed.sections[0].media_transcript == "Question: is this clear?"

    def test_transcript_with_embedded_quotes_rejected_at_parse(self) -> None:
        # Parser's attribute regex rejects `"` inside values.
        text = _doc('::audio path="x.wav" transcript="She said "hello""::\n')
        # Won't match the attribute-fence regex → falls through as
        # prose content. No AUDIO section produced.
        parsed = parse_text(text)
        assert not any(s.type == SectionType.AUDIO for s in parsed.sections)


class TestAudioSectionIdentity:
    def test_preingest_identity_is_path_only(self) -> None:
        a = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="clips/a.wav",
            media_transcript="hello",
        )
        b = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="clips/a.wav",
            media_transcript="hello",
        )
        assert a.section_id == b.section_id

    def test_transcript_change_does_not_affect_identity(self) -> None:
        # Transcript is training supervision metadata, not identity.
        # Editing the transcript produces a new training row against
        # the same audio bytes — same section_id.
        a = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="clips/a.wav",
            media_blob_sha="cc" * 32,
            media_transcript="hello",
        )
        b = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="clips/a.wav",
            media_blob_sha="cc" * 32,
            media_transcript="hello world",
        )
        assert a.section_id == b.section_id

    def test_blob_sha_change_flips_identity(self) -> None:
        a = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="clips/a.wav",
            media_blob_sha="aa" * 32,
            media_transcript="t",
        )
        b = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="clips/a.wav",
            media_blob_sha="bb" * 32,
            media_transcript="t",
        )
        assert a.section_id != b.section_id

    def test_audio_and_image_same_path_differ(self) -> None:
        # Type namespaces are disjoint.
        image = Section(
            type=SectionType.IMAGE,
            content="",
            media_path="shared/x",
            media_blob_sha="dd" * 32,
        )
        audio = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="shared/x",
            media_blob_sha="dd" * 32,
            media_transcript="t",
        )
        assert image.section_id != audio.section_id


class TestAudioSerializer:
    def test_roundtrip_minimal(self) -> None:
        text = _doc('::audio path="a.wav" transcript="hi"::\n')
        parsed = parse_text(text)
        assert serialize(parsed) == text

    def test_roundtrip_with_caption(self) -> None:
        text = _doc('::audio path="a.wav" transcript="hi"::\nRecorded indoors.\n')
        parsed = parse_text(text)
        assert serialize(parsed) == text

    def test_roundtrip_idempotent(self) -> None:
        text = _doc(
            '::audio path="clips/x.wav" transcript="Welcome to the session."::\n'
            "Recording quality notes.\n"
        )
        first = serialize(parse_text(text))
        second = serialize(parse_text(first))
        assert first == second

    def test_serialize_refuses_quote_in_transcript(self) -> None:
        section = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="x.wav",
            media_transcript='she said "hello"',
        )
        # Build a minimal ParsedDlm to serialize.
        from dlm.doc.parser import ParsedDlm
        from dlm.doc.schema import DlmFrontmatter

        parsed = ParsedDlm(
            frontmatter=DlmFrontmatter(
                dlm_id="01JZZZZZZZZZZZZZZZZZZZZZZZ",
                base_model="smollm2-135m",
            ),
            sections=(section,),
        )
        with pytest.raises(ValueError, match="double-quotes or"):
            serialize(parsed)

    def test_serialize_refuses_newline_in_transcript(self) -> None:
        section = Section(
            type=SectionType.AUDIO,
            content="",
            media_path="x.wav",
            media_transcript="line one\nline two",
        )
        from dlm.doc.parser import ParsedDlm
        from dlm.doc.schema import DlmFrontmatter

        parsed = ParsedDlm(
            frontmatter=DlmFrontmatter(
                dlm_id="01JZZZZZZZZZZZZZZZZZZZZZZZ",
                base_model="smollm2-135m",
            ),
            sections=(section,),
        )
        with pytest.raises(ValueError, match="double-quotes or"):
            serialize(parsed)


class TestAudioMixedWithOtherTypes:
    def test_audio_followed_by_instruction(self) -> None:
        body = (
            '::audio path="a.wav" transcript="hi"::\n'
            "\n"
            "::instruction::\n"
            "### Q\n"
            "What did the speaker say?\n"
            "### A\n"
            "They said hi.\n"
        )
        parsed = parse_text(_doc(body))
        types = [s.type for s in parsed.sections]
        assert types == [SectionType.AUDIO, SectionType.INSTRUCTION]

    def test_audio_and_image_in_same_doc(self) -> None:
        body = '::image path="fig.png"::\n\n::audio path="a.wav" transcript="hi"::\n'
        parsed = parse_text(_doc(body))
        types = [s.type for s in parsed.sections]
        assert types == [SectionType.IMAGE, SectionType.AUDIO]


class TestSchemaVersionBump:
    def test_current_schema_is_at_least_v11(self) -> None:
        from dlm.doc.schema import CURRENT_SCHEMA_VERSION

        assert CURRENT_SCHEMA_VERSION >= 11

    def test_v10_document_upgrades_to_v11(self) -> None:
        # v10→v11 is identity — a v10 doc (no audio fences) loads clean.
        body = (
            "---\n"
            "dlm_id: 01JZZZZZZZZZZZZZZZZZZZZZZZ\n"
            "dlm_version: 10\n"
            "base_model: smollm2-135m\n"
            "---\n"
            "::instruction::\n### Q\nhi?\n### A\nyes.\n"
        )
        parsed = parse_text(body)
        from dlm.doc.schema import CURRENT_SCHEMA_VERSION

        assert parsed.frontmatter.dlm_version == CURRENT_SCHEMA_VERSION
