"""Trainer-side directive integration: `_expand_directives` merges."""

from __future__ import annotations

from pathlib import Path

from dlm.doc.parser import parse_text
from dlm.train.trainer import _expand_directives

_VALID_ULID = "01ABCDEFGHJKMNPQRSTVWXYZ00"


def _make_parsed(body_yaml: str, base_path: Path):
    dlm_path = base_path / "doc.dlm"
    text = f"""---
dlm_id: {_VALID_ULID}
dlm_version: 6
base_model: smollm2-135m
training:
{body_yaml}
---

body prose
"""
    dlm_path.write_text(text)
    return parse_text(text, path=dlm_path)


def test_no_directives_is_passthrough(tmp_path: Path) -> None:
    parsed = _make_parsed("  precision: fp32\n", tmp_path)
    new_parsed, provenance, discovered = _expand_directives(parsed)
    assert new_parsed is parsed
    assert provenance == ()
    assert discovered == ()


def test_directives_merge_sections(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("print(1)\n")
    parsed = _make_parsed(
        "  sources:\n    - path: src\n      include: ['**/*.py']\n",
        tmp_path,
    )
    original_count = len(parsed.sections)
    new_parsed, provenance, _discovered = _expand_directives(parsed)

    # Original sections preserved, directive section appended.
    assert len(new_parsed.sections) == original_count + 1
    assert new_parsed.sections[:original_count] == parsed.sections
    assert new_parsed.sections[-1].content.startswith("# source: a.py")

    # Frontmatter + path preserved on the replaced dataclass.
    assert new_parsed.frontmatter == parsed.frontmatter
    assert new_parsed.source_path == parsed.source_path

    assert len(provenance) == 1
    assert provenance[0].path == "src"
    assert provenance[0].file_count == 1


def test_empty_directive_yields_empty_provenance_but_keeps_parsed(tmp_path: Path) -> None:
    src = tmp_path / "empty"
    src.mkdir()
    parsed = _make_parsed(
        "  sources:\n    - path: empty\n      include: ['**/*.py']\n",
        tmp_path,
    )
    new_parsed, provenance, _discovered = _expand_directives(parsed)
    # No files matched → no sections added → parsed returned as-is
    # with provenance recording the empty result.
    assert new_parsed is parsed
    assert len(provenance) == 1
    assert provenance[0].file_count == 0
