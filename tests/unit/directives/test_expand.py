"""End-to-end directive expansion behavior.

Covers: sections synthesis, provenance counts, caps enforcement,
binary skip, encoding skip, path-header identity, and the empty-
directives fast-path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dlm.directives import expand_sources
from dlm.directives.errors import DirectivePathError, DirectivePolicyError
from dlm.directives.expand import _iter_candidates
from dlm.doc.parser import parse_text
from dlm.doc.sections import SectionType
from dlm.store.blobs import BlobStore

_VALID_ULID = "01ABCDEFGHJKMNPQRSTVWXYZ00"


def _make_parsed(body_yaml: str, base_path: Path) -> tuple[object, Path]:
    """Build a ParsedDlm pointing at a dlm file inside `base_path`."""
    dlm_path = base_path / "doc.dlm"
    text = f"""---
dlm_id: {_VALID_ULID}
dlm_version: 6
base_model: smollm2-135m
training:
{body_yaml}
---
body
"""
    dlm_path.write_text(text)
    return parse_text(text, path=dlm_path), dlm_path


def test_no_sources_returns_empty(tmp_path: Path) -> None:
    parsed, _ = _make_parsed("  precision: fp32\n", tmp_path)
    result = expand_sources(parsed, base_path=tmp_path)  # type: ignore[arg-type]
    assert result.sections == ()
    assert result.provenance == ()


def test_single_directory_directive(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("print(1)\n")
    (src / "b.py").write_text("print(2)\n")
    body = "  sources:\n    - path: src\n      include: ['**/*.py']\n"
    parsed, _ = _make_parsed(body, tmp_path)

    result = expand_sources(parsed, base_path=tmp_path)  # type: ignore[arg-type]
    assert len(result.sections) == 2
    assert all(s.type == SectionType.PROSE for s in result.sections)
    # Content carries path header → distinct IDs even if bodies matched.
    assert result.sections[0].content.startswith("# source: a.py")
    assert result.sections[1].content.startswith("# source: b.py")
    assert len(result.provenance) == 1
    prov = result.provenance[0]
    assert prov.file_count == 2
    assert prov.total_bytes == len("print(1)\n") + len("print(2)\n")


def test_path_header_gives_distinct_ids(tmp_path: Path) -> None:
    """Two identical-content files in different paths must hash distinctly."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("same\n")
    (src / "b.py").write_text("same\n")
    body = "  sources:\n    - path: src\n      include: ['**/*.py']\n"
    parsed, _ = _make_parsed(body, tmp_path)

    result = expand_sources(parsed, base_path=tmp_path)  # type: ignore[arg-type]
    ids = {s.section_id for s in result.sections}
    assert len(ids) == 2


def test_max_files_truncates_deterministically(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(5):
        (src / f"{i}.py").write_text(f"# {i}\n")
    body = "  sources:\n    - path: src\n      include: ['**/*.py']\n      max_files: 2\n"
    parsed, _ = _make_parsed(body, tmp_path)
    result = expand_sources(parsed, base_path=tmp_path)  # type: ignore[arg-type]
    # Sorted: 0.py, 1.py land; 2/3/4 get dropped
    assert [s.content.split("\n")[0] for s in result.sections] == [
        "# source: 0.py",
        "# source: 1.py",
    ]


def test_max_bytes_per_file_skips_oversize(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "small.py").write_text("x\n")  # 2 bytes
    (src / "big.py").write_text("x" * 100)
    body = "  sources:\n    - path: src\n      include: ['**/*.py']\n      max_bytes_per_file: 10\n"
    parsed, _ = _make_parsed(body, tmp_path)
    result = expand_sources(parsed, base_path=tmp_path)  # type: ignore[arg-type]
    assert len(result.sections) == 1
    assert result.sections[0].content.startswith("# source: small.py")
    prov = result.provenance[0]
    assert prov.skipped_over_size == 1


def test_per_file_skip_logs_are_debug_not_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Audit 13 M13.5: per-file skip messages were emitted at INFO,
    spamming 243 lines to stderr on every ``dlm show`` of a 2k-file
    corpus. They're now DEBUG so the default-level log stream stays
    clean while ``--verbose``/``LOG_LEVEL=DEBUG`` still surfaces them
    for diagnosis. The summary count remains in provenance."""
    import logging

    src = tmp_path / "src"
    src.mkdir()
    for n in range(5):
        (src / f"big{n}.py").write_text("x" * 100)
    body = "  sources:\n    - path: src\n      include: ['**/*.py']\n      max_bytes_per_file: 10\n"
    parsed, _ = _make_parsed(body, tmp_path)

    with caplog.at_level(logging.INFO, logger="dlm.directives.expand"):
        result = expand_sources(parsed, base_path=tmp_path)  # type: ignore[arg-type]

    assert result.provenance[0].skipped_over_size == 5
    info_records = [r for r in caplog.records if r.levelno >= logging.INFO]
    assert info_records == [], (
        f"per-file skip should not emit INFO records, got: {[r.message for r in info_records]}"
    )


def test_binary_file_skipped(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "binary.dat").write_bytes(b"\x00\xff\x00content")
    (src / "text.txt").write_text("hello\n")
    body = "  sources:\n    - path: src\n      include: ['**/*']\n"
    parsed, _ = _make_parsed(body, tmp_path)
    result = expand_sources(parsed, base_path=tmp_path)  # type: ignore[arg-type]
    assert len(result.sections) == 1
    assert result.sections[0].content.startswith("# source: text.txt")
    prov = result.provenance[0]
    assert prov.skipped_binary == 1


def test_non_utf8_skipped(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    # Latin-1 bytes that are not valid UTF-8 (and have no NUL so
    # binary sniff lets them through).
    (src / "latin.txt").write_bytes(b"caf\xe9\n")
    (src / "utf.txt").write_text("utf\n")
    body = "  sources:\n    - path: src\n      include: ['**/*.txt']\n"
    parsed, _ = _make_parsed(body, tmp_path)
    result = expand_sources(parsed, base_path=tmp_path)  # type: ignore[arg-type]
    assert len(result.sections) == 1
    assert result.sections[0].content.startswith("# source: utf.txt")
    prov = result.provenance[0]
    assert prov.skipped_encoding == 1


def test_missing_path_raises_directive_path_error(tmp_path: Path) -> None:
    body = "  sources:\n    - path: nonexistent\n"
    parsed, _ = _make_parsed(body, tmp_path)
    with pytest.raises(DirectivePathError):
        expand_sources(parsed, base_path=tmp_path)  # type: ignore[arg-type]


def test_strict_policy_refuses_external_path(tmp_path: Path) -> None:
    outside = tmp_path.parent / "strict_outside"
    outside.mkdir(exist_ok=True)
    try:
        (outside / "a.py").write_text("x")
        body = f"  sources_policy: strict\n  sources:\n    - path: {outside}\n"
        parsed, _ = _make_parsed(body, tmp_path)
        with pytest.raises(DirectivePolicyError):
            expand_sources(parsed, base_path=tmp_path)  # type: ignore[arg-type]
    finally:
        for f in outside.iterdir():
            f.unlink()
        outside.rmdir()


def test_permissive_policy_allows_external_path(tmp_path: Path) -> None:
    outside = tmp_path.parent / "permissive_outside"
    outside.mkdir(exist_ok=True)
    try:
        (outside / "a.py").write_text("ok\n")
        body = f"  sources:\n    - path: {outside}\n      include: ['**/*.py']\n"
        parsed, _ = _make_parsed(body, tmp_path)
        result = expand_sources(parsed, base_path=tmp_path)  # type: ignore[arg-type]
        assert len(result.sections) == 1
    finally:
        for f in outside.iterdir():
            f.unlink()
        outside.rmdir()


def test_single_file_directive(tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    target.write_text("# top\n")
    body = "  sources:\n    - path: notes.md\n      include: ['*.md']\n"
    parsed, _ = _make_parsed(body, tmp_path)
    result = expand_sources(parsed, base_path=tmp_path)  # type: ignore[arg-type]
    assert len(result.sections) == 1
    assert result.sections[0].content.startswith("# source: notes.md")


def test_stat_failure_skips_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    src.mkdir()
    target = src / "a.py"
    target.write_text("print(1)\n", encoding="utf-8")
    body = "  sources:\n    - path: src\n      include: ['**/*.py']\n"
    parsed, _ = _make_parsed(body, tmp_path)
    real_stat = Path.stat
    seen_target = 0

    def _patched_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        nonlocal seen_target
        if path == target:
            seen_target += 1
            if seen_target >= 2:
                raise OSError("no stat")
        return real_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", _patched_stat)
    result = expand_sources(parsed, base_path=tmp_path)  # type: ignore[arg-type]
    assert result.sections == ()


def test_read_bytes_failure_skips_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    src.mkdir()
    target = src / "a.py"
    target.write_text("print(1)\n", encoding="utf-8")
    body = "  sources:\n    - path: src\n      include: ['**/*.py']\n"
    parsed, _ = _make_parsed(body, tmp_path)
    real_read_bytes = Path.read_bytes

    def _patched_read_bytes(path: Path, *args: object, **kwargs: object) -> bytes:
        if path == target:
            raise OSError("no read")
        return real_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", _patched_read_bytes)
    result = expand_sources(parsed, base_path=tmp_path)  # type: ignore[arg-type]
    assert result.sections == ()


def test_audio_transcript_unreadable_skips_audio(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "clip.wav").write_bytes(b"RIFF....fake wav")
    (corpus / "clip.txt").write_bytes(b"bad-\xff\n")
    parsed, _ = _make_parsed(
        '  sources:\n    - path: corpus\n      include: ["**/*.wav"]\n',
        tmp_path,
    )
    blob_store = BlobStore(tmp_path / "blobs")
    result = expand_sources(parsed, base_path=tmp_path, blob_store=blob_store)  # type: ignore[arg-type]
    assert result.sections == ()
    assert result.provenance[0].skipped_audio_no_transcript == 1


def test_iter_candidates_non_file_non_dir_yields_nothing(tmp_path: Path) -> None:
    assert list(_iter_candidates(tmp_path / "missing")) == []
