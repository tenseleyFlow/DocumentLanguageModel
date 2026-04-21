"""Path confinement, binary sniff, glob-to-regex, and enumeration."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from dlm.directives.errors import DirectivePolicyError
from dlm.directives.safety import (
    _compile_glob,
    confine_path,
    enumerate_matching_files,
    is_probably_binary,
)

# ---- glob compiler ----------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("*.py", "foo.py", True),
        ("*.py", "foo.txt", False),
        ("*.py", "sub/foo.py", False),  # single `*` doesn't cross `/`
        ("**/*.py", "foo.py", True),
        ("**/*.py", "a/b/foo.py", True),
        ("tests/**", "tests/a.py", True),
        ("tests/**", "tests/a/b.py", True),
        ("tests/**", "src/a.py", False),
        ("src/**/*.rs", "src/x.rs", True),
        ("src/**/*.rs", "src/a/b.rs", True),
        ("foo?.md", "foo1.md", True),
        ("foo?.md", "foo12.md", False),
    ],
)
def test_glob_compiler(pattern: str, path: str, expected: bool) -> None:
    assert bool(_compile_glob(pattern).fullmatch(path)) is expected


# ---- confine_path -----------------------------------------------------------


def test_confine_strict_accepts_child(tmp_path: Path) -> None:
    child = tmp_path / "a" / "b"
    child.mkdir(parents=True)
    resolved = confine_path(child, tmp_path, strict=True)
    assert resolved == child.resolve()


def test_confine_strict_rejects_sibling(tmp_path: Path) -> None:
    sibling = tmp_path.parent / "other"
    with pytest.raises(DirectivePolicyError):
        confine_path(sibling, tmp_path, strict=True)


def test_confine_permissive_allows_external(tmp_path: Path) -> None:
    external = tmp_path.parent
    # doesn't raise, returns resolved path
    result = confine_path(external, tmp_path, strict=False)
    assert result == external.resolve()


def test_confine_strict_symlink_escape_refused(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    try:
        link = tmp_path / "escape"
        link.symlink_to(outside)
        with pytest.raises(DirectivePolicyError):
            confine_path(link, tmp_path, strict=True)
    finally:
        if outside.exists():
            outside.rmdir()


def test_confine_permissive_symlink_escape_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    outside = tmp_path.parent / "outside_log"
    outside.mkdir(exist_ok=True)
    try:
        link = tmp_path / "escape"
        link.symlink_to(outside)
        caplog.set_level(logging.WARNING, logger="dlm.directives.safety")
        confine_path(link, tmp_path, strict=False)
        assert any("symlink" in rec.message for rec in caplog.records)
    finally:
        if outside.exists():
            outside.rmdir()


def test_confine_expands_tilde(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    # ~/foo resolves under fake home
    resolved = confine_path(Path("~/foo"), fake_home, strict=True)
    assert resolved == (fake_home / "foo").resolve()


# ---- binary sniff ----------------------------------------------------------


def test_is_probably_binary_finds_nul() -> None:
    assert is_probably_binary(b"hello\x00world") is True


def test_is_probably_binary_plain_text() -> None:
    assert is_probably_binary(b"hello world\nthis is text\n") is False


def test_is_probably_binary_nul_past_sample() -> None:
    # NUL beyond the first 1 KiB → not flagged
    data = b"A" * 2048 + b"\x00"
    assert is_probably_binary(data) is False


# ---- enumerate_matching_files ---------------------------------------------


def test_enumerate_is_deterministic(tmp_path: Path) -> None:
    for name in ("z.py", "a.py", "m.py"):
        (tmp_path / name).write_text("x")
    got = list(enumerate_matching_files(tmp_path, include=("**/*.py",), exclude=()))
    assert [p.name for p in got] == ["a.py", "m.py", "z.py"]


def test_enumerate_exclude_wins(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("x")
    (tmp_path / "skip.py").write_text("x")
    got = list(
        enumerate_matching_files(
            tmp_path, include=("**/*.py",), exclude=("skip.py",)
        )
    )
    assert [p.name for p in got] == ["keep.py"]


def test_enumerate_nested(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "nested.py").write_text("x")
    (tmp_path / "top.py").write_text("x")
    got = list(enumerate_matching_files(tmp_path, include=("**/*.py",), exclude=()))
    rels = [p.relative_to(tmp_path).as_posix() for p in got]
    assert rels == ["a/nested.py", "top.py"]


def test_enumerate_single_file(tmp_path: Path) -> None:
    target = tmp_path / "one.md"
    target.write_text("x")
    got = list(enumerate_matching_files(target, include=("*.md",), exclude=()))
    assert got == [target]


def test_enumerate_missing_root_yields_nothing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    got = list(enumerate_matching_files(missing, include=("**/*",), exclude=()))
    assert got == []
