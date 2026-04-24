"""`.dlm/ignore` grammar tests — comments, blanks, negation, anchors,
directory-only, globstar, malformed lines."""

from __future__ import annotations

import logging

import pytest

from dlm.directives.ignore_parser import (
    IgnoreRule,
    matches,
    parse_ignore_file,
)

# ---- parse_ignore_file -----------------------------------------------------


def test_parse_empty() -> None:
    assert parse_ignore_file("") == ()
    assert parse_ignore_file("   \n\n\t\n") == ()


def test_parse_skips_comments_and_blanks() -> None:
    text = "# top comment\n\n  # indented comment\n*.log\n"
    rules = parse_ignore_file(text)
    assert len(rules) == 1
    assert rules[0].pattern == "*.log"


def test_parse_negation() -> None:
    rules = parse_ignore_file("!keep.txt\n")
    assert rules[0].negate is True
    assert rules[0].pattern == "keep.txt"


def test_parse_anchored() -> None:
    rules = parse_ignore_file("/scripts/local.sh\n")
    assert rules[0].anchored is True
    assert rules[0].pattern == "scripts/local.sh"


def test_parse_directory_only() -> None:
    rules = parse_ignore_file("build/\n")
    assert rules[0].directory_only is True
    assert rules[0].pattern == "build"


def test_parse_combined_flags() -> None:
    rules = parse_ignore_file("!/scripts/build/\n")
    assert rules[0].negate is True
    assert rules[0].anchored is True
    assert rules[0].directory_only is True
    assert rules[0].pattern == "scripts/build"


def test_parse_bare_bang_skipped(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="dlm.directives.ignore_parser")
    rules = parse_ignore_file("!\n")
    assert rules == ()
    assert any("bare '!'" in rec.message for rec in caplog.records)


def test_parse_bare_slash_skipped(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="dlm.directives.ignore_parser")
    rules = parse_ignore_file("/\n")
    assert rules == ()
    assert any("bare '/'" in rec.message for rec in caplog.records)


def test_parse_pattern_reduced_to_empty_skipped(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="dlm.directives.ignore_parser")
    rules = parse_ignore_file("//\n")
    assert rules == ()
    assert any("pattern reduced to empty" in rec.message for rec in caplog.records)


# ---- matches ---------------------------------------------------------------


def _rule(
    pattern: str,
    *,
    anchored: bool = False,
    directory_only: bool = False,
    negate: bool = False,
) -> IgnoreRule:
    return IgnoreRule(
        pattern=pattern,
        anchored=anchored,
        directory_only=directory_only,
        negate=negate,
    )


def test_matches_unanchored_globstar() -> None:
    rule = _rule("**/__pycache__/**")
    assert matches(rule, "src/__pycache__/foo.pyc", is_dir=False)
    assert matches(rule, "a/b/__pycache__/c/d.pyc", is_dir=False)


def test_matches_unanchored_basename() -> None:
    rule = _rule("*.log")
    assert matches(rule, "debug.log", is_dir=False)
    assert matches(rule, "a/b/debug.log", is_dir=False)


def test_matches_anchored_only_at_root() -> None:
    rule = _rule("scripts/local.sh", anchored=True)
    assert matches(rule, "scripts/local.sh", is_dir=False)
    assert not matches(rule, "a/scripts/local.sh", is_dir=False)


def test_matches_directory_only_flags_subtree() -> None:
    rule = _rule("build", directory_only=True)
    # Any path under a "build" directory matches via ancestor-component
    assert matches(rule, "build/x.txt", is_dir=False)
    assert matches(rule, "a/build/x/y.txt", is_dir=False)
    # No "build" component → no match
    assert not matches(rule, "src/main.py", is_dir=False)


def test_matches_directory_only_with_file_not_dir() -> None:
    # A file *named* "build" without the directory flag matches; with
    # directory_only=True, a plain file at that path does NOT match
    # (only directories and their descendants do).
    rule = _rule("build", directory_only=True)
    # A file literally named "build" at root: no ancestor "build/" path.
    assert not matches(rule, "build", is_dir=False)
    # But matches when is_dir=True:
    assert matches(rule, "build", is_dir=True)


def test_matches_single_star_does_not_cross_slash() -> None:
    rule = _rule("*.py")
    assert matches(rule, "foo.py", is_dir=False)
    # `*.py` anchored (via unanchored suffix-matching) should match the
    # last component of a/b.py too, since we try suffixes.
    assert matches(rule, "a/b.py", is_dir=False)


def test_matches_question_mark_single_char() -> None:
    rule = _rule("foo?.txt")
    assert matches(rule, "foo1.txt", is_dir=False)
    assert not matches(rule, "foo12.txt", is_dir=False)


def test_matches_globstar_prefix() -> None:
    rule = _rule("tests/**")
    assert matches(rule, "tests/a.py", is_dir=False)
    assert matches(rule, "tests/a/b/c.py", is_dir=False)
    assert not matches(rule, "src/a.py", is_dir=False)
