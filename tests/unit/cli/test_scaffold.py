"""`scaffold_train_target` — dir detection, flag→frontmatter mapping,
first-run vs resume, --rescaffold, --name disambiguation."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm.cli.scaffold import ScaffoldError, scaffold_train_target
from dlm.doc.parser import parse_file


def _default_kwargs() -> dict[str, object]:
    return {
        "base": "smollm2-135m",
        "include": (),
        "exclude": (),
        "recursive": True,
        "name": "corpus",
        "policy": "strict",
        "rescaffold": False,
    }


# ---- Dir detection + input validation --------------------------------------


def test_missing_target_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(ScaffoldError, match="does not exist"):
        scaffold_train_target(missing, **_default_kwargs())  # type: ignore[arg-type]


def test_file_target_raises(tmp_path: Path) -> None:
    f = tmp_path / "file.dlm"
    f.write_text("x")
    with pytest.raises(ScaffoldError, match="expects a directory"):
        scaffold_train_target(f, **_default_kwargs())  # type: ignore[arg-type]


# ---- First-run scaffold ----------------------------------------------------


def test_first_run_requires_base(tmp_path: Path) -> None:
    kwargs = _default_kwargs()
    kwargs["base"] = None
    with pytest.raises(ScaffoldError, match="--base"):
        scaffold_train_target(tmp_path, **kwargs)  # type: ignore[arg-type]


def test_first_run_writes_scaffolded_dlm(tmp_path: Path) -> None:
    result = scaffold_train_target(tmp_path, **_default_kwargs())  # type: ignore[arg-type]
    assert result.scaffolded is True
    assert result.dlm_path == tmp_path / ".dlm" / "corpus.dlm"
    assert result.dlm_path.is_file()
    assert len(result.dlm_id) == 26
    # Parse it to confirm the frontmatter is valid
    parsed = parse_file(result.dlm_path)
    assert parsed.frontmatter.dlm_id == result.dlm_id
    assert parsed.frontmatter.base_model == "smollm2-135m"


def test_scaffold_default_include_recursive(tmp_path: Path) -> None:
    result = scaffold_train_target(tmp_path, **_default_kwargs())  # type: ignore[arg-type]
    parsed = parse_file(result.dlm_path)
    sources = parsed.frontmatter.training.sources
    assert sources is not None
    assert sources[0].include == ("**/*",)


def test_scaffold_default_include_non_recursive(tmp_path: Path) -> None:
    kwargs = _default_kwargs()
    kwargs["recursive"] = False
    result = scaffold_train_target(tmp_path, **kwargs)  # type: ignore[arg-type]
    parsed = parse_file(result.dlm_path)
    sources = parsed.frontmatter.training.sources
    assert sources is not None
    assert sources[0].include == ("*",)


def test_scaffold_explicit_include_passed_through(tmp_path: Path) -> None:
    kwargs = _default_kwargs()
    kwargs["include"] = ("**/*.f90", "**/*.F90")
    result = scaffold_train_target(tmp_path, **kwargs)  # type: ignore[arg-type]
    parsed = parse_file(result.dlm_path)
    sources = parsed.frontmatter.training.sources
    assert sources is not None
    assert sources[0].include == ("**/*.f90", "**/*.F90")


def test_scaffold_exclude_passed_through(tmp_path: Path) -> None:
    kwargs = _default_kwargs()
    kwargs["exclude"] = ("tests/**", "**/__pycache__/**")
    result = scaffold_train_target(tmp_path, **kwargs)  # type: ignore[arg-type]
    parsed = parse_file(result.dlm_path)
    sources = parsed.frontmatter.training.sources
    assert sources is not None
    assert sources[0].exclude == ("tests/**", "**/__pycache__/**")


def test_scaffold_policy_persisted(tmp_path: Path) -> None:
    kwargs = _default_kwargs()
    kwargs["policy"] = "permissive"
    result = scaffold_train_target(tmp_path, **kwargs)  # type: ignore[arg-type]
    parsed = parse_file(result.dlm_path)
    assert parsed.frontmatter.training.sources_policy == "permissive"


# ---- Resume / reuse --------------------------------------------------------


def test_second_run_reuses_existing(tmp_path: Path) -> None:
    first = scaffold_train_target(tmp_path, **_default_kwargs())  # type: ignore[arg-type]
    # Second run with base=None (scaffold shouldn't fire)
    kwargs = _default_kwargs()
    kwargs["base"] = None
    second = scaffold_train_target(tmp_path, **kwargs)  # type: ignore[arg-type]
    assert second.scaffolded is False
    assert second.dlm_path == first.dlm_path
    assert second.dlm_id == first.dlm_id


# ---- Multi-file disambiguation ---------------------------------------------


def test_multiple_dlms_refuses_without_explicit_name(tmp_path: Path) -> None:
    # Scaffold two files with non-default names so neither matches
    # the default `corpus` on the resume attempt.
    kwargs_a = _default_kwargs()
    kwargs_a["name"] = "code"
    scaffold_train_target(tmp_path, **kwargs_a)  # type: ignore[arg-type]
    kwargs_b = _default_kwargs()
    kwargs_b["name"] = "docs"
    scaffold_train_target(tmp_path, **kwargs_b)  # type: ignore[arg-type]

    # Resume attempt with default name (corpus) and no match → refuse
    kwargs_resume = _default_kwargs()
    kwargs_resume["base"] = None
    # name defaults to "corpus" which isn't in {code, docs}
    with pytest.raises(ScaffoldError, match="multiple .dlm files"):
        scaffold_train_target(tmp_path, **kwargs_resume)  # type: ignore[arg-type]


def test_multiple_dlms_name_picks_match(tmp_path: Path) -> None:
    # Scaffold two separate .dlm files
    scaffold_train_target(tmp_path, **_default_kwargs())  # type: ignore[arg-type]
    kwargs = _default_kwargs()
    kwargs["name"] = "docs"
    second = scaffold_train_target(tmp_path, **kwargs)  # type: ignore[arg-type]

    # Resume with explicit name picks the matching file
    kwargs3 = _default_kwargs()
    kwargs3["base"] = None
    kwargs3["name"] = "docs"
    resolved = scaffold_train_target(tmp_path, **kwargs3)  # type: ignore[arg-type]
    assert resolved.dlm_path == second.dlm_path
    assert resolved.scaffolded is False


# ---- --rescaffold ----------------------------------------------------------


def test_rescaffold_preserves_dlm_id(tmp_path: Path) -> None:
    first = scaffold_train_target(tmp_path, **_default_kwargs())  # type: ignore[arg-type]
    kwargs = _default_kwargs()
    kwargs["base"] = "qwen2.5-0.5b"
    kwargs["include"] = ("**/*.md",)
    kwargs["rescaffold"] = True
    second = scaffold_train_target(tmp_path, **kwargs)  # type: ignore[arg-type]
    assert second.scaffolded is True
    assert second.dlm_id == first.dlm_id  # same ULID
    # New frontmatter
    parsed = parse_file(second.dlm_path)
    assert parsed.frontmatter.base_model == "qwen2.5-0.5b"
    assert parsed.frontmatter.training.sources[0].include == ("**/*.md",)  # type: ignore[index]


def test_rescaffold_still_needs_base(tmp_path: Path) -> None:
    scaffold_train_target(tmp_path, **_default_kwargs())  # type: ignore[arg-type]
    kwargs = _default_kwargs()
    kwargs["base"] = None
    kwargs["rescaffold"] = True
    with pytest.raises(ScaffoldError, match="--base"):
        scaffold_train_target(tmp_path, **kwargs)  # type: ignore[arg-type]
