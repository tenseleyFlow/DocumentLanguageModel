"""Merge resolver — parent directive + discovered `.dlm/` → effective verdict."""

from __future__ import annotations

from pathlib import Path

from dlm.directives.discovery import discover_configs
from dlm.directives.merge import effective_config_for
from dlm.doc.schema import SourceDirective


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _directive(
    root: Path,
    *,
    include: tuple[str, ...] = ("**/*",),
    exclude: tuple[str, ...] = (),
) -> SourceDirective:
    return SourceDirective(path=str(root), include=include, exclude=exclude)


# ---- Core include/exclude resolution ---------------------------------------


def test_directive_include_wins_without_training_yaml(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "main.py", "x")
    _write(tmp_path / "src" / "readme.md", "x")
    directive = _directive(tmp_path, include=("**/*.py",))
    configs = discover_configs(tmp_path)
    eff_py = effective_config_for(
        tmp_path / "src" / "main.py",
        source_root=tmp_path,
        discovered=configs,
        parent_directive=directive,
    )
    eff_md = effective_config_for(
        tmp_path / "src" / "readme.md",
        source_root=tmp_path,
        discovered=configs,
        parent_directive=directive,
    )
    assert eff_py.included is True
    assert eff_md.included is False


def test_training_yaml_include_overrides_directive(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "main.py", "x")
    _write(tmp_path / "src" / "doc.md", "x")
    _write(
        tmp_path / ".dlm" / "training.yaml",
        "dlm_training_version: 1\ninclude: ['**/*.md']\n",
    )
    directive = _directive(tmp_path, include=("**/*.py",))
    configs = discover_configs(tmp_path)
    eff_py = effective_config_for(
        tmp_path / "src" / "main.py",
        source_root=tmp_path,
        discovered=configs,
        parent_directive=directive,
    )
    eff_md = effective_config_for(
        tmp_path / "src" / "doc.md",
        source_root=tmp_path,
        discovered=configs,
        parent_directive=directive,
    )
    # training.yaml include: **/*.md → py excluded, md included
    assert eff_py.included is False
    assert eff_md.included is True


def test_training_yaml_empty_include_inherits_parent(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "main.py", "x")
    _write(
        tmp_path / ".dlm" / "training.yaml",
        "dlm_training_version: 1\nexclude: ['**/*.min.*']\n",
    )
    directive = _directive(tmp_path, include=("**/*.py",))
    configs = discover_configs(tmp_path)
    eff = effective_config_for(
        tmp_path / "src" / "main.py",
        source_root=tmp_path,
        discovered=configs,
        parent_directive=directive,
    )
    assert eff.included is True


def test_training_yaml_exclude_blocks_file(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "main.py", "x")
    _write(tmp_path / "src" / "test_main.py", "x")
    _write(
        tmp_path / ".dlm" / "training.yaml",
        "dlm_training_version: 1\nexclude: ['**/test_*.py']\n",
    )
    directive = _directive(tmp_path, include=("**/*.py",))
    configs = discover_configs(tmp_path)
    assert (
        effective_config_for(
            tmp_path / "src" / "main.py",
            source_root=tmp_path,
            discovered=configs,
            parent_directive=directive,
        ).included
        is True
    )
    assert (
        effective_config_for(
            tmp_path / "src" / "test_main.py",
            source_root=tmp_path,
            discovered=configs,
            parent_directive=directive,
        ).included
        is False
    )


# ---- .dlm/ignore negation --------------------------------------------------


def test_ignore_rule_excludes_file(tmp_path: Path) -> None:
    _write(tmp_path / "debug.log", "x")
    _write(tmp_path / ".dlm" / "ignore", "*.log\n")
    directive = _directive(tmp_path)
    configs = discover_configs(tmp_path)
    eff = effective_config_for(
        tmp_path / "debug.log",
        source_root=tmp_path,
        discovered=configs,
        parent_directive=directive,
    )
    assert eff.included is False


def test_ignore_negation_re_includes_file(tmp_path: Path) -> None:
    _write(tmp_path / "debug.log", "x")
    _write(tmp_path / "special.log", "x")
    _write(tmp_path / ".dlm" / "ignore", "*.log\n!special.log\n")
    directive = _directive(tmp_path)
    configs = discover_configs(tmp_path)
    assert (
        effective_config_for(
            tmp_path / "debug.log",
            source_root=tmp_path,
            discovered=configs,
            parent_directive=directive,
        ).included
        is False
    )
    assert (
        effective_config_for(
            tmp_path / "special.log",
            source_root=tmp_path,
            discovered=configs,
            parent_directive=directive,
        ).included
        is True
    )


def test_deeper_ignore_negation_unblocks_parent_exclude(tmp_path: Path) -> None:
    """Nearest-ancestor last-match-wins: a deeper .dlm/ignore with
    `!path/specific.md` re-includes a file that a shallower rule
    excluded."""
    _write(tmp_path / "docs" / "guide.md", "x")
    _write(tmp_path / ".dlm" / "ignore", "*.md\n")
    _write(tmp_path / "docs" / ".dlm" / "ignore", "!guide.md\n")
    directive = _directive(tmp_path)
    configs = discover_configs(tmp_path)
    eff = effective_config_for(
        tmp_path / "docs" / "guide.md",
        source_root=tmp_path,
        discovered=configs,
        parent_directive=directive,
    )
    assert eff.included is True


# ---- Default excludes ------------------------------------------------------


def test_default_excludes_apply_by_default(tmp_path: Path) -> None:
    _write(tmp_path / ".git" / "HEAD", "x")
    _write(tmp_path / "src" / "main.py", "x")
    directive = _directive(tmp_path)
    configs = discover_configs(tmp_path)
    assert (
        effective_config_for(
            tmp_path / ".git" / "HEAD",
            source_root=tmp_path,
            discovered=configs,
            parent_directive=directive,
        ).included
        is False
    )
    assert (
        effective_config_for(
            tmp_path / "src" / "main.py",
            source_root=tmp_path,
            discovered=configs,
            parent_directive=directive,
        ).included
        is True
    )


def test_exclude_defaults_false_disables_default_set(tmp_path: Path) -> None:
    _write(tmp_path / ".git" / "HEAD", "x")
    _write(
        tmp_path / ".dlm" / "training.yaml",
        "dlm_training_version: 1\nexclude_defaults: false\n",
    )
    directive = _directive(tmp_path)
    configs = discover_configs(tmp_path)
    eff = effective_config_for(
        tmp_path / ".git" / "HEAD",
        source_root=tmp_path,
        discovered=configs,
        parent_directive=directive,
    )
    # With defaults off, .git/ files are included by `**/*`.
    assert eff.included is True


# ---- Metadata merging ------------------------------------------------------


def test_metadata_shallow_to_deep_merge(tmp_path: Path) -> None:
    _write(tmp_path / "vendor" / "dep.py", "x")
    _write(
        tmp_path / ".dlm" / "training.yaml",
        "dlm_training_version: 1\nmetadata:\n  language: python\n  domain: main\n",
    )
    _write(
        tmp_path / "vendor" / ".dlm" / "training.yaml",
        "dlm_training_version: 1\nmetadata:\n  domain: vendor_override\n  source: third_party\n",
    )
    directive = _directive(tmp_path)
    configs = discover_configs(tmp_path)
    eff = effective_config_for(
        tmp_path / "vendor" / "dep.py",
        source_root=tmp_path,
        discovered=configs,
        parent_directive=directive,
    )
    assert eff.tags == {
        "language": "python",  # from shallower
        "domain": "vendor_override",  # deeper overrides shallower
        "source": "third_party",  # from deeper only
    }


def test_metadata_empty_when_no_training_yaml(tmp_path: Path) -> None:
    _write(tmp_path / "main.py", "x")
    _write(tmp_path / ".dlm" / "ignore", "*.log\n")
    directive = _directive(tmp_path)
    configs = discover_configs(tmp_path)
    eff = effective_config_for(
        tmp_path / "main.py",
        source_root=tmp_path,
        discovered=configs,
        parent_directive=directive,
    )
    assert dict(eff.tags) == {}
