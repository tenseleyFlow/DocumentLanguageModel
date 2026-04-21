"""Resolve per-file effective config from parent directive + discovered `.dlm/`.

Decision order (applied top-down; last match wins within the exclude
bucket, matching `.gitignore`'s semantics exactly):

1. Parent directive's `include` / `exclude` (outermost shell)
2. Default-exclude set, unless the nearest `training.yaml` sets
   `exclude_defaults: false`
3. For each ancestor anchor (shallowest → deepest):
   a. `training.yaml.exclude` patterns
   b. `.dlm/ignore` rules (including `!negation`)
   Later rules can un-exclude earlier ones via negation.
4. Include resolution: nearest-ancestor `training.yaml.include` if
   non-empty, else parent-directive include.
5. Metadata: shallow-to-deep merge of every `training.yaml.metadata`
   on the ancestor path; deeper keys overwrite shallower on collision.

Returns `None` from `is_included` when the file is either not matched
by any include pattern OR matches a final exclude — the caller treats
None as "skip this file".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dlm.directives.defaults import DEFAULT_EXCLUDES
from dlm.directives.discovery import DiscoveredConfig
from dlm.directives.ignore_parser import matches as ignore_matches
from dlm.directives.safety import _compile_glob
from dlm.doc.schema import SourceDirective


@dataclass(frozen=True)
class EffectiveConfig:
    """Resolved rules for one file within a directive walk.

    `included` collapses the include/exclude verdict: True means the
    file should be ingested, False means skip. `tags` is the merged
    metadata dict to flow onto the synthesized Section.
    """

    included: bool
    tags: Mapping[str, str]


def ancestors_of(
    file_path: Path, discovered: tuple[DiscoveredConfig, ...]
) -> tuple[DiscoveredConfig, ...]:
    """Return DiscoveredConfigs whose anchor is an ancestor of file_path,
    sorted shallowest → deepest. Includes the direct-parent anchor."""
    abs_file = file_path.resolve()
    result = [d for d in discovered if _is_ancestor(d.anchor.resolve(), abs_file)]
    result.sort(key=lambda d: len(d.anchor.as_posix()))
    return tuple(result)


def _is_ancestor(anchor: Path, file_path: Path) -> bool:
    try:
        file_path.relative_to(anchor)
    except ValueError:
        return False
    return True


def effective_config_for(
    file_path: Path,
    *,
    source_root: Path,
    discovered: tuple[DiscoveredConfig, ...],
    parent_directive: SourceDirective,
    is_dir: bool = False,
) -> EffectiveConfig:
    """Resolve the applicable rules for `file_path`.

    `file_path` is the candidate file (already resolved from the
    filesystem walk). `source_root` is the directive's resolved
    root — used to compute paths relative to the include/exclude
    glob space. `parent_directive` is the frontmatter directive
    driving this walk; its `include`/`exclude` act as the outer shell.
    """
    ancestors = ancestors_of(file_path, discovered)
    rel_to_root = _relpath(file_path, source_root)

    # --- Include resolution -------------------------------------------
    # Nearest-ancestor training.yaml with non-empty include wins;
    # otherwise fall back to the parent directive's include.
    effective_include: tuple[str, ...] = parent_directive.include
    for d in reversed(ancestors):
        if d.config is not None and d.config.include:
            effective_include = d.config.include
            break
    included_by_positive = _matches_any(rel_to_root, effective_include)

    # --- Exclude resolution (last-match-wins across all sources) ------
    # Track a single boolean; flip it as we encounter matches. A
    # `!negation` match flips it back to included. Final value after
    # walking all sources is the verdict.
    excluded = False

    # Layer 1: parent directive's explicit excludes.
    if _matches_any(rel_to_root, parent_directive.exclude):
        excluded = True

    # Layer 2: default excludes (unless opted out by nearest training.yaml).
    apply_defaults = True
    for d in reversed(ancestors):
        if d.config is not None:
            apply_defaults = d.config.exclude_defaults
            break
    if apply_defaults and _matches_any(rel_to_root, DEFAULT_EXCLUDES):
        excluded = True

    # Layer 3: per-anchor training.yaml excludes, shallowest → deepest.
    for d in ancestors:
        if d.config is not None:
            rel_to_anchor = _relpath(file_path, d.anchor)
            if _matches_any(rel_to_anchor, d.config.exclude):
                excluded = True

    # Layer 4: per-anchor .dlm/ignore rules, shallowest → deepest.
    # Within a single file's rules, last match wins (including negation).
    for d in ancestors:
        if not d.ignore_rules:
            continue
        rel_to_anchor = _relpath(file_path, d.anchor)
        for rule in d.ignore_rules:
            if ignore_matches(rule, rel_to_anchor, is_dir=is_dir):
                excluded = not rule.negate

    # --- Metadata merge (shallow-to-deep) -----------------------------
    tags: dict[str, str] = {}
    for d in ancestors:
        if d.config is not None and d.config.metadata:
            tags.update(d.config.metadata)

    return EffectiveConfig(
        included=(included_by_positive and not excluded),
        tags=tags,
    )


def _matches_any(rel_path: str, patterns: tuple[str, ...]) -> bool:
    """Positive match against a tuple of globs using the shared
    `_compile_glob`. Empty tuple → False (callers handle the include
    case where empty means "inherit from parent")."""
    return any(_compile_glob(p).fullmatch(rel_path) is not None for p in patterns)


def _relpath(file_path: Path, anchor: Path) -> str:
    """POSIX-form relative path. Handles two edge cases:

    - Single-file directive: `file_path == anchor`. `relative_to`
      returns `.` in that case, but the include/exclude globs expect
      the filename, so we return `file_path.name`.
    - Non-ancestor anchor: defensive fallback to `file_path.name`.
      Shouldn't happen on the hot path, but returning the name keeps
      basename-style globs (`*.py`) working.
    """
    resolved_file = file_path.resolve()
    resolved_anchor = anchor.resolve()
    if resolved_file == resolved_anchor:
        return file_path.name
    try:
        return resolved_file.relative_to(resolved_anchor).as_posix()
    except ValueError:
        return file_path.name
