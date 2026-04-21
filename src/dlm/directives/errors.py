"""Typed errors for `dlm.directives` path resolution + expansion.

`DirectiveError` is the package-wide base. Callers can catch it to
cover every directive failure in one handler (the CLI reporter does
this). Concrete subclasses carry enough context for the error message
to point at the offending directive in the frontmatter.
"""

from __future__ import annotations

from pathlib import Path


class DirectiveError(Exception):
    """Base for every `dlm.directives` failure."""


class DirectivePolicyError(DirectiveError):
    """A directive's resolved path escapes the strict-policy root.

    Only raised when `training.sources_policy == "strict"`. Under
    permissive policy we only log a warning for symlink escapes and
    proceed — users opted into that mode by explicitly choosing it.
    """

    def __init__(self, resolved: Path, root: Path) -> None:
        self.resolved = resolved
        self.root = root
        super().__init__(
            f"directive path {resolved!s} escapes strict-policy root {root!s}; "
            "either move the path under the .dlm's parent directory or "
            "set `training.sources_policy: permissive` to allow external paths"
        )


class DirectivePathError(DirectiveError):
    """A directive points at a path that doesn't exist or isn't readable."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"directive path {path!s}: {reason}")
