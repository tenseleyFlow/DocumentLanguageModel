"""`.dlm/ignore` — gitignore-subset parser.

Drive-by exclusions for users who don't want full `.dlm/training.yaml`
config. Drop a three-line `.dlm/ignore`, done. Grammar is a strict
subset of `.gitignore`:

Supported:
- `#` starts a comment; blank lines skipped
- `**` globstar: matches zero-or-more path components
- `!pattern` negates — re-includes an otherwise-excluded file
- Trailing `/` matches directories only
- Leading `/` anchors to the `.dlm/`'s parent (not any ancestor)

NOT supported (document explicitly in docs/format/dlm-ignore.md):
- Backslash escapes
- Character classes `[abc]`
- Whitespace-escape with backslash

Malformed lines log one WARN and are dropped — ignore files are a
drive-by UX, a typo shouldn't kill the training run.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class IgnoreRule:
    """One line from a `.dlm/ignore` file, pre-parsed.

    `pattern` is the raw glob (minus leading `!` and trailing `/`).
    `anchored` = leading `/` in source → match only at anchor root.
    `directory_only` = trailing `/` → match only directories.
    `negate` = leading `!` → re-include a previously-excluded path.
    """

    pattern: str
    anchored: bool
    directory_only: bool
    negate: bool


def parse_ignore_file(text: str) -> tuple[IgnoreRule, ...]:
    """Parse a `.dlm/ignore` body into rule tuples.

    Skips blanks and comments. Malformed lines log + drop (never raise)
    — the whole point of `.dlm/ignore` is low-ceremony, so a syntax
    error in one line shouldn't fail the walk.
    """
    rules: list[IgnoreRule] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        negate = line.startswith("!")
        if negate:
            line = line[1:]
        if not line:
            _LOG.warning("dlm/ignore:%d: bare '!' with no pattern; skipping", lineno)
            continue

        anchored = line.startswith("/")
        if anchored:
            line = line[1:]
        if not line:
            _LOG.warning("dlm/ignore:%d: bare '/' with no pattern; skipping", lineno)
            continue

        directory_only = line.endswith("/")
        if directory_only:
            line = line[:-1]
        if not line:
            _LOG.warning("dlm/ignore:%d: pattern reduced to empty; skipping", lineno)
            continue

        rules.append(
            IgnoreRule(
                pattern=line,
                anchored=anchored,
                directory_only=directory_only,
                negate=negate,
            )
        )
    return tuple(rules)


def matches(rule: IgnoreRule, relpath: str, *, is_dir: bool) -> bool:
    """Return True if `relpath` matches `rule`.

    `relpath` is the POSIX-form path relative to the anchor directory
    (the `.dlm/`'s parent). `is_dir` is used to honor `directory_only`.

    Semantics follow `.gitignore`:
    - `anchored=True` matches only when the pattern matches the path
      from position 0 (`src/foo.py` matches `/src/**`, not `vendor/src/foo.py`).
    - `anchored=False` matches if any directory-anchored suffix matches
      (`node_modules/**` matches `a/b/node_modules/c.js`).
    - `directory_only=True` only counts full-path matches when `is_dir`
      is True. For files, the rule still matches if any ancestor path
      component matches the pattern (so `build/` flags files under
      any directory named `build`).
    """
    regex = _compile_ignore_pattern(rule.pattern)

    if rule.anchored:
        candidate_paths = [relpath]
    else:
        # Unanchored: try every "from this component onward" suffix.
        parts = relpath.split("/")
        candidate_paths = ["/".join(parts[i:]) for i in range(len(parts))]

    full_path_match = any(regex.fullmatch(c) is not None for c in candidate_paths)

    if full_path_match:
        # Directory-only: a full match counts only if relpath IS a
        # directory. A file literally named `build` (relpath="build",
        # is_dir=False) does NOT match `build/` on its own path;
        # it only matches via the ancestor-component check below.
        if rule.directory_only and not is_dir:
            pass
        else:
            return True

    # Directory-only + unanchored: a file is also covered when any
    # ancestor directory in its path matches the pattern.
    if rule.directory_only and not rule.anchored:
        parts = relpath.split("/")
        # For files: only check ancestors (skip the final component).
        # For dirs: check all components including the last.
        last_index = len(parts) - (0 if is_dir else 1)
        for i in range(last_index):
            if regex.fullmatch(parts[i]) is not None:
                return True

    return False


def _compile_ignore_pattern(pattern: str) -> re.Pattern[str]:
    """Translate the ignore-file glob grammar to a regex.

    Shares the `**` / `*` / `?` semantics with
    `dlm.directives.safety._compile_glob`, which the directive
    include/exclude filters already use. Keeping the logic in sync
    matters — a user who writes `tests/**` in `training.yaml.exclude`
    should get the same match set as writing it in `.dlm/ignore`.
    """
    i = 0
    n = len(pattern)
    out: list[str] = ["^"]
    while i < n:
        c = pattern[i]
        if c == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                out.append(".*")
                i += 2
                if i < n and pattern[i] == "/":
                    i += 1
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    out.append("$")
    return re.compile("".join(out))
