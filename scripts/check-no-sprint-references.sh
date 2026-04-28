#!/usr/bin/env bash
# Refuse new "Sprint N" / "audit-N" references in src/dlm/.
#
# Sprint IDs and audit IDs belong in `.docs/sprints/` and
# `.docs/audits/`, not in source code. Code that mentions a sprint by
# number rots the moment that sprint's scope shifts. We caught and
# swept this multiple times across audits 09 / 11 / 12 — this hook
# enforces the norm so the next sweep doesn't have to be mechanical.
#
# Runs against the staged diff (pre-commit) — only NEW additions are
# checked. Existing lines are tolerated (the sweep already cleaned
# them up; future sweeps catch any drift the diff misses).
#
# To run manually:
#     ./scripts/check-no-sprint-references.sh
# To check only what's staged:
#     ./scripts/check-no-sprint-references.sh --staged

set -euo pipefail

PATTERN='([Ss]print[[:space:]]+[0-9]+|[Aa]udit[[:space:]]*-?[[:space:]]*[0-9]+)'

if [[ "${1:-}" == "--staged" ]]; then
    diff="$(git diff --cached --no-color --unified=0 -- 'src/dlm/*.py')"
    # Only inspect added lines (start with '+', not '+++ ').
    hits="$(printf '%s\n' "$diff" \
        | grep -E '^\+[^+]' \
        | grep -E "$PATTERN" || true)"
    if [[ -n "$hits" ]]; then
        echo "Refusing commit: new sprint/audit references in src/dlm/." >&2
        echo "Move them into .docs/sprints/ or .docs/audits/." >&2
        echo >&2
        printf '%s\n' "$hits" >&2
        exit 1
    fi
    exit 0
fi

# Full-tree audit (use this in CI / on demand).
hits="$(git grep -nE "$PATTERN" -- 'src/dlm/*.py' || true)"
if [[ -n "$hits" ]]; then
    echo "Sprint/audit references found in src/dlm/:" >&2
    printf '%s\n' "$hits" >&2
    exit 1
fi
echo "src/dlm/ clean — no sprint/audit references."
