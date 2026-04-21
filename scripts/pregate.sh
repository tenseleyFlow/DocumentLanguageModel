#!/usr/bin/env bash
# Pre-push gate — run the cheap checks CI will otherwise catch later.
#
# Mirrors the three green-required jobs of the `CI` workflow:
#   1. ruff check
#   2. mypy --strict
#   3. pytest on the unit suite (no slow marker)
#
# Plus a best-effort grep for two patterns that CI has historically
# failed on but unit tests don't catch:
#   - slow tests asserting `plan is not None` (CPU-only CI has no plan)
#   - tests pinning `dlm_version == N` for a stale N (schema bumps break these)
#
# Usage:
#   ./scripts/pregate.sh          # runs the full gate
#   ./scripts/pregate.sh --fast   # skip mypy + pytest, only ruff + pattern checks
#
# Wire as a git hook with:
#   ln -s ../../scripts/pregate.sh .git/hooks/pre-push

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

fast=0
if [[ "${1:-}" == "--fast" ]]; then
    fast=1
fi

echo "==> ruff check"
uv run ruff check .

echo "==> ruff format (check)"
uv run ruff format --check . || {
    echo "  hint: run \`uv run ruff format .\` to fix"
    exit 1
}

if [[ $fast -eq 0 ]]; then
    echo "==> mypy --strict"
    uv run mypy src

    echo "==> pytest (unit; no slow marker)"
    uv run pytest tests/unit -q --no-header
fi

# --- Pattern checks that mirror known CI foot-guns --------------------

echo "==> advisory: assert plan is not None in slow tests"
# Slow-marked tests that hard-assert the plan can fail on CPU-only CI
# unless they're upstream-guarded (e.g., `trained_store` fixture skips
# first). This is advisory: a match is worth a look, not necessarily
# a bug. Prefer `pytest.skip(...)` when the test body itself owns the
# doctor() call.
advisory_hits=$(git grep -n "assert plan is not None" -- 'tests/integration/**' 2>/dev/null || true)
if [[ -n "$advisory_hits" ]]; then
    echo "$advisory_hits" | sed 's/^/  advisory: /'
    echo "  (advisory only — confirm each has an upstream plan guard)"
fi

echo "==> stale dlm_version pin"
# Any test that hard-pins a frontmatter version exact-match should use
# >= so schema bumps don't retroactively break the test. Exact pins are
# fine in unit-test-of-migrator paths (compares against a literal).
stale=$(git grep -nE 'fm\.dlm_version ==|frontmatter\.dlm_version ==' -- 'tests/integration/**' 2>/dev/null || true)
if [[ -n "$stale" ]]; then
    echo "$stale"
    echo "  integration tests pinning dlm_version exact — prefer >= so schema bumps don't break."
    exit 1
fi

echo "==> gate clean"
