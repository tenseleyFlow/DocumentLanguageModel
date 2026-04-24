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
#   ./scripts/pregate.sh              # runs the full gate
#   ./scripts/pregate.sh --fast       # skip mypy + pytest, only ruff + pattern checks
#   ./scripts/pregate.sh --coverage   # also mirror the Ubuntu package coverage gates
#
# Wire as a git hook with:
#   ln -s ../../scripts/pregate.sh .git/hooks/pre-push

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

fast=0
coverage=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast)
            fast=1
            ;;
        --coverage)
            coverage=1
            ;;
        *)
            echo "usage: ./scripts/pregate.sh [--fast] [--coverage]" >&2
            exit 2
            ;;
    esac
    shift
done

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

if [[ $coverage -eq 1 ]]; then
    echo "==> coverage gates (Ubuntu mirror)"
    ./scripts/coverage-gates.sh
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

echo "==> modality scatter outside dlm.modality"
# Sprint 38 B8.6: every `spec.modality == "..."` comparison lives in
# src/dlm/modality/. Callers elsewhere go through predicate flags
# (accepts_images, accepts_audio, requires_processor) or
# modality_for(spec).dispatch_export() / .load_processor(). New
# scatter means the modality abstraction is leaking again — refuse
# the push and route the new code into the dispatch package instead.
scatter=$(git grep -nE 'spec\.modality ==' -- 'src/dlm/**' 2>/dev/null | grep -v "src/dlm/modality/" || true)
if [[ -n "$scatter" ]]; then
    echo "$scatter"
    echo "  modality scatter outside src/dlm/modality/ — route through modality_for(spec)."
    exit 1
fi

echo "==> new sprint jargon in src/dlm"
# Sprint 39 M4: planning terms like `Sprint 23` or `audit-08` should
# not leak into newly added product/runtime strings under src/dlm.
# Compare the current tree against the upstream merge-base when one
# exists, so committed fixes in the working tree override older
# branch-local additions that have not been pushed yet.
collect_src_dlm_diff() {
    local upstream
    upstream=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || true)
    if [[ -n "$upstream" ]]; then
        local merge_base
        merge_base=$(git merge-base "$upstream" HEAD 2>/dev/null || true)
        if [[ -n "$merge_base" ]]; then
            git diff --unified=0 --no-color "$merge_base" -- 'src/dlm/**' 2>/dev/null || true
            return
        fi
    fi

    git diff --unified=0 --no-color HEAD -- 'src/dlm/**' 2>/dev/null || true
}

jargon_hits=$(
    collect_src_dlm_diff | awk '
        /^diff --git / {
            file = $4
            sub("^b/", "", file)
            next
        }
        /^\+\+\+ b\// {
            file = substr($0, 7)
            next
        }
        /^\+[^+]/ && ($0 ~ /Sprint [0-9]+/ || $0 ~ /audit-[0-9]+/) {
            print file ":" substr($0, 2)
        }
    ' | sort -u
)
if [[ -n "$jargon_hits" ]]; then
    echo "$jargon_hits"
    echo "  new Sprint/audit jargon leaked into src/dlm/ — translate it into product or operator language."
    exit 1
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
