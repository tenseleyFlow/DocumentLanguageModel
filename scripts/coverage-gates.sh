#!/usr/bin/env bash
# Mirror the Ubuntu per-package coverage gates from `.github/workflows/ci.yml`.
#
# Usage:
#   ./scripts/coverage-gates.sh
#   ./scripts/coverage-gates.sh --list
#   ./scripts/coverage-gates.sh --only lock --only export
#
# Best match with CI:
#   uv sync --all-extras --dev

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

show_list=0
selected=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)
            show_list=1
            ;;
        --only)
            shift
            if [[ $# -eq 0 ]]; then
                echo "error: --only requires a gate name" >&2
                exit 2
            fi
            selected+=("$1")
            ;;
        *)
            echo "usage: ./scripts/coverage-gates.sh [--list] [--only <gate>]..." >&2
            exit 2
            ;;
    esac
    shift
done

gates=(
    "doc|tests/unit/doc|src/dlm/doc"
    "store|tests/unit/store|src/dlm/store"
    "hardware|tests/unit/hardware|src/dlm/hardware"
    "base_models|tests/unit/base_models|src/dlm/base_models"
    "data|tests/unit/data|src/dlm/data"
    "replay|tests/unit/replay|src/dlm/replay"
    "train|tests/unit/train|src/dlm/train"
    "train_preference|tests/unit/train/preference|src/dlm/train/preference"
    "eval|tests/unit/eval|src/dlm/eval"
    "inference|tests/unit/inference|src/dlm/inference"
    "export|tests/unit/export|src/dlm/export"
    "export_ollama|tests/unit/export/ollama|src/dlm/export/ollama"
    "cli_reporter|tests/unit/cli|dlm.cli.reporter"
    "io_ulid|tests/unit/test_io_ulid.py|dlm.io.ulid"
    "pack|tests/unit/pack tests/integration/pack|src/dlm/pack"
    "lock|tests/unit/lock|src/dlm/lock"
)

gate_selected() {
    local name="$1"
    if [[ ${#selected[@]} -eq 0 ]]; then
        return 0
    fi
    local needle
    for needle in "${selected[@]}"; do
        if [[ "$needle" == "$name" ]]; then
            return 0
        fi
    done
    return 1
}

if [[ $show_list -eq 1 ]]; then
    for gate in "${gates[@]}"; do
        IFS="|" read -r name _tests _cov <<< "$gate"
        printf '%s\n' "$name"
    done
    exit 0
fi

ran_any=0
for gate in "${gates[@]}"; do
    IFS="|" read -r name tests cov <<< "$gate"
    if ! gate_selected "$name"; then
        continue
    fi
    ran_any=1
    echo "==> Coverage gate: $name ($cov)"
    # shellcheck disable=SC2206
    tests_arr=($tests)
    uv run pytest \
        "${tests_arr[@]}" \
        --cov="$cov" \
        --cov-report=term-missing \
        --cov-fail-under=100 \
        -q
    echo
done

if [[ $ran_any -eq 0 ]]; then
    echo "error: no matching coverage gate names selected" >&2
    exit 2
fi

echo "==> coverage gates clean"
