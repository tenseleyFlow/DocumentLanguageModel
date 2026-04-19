#!/usr/bin/env python3
"""Regenerate Sprint 12.6's per-dialect chat-template token-count goldens.

For every registered Go-template dialect that has a representative
base spec in the registry, this script:

1. Loads the HF tokenizer for the representative base.
2. Walks the shared scenario matrix (`tests/golden/chat-templates/
   scenarios.json`).
3. For each scenario, renders via `apply_chat_template(...,
   add_generation_prompt=True, tokenize=True)` and records the
   token count.
4. Writes `tests/golden/chat-templates/<dialect>/<scenario>.json`.

The emitted files are byte-identical across runs against the same
pinned base revision + transformers version — they power the
Sprint 12.6 closed-loop check (Ollama Go template's
`prompt_eval_count` must equal these HF counts).

Usage:
    uv run python scripts/refresh-chat-template-goldens.py
    uv run python scripts/refresh-chat-template-goldens.py --check
    uv run python scripts/refresh-chat-template-goldens.py --dialect chatml

Requires a hot HF cache for each dialect's representative base —
`--dialect NAME` lets you refresh one at a time if others aren't cached.

`--check` exits 0 when every existing golden matches the freshly-
computed value; exits 1 + prints a diff otherwise. Used by the
weekly drift workflow and by operators validating a base bump.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GOLDENS_ROOT = _REPO_ROOT / "tests" / "golden" / "chat-templates"
_SCENARIOS_PATH = _GOLDENS_ROOT / "scenarios.json"
_DIALECT_SPECS_PATH = _GOLDENS_ROOT / "dialect-specs.json"


def _load_scenarios() -> list[dict[str, Any]]:
    blob: dict[str, Any] = json.loads(_SCENARIOS_PATH.read_text(encoding="utf-8"))
    scenarios: list[dict[str, Any]] = blob["scenarios"]
    return scenarios


def _load_dialect_specs() -> dict[str, str | None]:
    blob = json.loads(_DIALECT_SPECS_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in blob.items() if not k.startswith("_")}


def _golden_path(dialect: str, scenario_name: str) -> Path:
    return _GOLDENS_ROOT / dialect / f"{scenario_name}.json"


def _compute_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    # `return_dict=False` makes HF return a plain `list[int]`; without it
    # newer tokenizers hand back a `BatchEncoding` whose `len(...)` is
    # the number of keys (2), not the number of tokens.
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    return len(rendered)


def _load_tokenizer(registry_key: str) -> Any:
    from dlm.base_models import BASE_MODELS

    spec = BASE_MODELS[registry_key]
    from transformers import AutoTokenizer

    # `use_fast=True` is the default but we spell it for clarity —
    # `apply_chat_template` behaves identically across fast/slow.
    return AutoTokenizer.from_pretrained(
        spec.hf_id,
        revision=spec.revision,
        use_fast=True,
        trust_remote_code=False,
    )


def _write_golden(
    path: Path,
    *,
    dialect: str,
    scenario: dict[str, Any],
    registry_key: str,
    token_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob: dict[str, Any] = {
        "dialect": dialect,
        "scenario": scenario["name"],
        "representative_base": registry_key,
        "messages": scenario["messages"],
        "expected_hf_token_count": token_count,
        "regenerated_at": datetime.now(UTC).replace(tzinfo=None, microsecond=0).isoformat(),
    }
    path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")


def _read_recorded(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    val = blob.get("expected_hf_token_count")
    return val if isinstance(val, int) else None


def _refresh_dialect(
    dialect: str,
    registry_key: str | None,
    scenarios: list[dict[str, Any]],
    *,
    check: bool,
) -> tuple[int, int]:
    """Return `(written_or_matched, drifted)` counts for reporting."""
    if registry_key is None:
        print(f"[skip] {dialect}: no representative base in registry")
        return (0, 0)

    print(f"[load] {dialect}: using {registry_key}")
    tokenizer = _load_tokenizer(registry_key)

    written = 0
    drifted = 0
    for scenario in scenarios:
        target = _golden_path(dialect, scenario["name"])
        actual = _compute_token_count(tokenizer, scenario["messages"])
        recorded = _read_recorded(target)

        if check:
            if recorded is None:
                print(f"  [MISS] {scenario['name']}: no golden on disk")
                drifted += 1
            elif recorded != actual:
                print(
                    f"  [DRIFT] {scenario['name']}: "
                    f"golden={recorded} actual={actual} delta={actual - recorded:+d}"
                )
                drifted += 1
            else:
                written += 1
        else:
            _write_golden(
                target,
                dialect=dialect,
                scenario=scenario,
                registry_key=registry_key,
                token_count=actual,
            )
            status = "=" if recorded == actual else "+"
            print(f"  [{status}] {scenario['name']}: {actual} tokens")
            written += 1

    return written, drifted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero on drift; don't write.",
    )
    parser.add_argument(
        "--dialect",
        help="Refresh only this dialect (default: all).",
    )
    args = parser.parse_args()

    scenarios = _load_scenarios()
    dialect_specs = _load_dialect_specs()
    if args.dialect is not None:
        if args.dialect not in dialect_specs:
            print(
                f"error: unknown dialect {args.dialect!r}; known: {sorted(dialect_specs)}",
                file=sys.stderr,
            )
            return 2
        dialect_specs = {args.dialect: dialect_specs[args.dialect]}

    total_written = 0
    total_drifted = 0
    for dialect, registry_key in dialect_specs.items():
        written, drifted = _refresh_dialect(dialect, registry_key, scenarios, check=args.check)
        total_written += written
        total_drifted += drifted

    if args.check:
        if total_drifted:
            print(
                f"\nFAIL: {total_drifted} golden(s) drifted. Run without "
                "`--check` to regenerate, then review the diff.",
                file=sys.stderr,
            )
            return 1
        print(f"\nOK: {total_written} golden(s) match current tokenizers.")
    else:
        print(f"\nOK: {total_written} golden(s) written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
