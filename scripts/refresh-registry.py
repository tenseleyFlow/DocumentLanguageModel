#!/usr/bin/env python
"""Re-resolve every curated base-model entry against its live sources.

Two modes:

- Default: print a human-readable diff for each entry whose pinned SHA
  no longer matches its live fetch source (or whose license/gating /
  provenance changed).
  Exit 0.
- `--check`: exit 1 if *any* entry has drifted. Used by the weekly
  CI job to open an issue when maintainer action is needed.

Does **not** write back to `registry.py` automatically — drifted SHAs
are a signal for a human to review the upstream change (new license
terms, tokenizer surgery, provenance changes, etc.). The script prints
the ready-to-paste field values so the manual update is trivial.

Usage:
    uv run python scripts/refresh-registry.py            # print diff
    uv run python scripts/refresh-registry.py --check    # CI gate
"""

from __future__ import annotations

import argparse
import sys

from dlm.base_models import BASE_MODELS
from dlm.base_models.registry_refresh import check_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any entry has drifted (for CI).",
    )
    args = parser.parse_args()

    drifts = check_registry()

    if not drifts:
        print(f"All {len(BASE_MODELS)} registry entries match their live sources.")
        return 0

    print(f"{len(drifts)} of {len(BASE_MODELS)} entries have drifted:")
    for drift in drifts:
        print(drift.render())
    print()
    print(
        "Review each upstream change (commit log / license / gating / provenance) and "
        "update `src/dlm/base_models/registry.py` by hand."
    )

    return 1 if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
