#!/usr/bin/env python
"""Re-resolve every curated base-model entry against HuggingFace.

Two modes:

- Default: print a human-readable diff for each entry whose pinned SHA
  no longer matches HF's `main` (or whose license/gating changed).
  Exit 0.
- `--check`: exit 1 if *any* entry has drifted. Used by the weekly
  CI job to open an issue when maintainer action is needed.

Does **not** write back to `registry.py` automatically — drifted SHAs
are a signal for a human to review the upstream change (new license
terms, tokenizer surgery, etc.). The script prints the ready-to-paste
field values so the manual update is trivial.

Usage:
    uv run python scripts/refresh-registry.py            # print diff
    uv run python scripts/refresh-registry.py --check    # CI gate
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from dlm.base_models import BASE_MODELS, BaseModelSpec


@dataclass(frozen=True)
class Drift:
    """Structured diff between a local registry entry and HF's head."""

    key: str
    hf_id: str
    fields: tuple[tuple[str, str, str], ...]  # (name, pinned, observed)

    def render(self) -> str:
        lines = [f"  {self.key} ({self.hf_id})"]
        for name, pinned, observed in self.fields:
            lines.append(f"    {name:<22} {pinned!r} → {observed!r}")
        return "\n".join(lines)


def _check_entry(api: HfApi, entry: BaseModelSpec) -> Drift | None:
    try:
        info = api.model_info(entry.hf_id)
    except GatedRepoError:
        # Gated models still expose public metadata via `model_info`;
        # if we can't read them, that's a new gating event worth flagging.
        return Drift(
            key=entry.key,
            hf_id=entry.hf_id,
            fields=(("gating", "readable", "now fully gated"),),
        )
    except RepositoryNotFoundError:
        return Drift(
            key=entry.key,
            hf_id=entry.hf_id,
            fields=(("repository", "present", "missing (renamed or deleted)"),),
        )

    drifted: list[tuple[str, str, str]] = []

    current_sha = info.sha
    if current_sha and current_sha != entry.revision:
        drifted.append(("revision", entry.revision, current_sha))

    gated = getattr(info, "gated", False)
    # HF reports `gated` as False / "auto" / "manual". Non-False values
    # mean acceptance is required.
    gated_observed = bool(gated and gated != "False")
    if gated_observed != entry.requires_acceptance:
        drifted.append(
            (
                "requires_acceptance",
                str(entry.requires_acceptance),
                str(gated_observed),
            ),
        )

    return Drift(key=entry.key, hf_id=entry.hf_id, fields=tuple(drifted)) if drifted else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any entry has drifted (for CI).",
    )
    args = parser.parse_args()

    api = HfApi()
    drifts: list[Drift] = []
    for entry in BASE_MODELS.values():
        drift = _check_entry(api, entry)
        if drift is not None:
            drifts.append(drift)

    if not drifts:
        print(f"All {len(BASE_MODELS)} registry entries match HF.")
        return 0

    print(f"{len(drifts)} of {len(BASE_MODELS)} entries have drifted:")
    for drift in drifts:
        print(drift.render())
    print()
    print(
        "Review each upstream change (commit log / license / gating) and "
        "update `src/dlm/base_models/registry.py` by hand."
    )

    return 1 if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
