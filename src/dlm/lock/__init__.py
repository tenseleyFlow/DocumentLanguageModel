"""Per-store `dlm.lock` — determinism contract for one `.dlm` (Sprint 15).

Separate from the repo-level `uv.lock` (tool-dep pins) and from the
`manifest.json` (training run narrative). The store-level `dlm.lock`
pins the tuple `(torch, transformers, peft, trl, bitsandbytes,
accelerate, llama.cpp tag, cuda/rocm, hardware_tier, seed,
determinism_flags, determinism_class)` and carries:

- the hash of the `.dlm` source at the time the lock was written
- the base-model revision + content hash
- the license-acceptance record (Sprint 12b handoff)
- the last run id so drift-detection can date-window its output

`dlm train` validates the lock on every run, writes / updates it on
success, and honors `--strict-lock` / `--update-lock` / `--ignore-lock`.
Mismatches categorize into `allow`, `warn`, or `error` via the policy
table in `policy.py`.
"""

from __future__ import annotations

from dlm.lock.errors import LockError, LockSchemaError, LockValidationError

__all__ = [
    "LockError",
    "LockSchemaError",
    "LockValidationError",
]
