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

from dlm.lock.builder import build_lock, hardware_tier_from_backend, hash_dlm_file
from dlm.lock.errors import LockError, LockSchemaError, LockValidationError, LockWriteError
from dlm.lock.policy import Severity, classify_mismatches
from dlm.lock.schema import CURRENT_LOCK_VERSION, LOCK_FILENAME, DlmLock
from dlm.lock.validator import LockDecision, LockMode, validate_lock
from dlm.lock.writer import load_lock, lock_path, write_lock

__all__ = [
    "CURRENT_LOCK_VERSION",
    "LOCK_FILENAME",
    "DlmLock",
    "LockDecision",
    "LockError",
    "LockMode",
    "LockSchemaError",
    "LockValidationError",
    "LockWriteError",
    "Severity",
    "build_lock",
    "classify_mismatches",
    "hardware_tier_from_backend",
    "hash_dlm_file",
    "load_lock",
    "lock_path",
    "validate_lock",
    "write_lock",
]
