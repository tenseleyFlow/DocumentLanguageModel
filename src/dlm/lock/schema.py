"""`DlmLock` Pydantic schema.

Immutable-by-default per-store record of the determinism contract in
effect for one training-or-export cycle. Written atomically after
each successful run; validated against the host's current runtime
on the next run.

Kept small and stable: this file is the diff unit `dlm show`,
`dlm migrate`, and the weekly drift jobs reason about. Unknown keys
are forbidden so a future schema bump fails loudly rather than
silently losing data.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from dlm.base_models.license import LicenseAcceptance

LOCK_FILENAME: Final[str] = "dlm.lock"

# Bump when the DlmLock shape changes. Readers reject newer versions
# rather than guessing at unknown fields; writers only emit `CURRENT`.
CURRENT_LOCK_VERSION: Final[int] = 1

HardwareTier = Literal[
    "cuda-sm80+",
    "cuda-sm<80",
    "mps",
    "cpu",
    "rocm",
]

DeterminismClass = Literal[
    "strong",
    "best-effort",
    "advisory",
]


class DlmLock(BaseModel):
    """Per-store determinism contract.

    Stored at `<store_root>/dlm.lock` — sibling to `manifest.json` and
    `training_state.pt`. Round-trips through `writer.write_lock` /
    `writer.load_lock` with byte-stable JSON (sort_keys=True).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lock_version: int = Field(CURRENT_LOCK_VERSION, ge=1)
    created_at: datetime
    dlm_id: str = Field(..., min_length=1)
    # SHA-256 hex of the `.dlm` source text at lock-write time.
    dlm_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    base_model_revision: str = Field(..., min_length=1)
    base_model_sha256: str | None = None

    # Keys: torch, transformers, peft, trl, bitsandbytes, accelerate,
    # llama_cpp. Values: version string ("2.5.1", "0.45.0", "b8816", …).
    # Absent keys mean the runtime didn't load that package (e.g. bnb
    # is CUDA-only; llama_cpp only populated after export).
    pinned_versions: dict[str, str] = Field(default_factory=dict)

    cuda_version: str | None = None
    rocm_version: str | None = None
    hardware_tier: HardwareTier
    seed: int
    # Free-form: `{"cublas_workspace": ":4096:8",
    # "use_deterministic_algorithms": True, ...}`. Values are `Any` to
    # allow bools, strings, and numbers without a per-key schema.
    determinism_flags: dict[str, Any] = Field(default_factory=dict)
    determinism_class: DeterminismClass
    license_acceptance: LicenseAcceptance | None = None
    last_run_id: int = Field(..., ge=1)
    # Number of data-parallel ranks this run used. 1 for
    # single-GPU / single-process. Multi-GPU runs record the world
    # size so a resume with a different world_size can be flagged.
    world_size: int = Field(1, ge=1)
