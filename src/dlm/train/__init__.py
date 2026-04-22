"""Training engine — load base, attach LoRA/QLoRA, run one SFT cycle.

Heavy imports (`torch`, `transformers`, `peft`, `trl`,
`bitsandbytes`) are deferred to the functions that use them so
`import dlm.train` stays cheap.
"""

from __future__ import annotations

from dlm.train.checkpoint_commit import (
    allocate_next_version,
    commit_version,
    list_pending_versions,
)
from dlm.train.determinism import DeterminismSummary, seed_everything
from dlm.train.disk_preflight import estimate_checkpoint_bytes, preflight_disk
from dlm.train.errors import (
    DiskSpaceError,
    OOMError,
    ResumeIntegrityError,
    TrainingError,
    VersionDriftWarning,
)
from dlm.train.integrity import (
    NaNEvalError,
    NaNWeightsError,
    assert_eval_finite,
    assert_finite_adapter,
    audit_trainable_finite,
)
from dlm.train.logger import Banner, StepLogger, log_path_for
from dlm.train.oom_guard import format_oom_message, recommend_grad_accum
from dlm.train.state_sidecar import (
    STATE_FILENAME,
    STATE_SHA_FILENAME,
    TRAINING_RUN_FILENAME,
    VERSIONS_FILENAME,
    PinnedVersions,
    TrainingState,
    capture_runtime_versions,
    load_state,
    save_state,
)
from dlm.train.trainer import TrainingRunResult, run

__all__ = [
    "Banner",
    "DeterminismSummary",
    "DiskSpaceError",
    "NaNEvalError",
    "NaNWeightsError",
    "OOMError",
    "PinnedVersions",
    "ResumeIntegrityError",
    "STATE_FILENAME",
    "STATE_SHA_FILENAME",
    "StepLogger",
    "TRAINING_RUN_FILENAME",
    "TrainingError",
    "TrainingRunResult",
    "TrainingState",
    "VERSIONS_FILENAME",
    "VersionDriftWarning",
    "allocate_next_version",
    "assert_eval_finite",
    "assert_finite_adapter",
    "audit_trainable_finite",
    "capture_runtime_versions",
    "commit_version",
    "estimate_checkpoint_bytes",
    "format_oom_message",
    "list_pending_versions",
    "load_state",
    "log_path_for",
    "preflight_disk",
    "recommend_grad_accum",
    "run",
    "save_state",
    "seed_everything",
]
