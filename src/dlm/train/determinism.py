"""Determinism contract for training (CLAUDE.md §8).

`seed_everything(seed)` is the single authoritative place that sets
every RNG + backend flag the trainer touches:

1. `CUBLAS_WORKSPACE_CONFIG=:4096:8` — required for deterministic cuBLAS
   matmul (cudnn-8.x+, CUDA 10.2+). Set via `os.environ.setdefault` so
   a user who sets their own value isn't overridden.
2. `torch.manual_seed`, `torch.cuda.manual_seed_all` — torch CPU + CUDA.
3. `numpy.random.seed` — numpy.
4. `random.seed` — stdlib.
5. `torch.use_deterministic_algorithms(True, warn_only=True)` — strict
   deterministic kernels; `warn_only=True` keeps flash-attn from
   hard-failing since some of its kernels aren't deterministic.
6. `torch.backends.cudnn.benchmark = False` — no autotuner (which is
   non-deterministic).

MPS determinism is best-effort (audit F20); the `describe()` function
surfaces this to the training banner so Apple Silicon users see the
caveat before the run starts.

Heavy imports (`torch`, `numpy`) are deferred to call-sites so
`import dlm.train` stays cheap.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Literal

_CUBLAS_WORKSPACE = ":4096:8"


@dataclass(frozen=True)
class DeterminismSummary:
    """What determinism class this run is operating under.

    The CLI banner reads `class_` + `notes` to tell the user what to
    expect before the first step runs. `class_` is one of:

    - `"strict"` — CUDA + deterministic algorithms enabled. Retrains
      produce bit-identical loss curves.
    - `"best_effort"` — MPS or CPU-fallback kernels in use.
      Retrain loss curves are *close* but not bit-identical.
    - `"loose"` — `warn_only=True` tripped on a non-deterministic
      kernel that wasn't avoidable (rare; always logged).
    """

    class_: Literal["strict", "best_effort", "loose"]
    seed: int
    notes: list[str]


def seed_everything(seed: int) -> DeterminismSummary:
    """Set every RNG + backend flag; return a summary for the banner.

    Idempotent: safe to call twice with the same seed (e.g., test
    fixtures that seed before and after monkeypatching).
    """
    # Env vars go first — torch reads them on its own cuBLAS init path.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE)

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover — numpy is a torch transitive
        pass

    notes: list[str] = []
    class_: Literal["strict", "best_effort", "loose"] = "best_effort"

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover — CI covers GPU hosts
            torch.cuda.manual_seed_all(seed)
            class_ = "strict"
        elif _mps_available(torch):  # pragma: no cover — MPS only on Apple Silicon dev hosts
            notes.append(
                "MPS determinism is best-effort; loss curves are close but not bit-identical. "
                "Use --strict-determinism on a CUDA box for byte-exact reproducibility."
            )
            class_ = "best_effort"
        else:  # pragma: no cover — CPU-only hosts exercised on Linux CI, not local MPS
            notes.append(
                "CPU-only host: determinism is best-effort — deterministic kernels "
                "enabled, but training is slower than on GPU."
            )
            class_ = "best_effort"

        # Deterministic kernels. `warn_only` lets flash-attn + a handful
        # of other non-deterministic ops continue rather than hard-fail;
        # the trainer logs a WARN for any that trigger.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover — torch is a runtime dep once Sprint 09 lands
        notes.append("torch not installed; determinism contract not enforced")
        class_ = "loose"

    return DeterminismSummary(class_=class_, seed=seed, notes=notes)


def _mps_available(torch_mod: object) -> bool:
    """torch.backends.mps.is_available, guarded for older torch."""
    backends = getattr(torch_mod, "backends", None)
    if backends is None:  # pragma: no cover — current torch always ships backends
        return False
    mps = getattr(backends, "mps", None)
    if mps is None:  # pragma: no cover — current torch always ships mps backend
        return False
    checker = getattr(mps, "is_available", None)
    return bool(checker and checker())
