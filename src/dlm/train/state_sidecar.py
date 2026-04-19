"""`training_state.pt` save + load with sha256 integrity (audit F12).

`save_pretrained` only persists adapter weights + config, so resume
needs a separate blob with:

- optimizer state
- scheduler state
- scaler state (mixed precision)
- all RNGs (torch, cuda, numpy, python)
- training step counter
- pinned runtime versions (for drift detection on resume)

We write two files side by side:

    <dir>/training_state.pt           <- torch.save'd dict
    <dir>/training_state.pt.sha256    <- hex digest, one line

`load()` recomputes the digest and refuses to deserialize on mismatch.
This is a belt-and-suspenders check: the two-phase checkpoint commit
already prevents torn writes, but a corrupted SSD sector or a partial
disk full won't be caught by os.replace alone.

Heavy imports (`torch`) are deferred to the functions that need them;
the TypedDict describing the payload is importable without torch.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, TypedDict

from dlm.train.errors import ResumeIntegrityError, VersionDriftWarning

# Files written alongside the adapter directory.
STATE_FILENAME = "training_state.pt"
STATE_SHA_FILENAME = "training_state.pt.sha256"
VERSIONS_FILENAME = "pinned_versions.json"


class PinnedVersions(TypedDict, total=False):
    """Runtime versions captured at save time.

    Used on resume to detect drift. Missing keys are treated as
    "unknown, don't warn"; present keys that differ trigger a
    `VersionDriftWarning`.
    """

    torch: str
    transformers: str
    peft: str
    trl: str
    bitsandbytes: str | None


class TrainingState(TypedDict):
    """Complete resumable training state."""

    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any]
    scaler_state_dict: dict[str, Any] | None
    torch_rng_state: Any  # torch.Tensor
    cuda_rng_state: Any | None  # list[torch.Tensor] or None
    numpy_rng_state: Any  # tuple from np.random.get_state()
    python_random_state: Any  # tuple from random.getstate()
    global_step: int
    epoch: float
    best_val_loss: float | None
    dlm_manifest_hash: str | None
    base_model_revision: str
    pinned_versions: PinnedVersions


def save_state(directory: Path, state: TrainingState) -> None:
    """Serialize `state` into `<directory>/training_state.pt` + `.sha256`.

    Uses `torch.save` to a bytes buffer first, computes sha256 over the
    exact bytes, then writes both files atomically (tmp + rename).
    Parent directory must exist.
    """
    import torch

    buf = io.BytesIO()
    torch.save(state, buf)
    blob = buf.getvalue()

    digest = hashlib.sha256(blob).hexdigest()

    state_path = directory / STATE_FILENAME
    sha_path = directory / STATE_SHA_FILENAME

    from dlm.io.atomic import write_bytes, write_text

    write_bytes(state_path, blob)
    write_text(sha_path, digest + "\n")

    # Pinned versions are also written separately as JSON for
    # human-grep-ability without torch.load. This is advisory; the
    # source-of-truth for resume verification is the embedded dict.
    pinned_path = directory / VERSIONS_FILENAME
    write_text(
        pinned_path,
        json.dumps(dict(state["pinned_versions"]), sort_keys=True, indent=2) + "\n",
    )


def load_state(directory: Path, *, runtime_versions: PinnedVersions) -> TrainingState:
    """Load + integrity-check + version-drift-check.

    Raises `ResumeIntegrityError` if:
    - `training_state.pt` or `.sha256` is missing
    - the recomputed sha256 disagrees with the stored digest
    - torch.load fails for any reason (tampered bytes, etc.)

    Emits `VersionDriftWarning` (via `warnings.warn`) if the pinned
    versions differ from `runtime_versions` — non-fatal.
    """
    import warnings

    import torch

    state_path = directory / STATE_FILENAME
    sha_path = directory / STATE_SHA_FILENAME
    if not state_path.exists():
        raise ResumeIntegrityError(f"missing training state file: {state_path}")
    if not sha_path.exists():
        raise ResumeIntegrityError(f"missing sha256 sidecar: {sha_path}")

    blob = state_path.read_bytes()
    actual = hashlib.sha256(blob).hexdigest()
    expected = sha_path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise ResumeIntegrityError(
            f"training_state.pt sha256 mismatch: expected {expected[:16]}…, got {actual[:16]}…"
        )

    try:
        state = torch.load(io.BytesIO(blob), weights_only=False)
    except Exception as exc:
        raise ResumeIntegrityError(f"torch.load failed: {exc}") from exc

    pinned = state.get("pinned_versions", {})
    drift = _version_diff(pinned, runtime_versions)
    if drift:
        warnings.warn(
            f"pinned-version drift detected: {', '.join(drift)}. "
            "Training will continue but loss curves may diverge from the saved snapshot.",
            VersionDriftWarning,
            stacklevel=2,
        )

    return state  # type: ignore[no-any-return]


def _version_diff(pinned: PinnedVersions, runtime: PinnedVersions) -> list[str]:
    """Return `["key: saved→current", ...]` for keys whose versions differ."""
    diffs: list[str] = []
    for key in sorted(pinned.keys() | runtime.keys()):
        saved = pinned.get(key)
        current = runtime.get(key)
        if saved is None or current is None:
            # Missing key on either side — don't warn; could just be
            # an optional dependency like bitsandbytes.
            continue
        if saved != current:
            diffs.append(f"{key}: {saved}→{current}")
    return diffs


def capture_runtime_versions() -> PinnedVersions:
    """Read the actual runtime versions of pinned training packages.

    Imports are deferred; missing packages get `None` (bitsandbytes on
    Apple Silicon is the canonical example).
    """
    versions: PinnedVersions = {}

    def _v(name: str) -> str | None:
        try:
            mod = __import__(name)
        except ImportError:
            return None
        return getattr(mod, "__version__", None)

    for name in ("torch", "transformers", "peft", "trl"):
        v = _v(name)
        if v is not None:
            versions[name] = v
    bnb = _v("bitsandbytes")
    versions["bitsandbytes"] = bnb
    return versions
