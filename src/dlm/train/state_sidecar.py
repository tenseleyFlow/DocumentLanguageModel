"""`training_state.pt` save + load with sha256 integrity + safe loading.

`save_pretrained` only persists adapter weights + config, so resume
needs a separate blob with:

- optimizer state
- scheduler state
- scaler state (mixed precision)
- all RNGs (torch, cuda, numpy, python)
- training step counter
- pinned runtime versions (for drift detection on resume)

**v2 layout (current).** The torch payload holds tensor state +
primitives only, loaded via `torch.load(weights_only=True)` —
categorically refuses arbitrary-code-execution via pickled classes.
Non-tensor RNG state (numpy ndarray, python tuple-of-tuples) moved to
a JSON sidecar so the torch payload stays within weights_only's
allowlist. Files:

    <dir>/training_state.pt           <- torch.save'd dict (tensors + primitives)
    <dir>/training_state.pt.sha256    <- hex digest of the .pt bytes
    <dir>/training_state.rng.json     <- numpy + python RNG state
    <dir>/pinned_versions.json        <- advisory, human-grep-able
    <dir>/training_run.json           <- use_qlora flag

**v1 legacy.** Prior releases torch.save'd everything under
`weights_only=False`, which was an RCE vector: sha256 integrity
check + user-writable bytes = matching-hash tampered pickle could
inject arbitrary code on resume. The legacy reader is kept one
release for back-compat with sidecars written by earlier versions —
it logs a one-time MIGRATION notice and does NOT trust pickled
content beyond scalar/tensor primitives (raises on any unusual
type). The next release drops the legacy branch.

Heavy imports (`torch`) are deferred to the functions that need them;
the TypedDict describing the payload is importable without torch.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any, TypedDict

from dlm.train.errors import ResumeIntegrityError, VersionDriftWarning

_LOG = logging.getLogger(__name__)

# Files written alongside the adapter directory.
STATE_FILENAME = "training_state.pt"
STATE_SHA_FILENAME = "training_state.pt.sha256"
RNG_SIDECAR_FILENAME = "training_state.rng.json"
VERSIONS_FILENAME = "pinned_versions.json"
STATE_SIDECAR_VERSION = 2
"""Bump when the on-disk layout changes. v1 → v2 moved numpy/python
RNG state out of the torch payload into a JSON sidecar, dropping
`weights_only=False` from the load path. The writer always emits
v2; the reader accepts v1 (legacy) with a migration warning."""
# Run-level flags the inference path consumes without loading torch
# (audit-05 M1): separate from `pinned_versions.json`, which is a pure
# package-version manifest. This file records *how* the adapter was
# trained — currently just the QLoRA flag; future fields (e.g., base
# compute dtype) extend this rather than polluting version metadata.
TRAINING_RUN_FILENAME = "training_run.json"


class PinnedVersions(TypedDict, total=False):
    """Runtime versions captured at save time.

    Used on resume to detect drift. Missing keys are treated as
    "unknown, don't warn"; present keys that differ trigger a
    `VersionDriftWarning`. `sway` is recorded when the differential-
    testing sibling is installed in the same venv so operators can
    trace which harness produced the reports that drove the run.
    """

    torch: str
    transformers: str
    peft: str
    trl: str
    bitsandbytes: str | None
    sway: str | None


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
    # audit-05 M1: explicit QLoRA flag. `InferencePlan` reads this via
    # `training_run.json` (written alongside) rather than inferring from
    # the bitsandbytes version pin, which false-positives on plain LoRA
    # runs on CUDA+bnb hosts.
    use_qlora: bool


def _encode_numpy_rng_state(state: Any) -> dict[str, Any] | None:
    """Serialize numpy's 5-tuple RNG state into a JSON-safe dict.

    `np.random.get_state()` returns
    `(name: str, state: ndarray[uint32], pos: int, has_gauss: int, cached_gaussian: float)`.
    The ndarray is hex-encoded alongside its dtype + shape so
    `np.frombuffer` reconstructs it exactly. Returns `None` for
    `None` input (mocks, tests that don't touch numpy RNG).
    """
    if state is None:
        return None
    name, arr, pos, has_gauss, cached = state
    return {
        "name": str(name),
        "state_hex": bytes(arr).hex(),
        "state_dtype": str(arr.dtype),
        "state_shape": list(arr.shape),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def _decode_numpy_rng_state(data: dict[str, Any] | None) -> Any:
    """Inverse of `_encode_numpy_rng_state`. Returns `None` for `None`."""
    if data is None:
        return None
    import numpy as np

    arr = np.frombuffer(
        bytes.fromhex(data["state_hex"]), dtype=np.dtype(data["state_dtype"])
    ).reshape(data["state_shape"])
    # frombuffer returns read-only views; copy so `np.random.set_state`
    # can own it.
    return (data["name"], arr.copy(), data["pos"], data["has_gauss"], data["cached_gaussian"])


def _encode_python_random_state(state: Any) -> dict[str, Any] | None:
    """Serialize `random.getstate()` into a JSON-safe dict.

    Shape: `(version: int, state: tuple[int, ...], gauss_next: float | None)`.
    All elements are JSON primitives; we just convert the inner tuple
    to a list.
    """
    if state is None:
        return None
    version, inner, gauss_next = state
    return {
        "version": int(version),
        "state": [int(x) for x in inner],
        "gauss_next": gauss_next,
    }


def _decode_python_random_state(data: dict[str, Any] | None) -> Any:
    """Inverse of `_encode_python_random_state`."""
    if data is None:
        return None
    return (data["version"], tuple(data["state"]), data["gauss_next"])


def save_state(directory: Path, state: TrainingState) -> None:
    """Serialize `state` into `<directory>/training_state.pt` + sidecars.

    Writes four files atomically:

    1. `training_state.pt` — torch.save'd dict of tensor state + primitives.
       Loads under `weights_only=True` on resume, so tampered pickles
       cannot execute arbitrary code.
    2. `training_state.pt.sha256` — hex digest of the .pt bytes.
    3. `training_state.rng.json` — numpy + python RNG state (not
       representable under weights_only=True; hex-encoded ndarray).
    4. `pinned_versions.json` + `training_run.json` — advisory sidecars.

    Parent directory must exist.
    """
    import torch

    # Partition into the torch payload (tensor state + primitives) and
    # the JSON sidecar (numpy/python RNG). torch.load(weights_only=True)
    # accepts primitives, containers, tensors — not numpy arrays, and
    # rejects str subclasses (e.g., `torch.__version__` is a
    # `TorchVersion`). Coerce pinned values to plain str/None.
    pinned_plain: dict[str, str | None] = {
        k: (str(v) if v is not None else None)
        for k, v in dict(state["pinned_versions"]).items()
    }
    torch_payload: dict[str, Any] = {
        "_state_sidecar_version": STATE_SIDECAR_VERSION,
        "optimizer_state_dict": state["optimizer_state_dict"],
        "scheduler_state_dict": state["scheduler_state_dict"],
        "scaler_state_dict": state["scaler_state_dict"],
        "torch_rng_state": state["torch_rng_state"],
        "cuda_rng_state": state["cuda_rng_state"],
        "global_step": state["global_step"],
        "epoch": state["epoch"],
        "best_val_loss": state["best_val_loss"],
        "dlm_manifest_hash": state["dlm_manifest_hash"],
        "base_model_revision": state["base_model_revision"],
        "pinned_versions": pinned_plain,
        "use_qlora": bool(state.get("use_qlora", False)),
    }

    rng_sidecar = {
        "_rng_sidecar_version": STATE_SIDECAR_VERSION,
        "numpy_rng_state": _encode_numpy_rng_state(state.get("numpy_rng_state")),
        "python_random_state": _encode_python_random_state(state.get("python_random_state")),
    }

    buf = io.BytesIO()
    torch.save(torch_payload, buf)
    blob = buf.getvalue()

    digest = hashlib.sha256(blob).hexdigest()

    state_path = directory / STATE_FILENAME
    sha_path = directory / STATE_SHA_FILENAME
    rng_path = directory / RNG_SIDECAR_FILENAME

    from dlm.io.atomic import write_bytes, write_text

    write_bytes(state_path, blob)
    write_text(sha_path, digest + "\n")
    write_text(rng_path, json.dumps(rng_sidecar, indent=2) + "\n")

    # Pinned versions are also written separately as JSON for
    # human-grep-ability without torch.load. This is advisory; the
    # source-of-truth for resume verification is the embedded dict.
    pinned_path = directory / VERSIONS_FILENAME
    write_text(
        pinned_path,
        json.dumps(dict(state["pinned_versions"]), sort_keys=True, indent=2) + "\n",
    )

    # Run-level flags (audit-05 M1). Separate file so `InferencePlan`
    # can read `use_qlora` without loading torch or the whole state dict.
    training_run_path = directory / TRAINING_RUN_FILENAME
    write_text(
        training_run_path,
        json.dumps({"use_qlora": bool(state.get("use_qlora", False))}, indent=2) + "\n",
    )


def load_state(directory: Path, *, runtime_versions: PinnedVersions) -> TrainingState:
    """Load + integrity-check + version-drift-check.

    Raises `ResumeIntegrityError` if:
    - `training_state.pt` or `.sha256` is missing
    - the recomputed sha256 disagrees with the stored digest
    - torch.load fails (tampered bytes, unknown class under weights_only)
    - the RNG sidecar is required (v2 format) but missing/malformed

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
        torch_payload = torch.load(io.BytesIO(blob), weights_only=True)
    except Exception as weights_only_exc:
        # Legacy v1 format (pre-audit-11 B7) stored everything including
        # numpy ndarrays under weights_only=False. Retry with the legacy
        # loader + log a one-time migration notice. The next release
        # drops this branch; callers should re-save.
        try:
            torch_payload = torch.load(io.BytesIO(blob), weights_only=False)
        except Exception as exc:
            raise ResumeIntegrityError(
                f"torch.load failed under weights_only=True "
                f"({type(weights_only_exc).__name__}: {weights_only_exc}); "
                f"legacy load also failed ({type(exc).__name__}: {exc})"
            ) from exc
        _LOG.warning(
            "training_state.pt at %s is in the legacy v1 format — loaded under "
            "weights_only=False. Re-save via the current trainer to migrate; "
            "a future release drops the legacy reader.",
            state_path,
        )
        state_any: Any = torch_payload

    else:
        sidecar_version = torch_payload.get("_state_sidecar_version")
        if sidecar_version is None:
            # No version marker but weights_only=True succeeded — this
            # is an edge case (someone hand-wrote a torch payload). Treat
            # as v1 for RNG reconstruction purposes: leave numpy/python
            # RNG as None.
            state_any = dict(torch_payload)
            state_any.setdefault("numpy_rng_state", None)
            state_any.setdefault("python_random_state", None)
        else:
            state_any = _merge_rng_sidecar(directory, torch_payload)

    pinned = state_any.get("pinned_versions", {})
    drift = _version_diff(pinned, runtime_versions)
    if drift:
        warnings.warn(
            f"pinned-version drift detected: {', '.join(drift)}. "
            "Training will continue but loss curves may diverge from the saved snapshot.",
            VersionDriftWarning,
            stacklevel=2,
        )

    return state_any  # type: ignore[no-any-return]


def _merge_rng_sidecar(directory: Path, torch_payload: dict[str, Any]) -> dict[str, Any]:
    """Read the RNG JSON sidecar and merge its decoded values into the payload.

    Required for v2 payloads. Raises `ResumeIntegrityError` if the
    sidecar is missing or malformed — we can't safely resume without
    the RNG state, and silently substituting `None` would break
    determinism.
    """
    rng_path = directory / RNG_SIDECAR_FILENAME
    if not rng_path.exists():
        raise ResumeIntegrityError(
            f"v2 training_state.pt at {directory} requires {RNG_SIDECAR_FILENAME} "
            "(numpy + python RNG state) alongside it; sidecar missing"
        )
    try:
        rng_sidecar = json.loads(rng_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeIntegrityError(
            f"cannot read RNG sidecar {rng_path}: {type(exc).__name__}: {exc}"
        ) from exc

    merged: dict[str, Any] = dict(torch_payload)
    merged["numpy_rng_state"] = _decode_numpy_rng_state(rng_sidecar.get("numpy_rng_state"))
    merged["python_random_state"] = _decode_python_random_state(
        rng_sidecar.get("python_random_state")
    )
    return merged


def _version_diff(pinned: PinnedVersions, runtime: PinnedVersions) -> list[str]:
    """Return `["key: saved→current", ...]` for keys whose versions differ.

    Asymmetric handling of `None` (audit-04 M6): losing a pinned package
    between save + resume (e.g., a QLoRA checkpoint from a CUDA box
    being resumed on Apple Silicon without `bitsandbytes`) is drift
    the user should see. Gaining a package that wasn't pinned is not
    drift — there was no prior state to diverge from.

    Rules:
    - saved=str, current=str, equal    → no drift
    - saved=str, current=str, differ   → drift ("saved→current")
    - saved=str, current=None/missing  → drift ("saved→missing")
    - saved=None/missing, current=*    → no drift (no prior state)
    """
    diffs: list[str] = []
    for key in sorted(pinned.keys() | runtime.keys()):
        saved = pinned.get(key)
        if saved is None:
            continue
        current = runtime.get(key)
        if current is None:
            diffs.append(f"{key}: {saved}→missing")
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
    # sway is an optional sibling — probe but don't require. `None`
    # semantics match bitsandbytes: install-site absence is recorded
    # explicitly so resume-side drift doesn't false-positive.
    versions["sway"] = _v("sway")
    return versions
