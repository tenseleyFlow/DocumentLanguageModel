"""Cross-check adapter embedding rows against base GGUF rows (Sprint 11.5).

Closes audit-04 Q2. Sprint 07's pad-fallback path sets
`modules_to_save=["embed_tokens","lm_head"]` so the LoRA adapter
carries its own trained embedding / lm-head rows. At export time the
base GGUF's corresponding rows must match byte-for-byte, or the
adapter's added-token embeddings end up multiplied against
uninitialized base rows — the "gibberish on `<|pad|>`" failure mode.

Contract:

- Runs only when `adapter_config.json::modules_to_save` includes
  either `embed_tokens` or `lm_head`. Otherwise nothing to compare.
- Hashes the added-special-token rows from both sides (PEFT
  safetensors, base GGUF) and compares.
- Skips cleanly when the adapter tokenizer has no added specials
  (nothing changed; base rows are authoritative).
- Skips cleanly when the base GGUF's embedding is block-quantized
  (re-quantize-after-the-fact pipeline; row-level check impossible)
  with an explanatory `PreflightError.probe="embedding_row_sha"`
  detail. Kept informational rather than failing until we see a real
  user hit it.

The per-architecture tensor-name map is small — llama-family /
chatml / phi3 / mistral all use the same convention
(`token_embd.weight` / `output.weight`), so one entry covers our v1
registry. Future archs can extend `_ARCH_TENSOR_NAMES` without
touching the assertion logic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from dlm.export.errors import PreflightError
from dlm.export.gguf_tensors import _SCALAR_BYTES, load_tensor_index


@dataclass(frozen=True)
class _TensorMap:
    """Maps a dlm logical module name to GGUF + PEFT name fragments."""

    gguf_name: str
    safetensors_suffix: str


# dlm-internal logical name → (GGUF tensor name, PEFT safetensors key suffix).
# `modules_to_save` layers are written under
# `base_model.model.<path>.modules_to_save.default.weight` by PEFT.
# The <path> prefix varies per arch but the trailing suffix is stable;
# we scan the file's key index for the suffix rather than hard-coding
# the full path.
_DEFAULT_TENSOR_MAP: Final[dict[str, _TensorMap]] = {
    "embed_tokens": _TensorMap(
        gguf_name="token_embd.weight",
        safetensors_suffix="embed_tokens.modules_to_save.default.weight",
    ),
    "lm_head": _TensorMap(
        gguf_name="output.weight",
        safetensors_suffix="lm_head.modules_to_save.default.weight",
    ),
}


def assert_embedding_rows_match(
    adapter_dir: Path,
    base_gguf: Path,
) -> None:
    """Verify added-token rows agree between adapter safetensors and base GGUF.

    Skip conditions (no-raise):

    - `adapter_config.json` missing or unreadable — Sprint 11's
      `check_adapter_config` already owns that error path.
    - `modules_to_save` absent / doesn't include embed_tokens or
      lm_head — adapter doesn't own any embedding rows.
    - Tokenizer config has no added special tokens — nothing to
      check; base rows are authoritative.

    Raises `PreflightError(probe="embedding_row_sha")` on a real
    mismatch or on a file-level corruption (missing safetensors,
    absent GGUF tensor, dtype unsupported).
    """
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        return
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Sprint 11's preflight owns the "unreadable adapter config"
        # error path; we silently opt out here rather than double-report.
        return

    saved_modules = cfg.get("modules_to_save") or []
    if not isinstance(saved_modules, list):
        return
    applicable = [m for m in saved_modules if m in _DEFAULT_TENSOR_MAP]
    if not applicable:
        return

    added_token_ids = _added_special_token_ids(adapter_dir)
    if not added_token_ids:
        return

    # Load the base GGUF's tensor index once; reused across modules.
    index = load_tensor_index(base_gguf)

    # Load the adapter safetensors once; reused across modules.
    safetensors_data = _load_adapter_safetensors(adapter_dir)

    mismatches: list[str] = []
    for module in applicable:
        tmap = _DEFAULT_TENSOR_MAP[module]
        adapter_tensor = _find_module_tensor(safetensors_data, tmap.safetensors_suffix)
        if adapter_tensor is None:
            # modules_to_save declared but the tensor isn't in the file —
            # surface as a real error; adapter is malformed.
            raise PreflightError(
                probe="embedding_row_sha",
                detail=(
                    f"adapter declares modules_to_save={module!r} but "
                    f"{tmap.safetensors_suffix} is absent from adapter "
                    f"safetensors in {adapter_dir}"
                ),
            )
        gguf_entry = index.find(tmap.gguf_name)
        if gguf_entry is None:
            raise PreflightError(
                probe="embedding_row_sha",
                detail=(
                    f"base GGUF {base_gguf.name} is missing tensor "
                    f"{tmap.gguf_name!r}; cannot verify adapter "
                    f"modules_to_save={module!r}"
                ),
            )
        if gguf_entry.dtype not in _SCALAR_BYTES:
            # The base was quantized to a block-quantized type (user
            # ran the flow with a non-default quant that touches the
            # embedding). We can't read rows. Surface as a preflight
            # error so the operator knows the check didn't run rather
            # than silently passing.
            raise PreflightError(
                probe="embedding_row_sha",
                detail=(
                    f"{tmap.gguf_name!r} in {base_gguf.name} is "
                    f"block-quantized (ggml dtype {gguf_entry.dtype}); "
                    "re-export with embedding tensors left at F16 or "
                    "disable the embedding_rows checker explicitly."
                ),
            )

        adapter_rows = _as_row_list(adapter_tensor)
        for tid in added_token_ids:
            if tid < 0 or tid >= len(adapter_rows):
                # Added token id is out of the adapter-tensor vocab
                # range. Either the tokenizer added tokens WITHOUT
                # resizing embed_tokens (should be impossible post
                # audit-04 M6), or the safetensors shape is stale.
                raise PreflightError(
                    probe="embedding_row_sha",
                    detail=(
                        f"added token id {tid} is out of range for "
                        f"{module} (adapter has {len(adapter_rows)} rows)"
                    ),
                )
            adapter_sha = hashlib.sha256(adapter_rows[tid]).hexdigest()
            # `index.row_bytes` raises `PreflightError` on dtype mismatch /
            # out-of-range — those bubble up naturally; no catch-and-rethrow.
            base_row = index.row_bytes(tmap.gguf_name, tid)
            base_sha = hashlib.sha256(base_row).hexdigest()
            if adapter_sha != base_sha:
                mismatches.append(
                    f"{module}[{tid}]: adapter={adapter_sha[:12]}… base={base_sha[:12]}…"
                )

    if mismatches:
        raise PreflightError(
            probe="embedding_row_sha",
            detail=(
                "adapter embedding rows disagree with base GGUF for "
                f"{len(mismatches)} added token(s): {'; '.join(mismatches)}. "
                "The base was regenerated against a different tokenizer; "
                "re-run `dlm export` with a fresh base conversion."
            ),
        )


# --- internals ----------------------------------------------------------------


def _added_special_token_ids(adapter_dir: Path) -> list[int]:
    """Return the sorted list of added-special-token ids, or `[]`."""
    cfg_path = adapter_dir / "tokenizer_config.json"
    if not cfg_path.is_file():
        return []
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    added = cfg.get("added_tokens_decoder") or {}
    if not isinstance(added, dict):
        return []

    ids: list[int] = []
    for key, entry in added.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("special") is not True:
            continue
        try:
            tid = int(key)
        except (TypeError, ValueError):
            continue
        ids.append(tid)
    return sorted(set(ids))


def _load_adapter_safetensors(adapter_dir: Path) -> Any:
    """Open the PEFT `adapter_model.safetensors` for lazy tensor access.

    Returns the `safetensors.safe_open` handle (context-managed by the
    caller) OR a dict of {key: numpy-array} if we need to materialize
    for row extraction. For simplicity we materialize the two
    modules_to_save tensors eagerly when present — they're the biggest
    tensors in a modules_to_save adapter by far, but still only one
    embedding matrix's worth, which is bounded by
    `vocab_size * hidden * dtype_bytes` (~100 MB worst-case for our
    launch registry).
    """
    path = adapter_dir / "adapter_model.safetensors"
    if not path.is_file():
        raise PreflightError(
            probe="embedding_row_sha",
            detail=(
                f"adapter_model.safetensors not found in {adapter_dir}; "
                "PEFT writes this on save_pretrained — the adapter may "
                "have been interrupted mid-save"
            ),
        )

    from safetensors import safe_open

    materialized: dict[str, Any] = {}
    try:
        with safe_open(str(path), framework="numpy") as handle:  # type: ignore[no-untyped-call]
            for key in handle.keys():  # noqa: SIM118 — safetensors API
                # Only materialize the two modules_to_save shapes we
                # care about; LoRA A/B matrices are smaller but numerous.
                if key.endswith(
                    (
                        "embed_tokens.modules_to_save.default.weight",
                        "lm_head.modules_to_save.default.weight",
                    )
                ):
                    materialized[key] = handle.get_tensor(key)
    except OSError as exc:
        raise PreflightError(
            probe="embedding_row_sha",
            detail=f"cannot read adapter safetensors at {path}: {exc}",
        ) from exc
    return materialized


def _find_module_tensor(safetensors_data: Any, suffix: str) -> Any:
    """Return the safetensors entry whose key ends with `suffix`, or None.

    PEFT prefixes the key with `base_model.model.` or
    `base_model.model.model.` depending on whether the base model has
    a `model` submodule; matching on suffix sidesteps that variation.
    """
    for key, tensor in safetensors_data.items():
        if key.endswith(suffix):
            return tensor
    return None


def _as_row_list(tensor: Any) -> list[bytes]:
    """Turn a numpy-like tensor into a per-row `bytes` list.

    The PEFT-saved embedding is (vocab_size, hidden). We slice row i
    to bytes of that single row's elements — then sha256 compares
    row-by-row without materializing the whole matrix twice.

    We don't convert dtypes here; the comparison is byte-level on both
    sides, and `convert_hf_to_gguf.py` writes the embedding in the
    tensor's native dtype (F16 → F16, F32 → F32). If the adapter was
    saved in BF16 but the base GGUF ended up as F16, the bytes differ
    even for mathematically-equal values — that's a real mismatch we
    want to flag (silent dtype conversions break inference).
    """
    import numpy as np

    arr = np.asarray(tensor)
    if arr.ndim < 2:
        # embed_tokens / lm_head are always 2D; a 1D tensor means a
        # shape mismatch. Return empty to cause the caller's bounds
        # check to surface the problem.
        return []
    # Slice ensures contiguity; tobytes on a non-contiguous row would
    # silently re-pack and mask dtype drift.
    rows: list[bytes] = []
    for i in range(arr.shape[0]):
        row = np.ascontiguousarray(arr[i])
        rows.append(bytes(row.tobytes()))
    return rows
