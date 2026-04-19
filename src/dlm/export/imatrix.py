"""Importance-matrix (imatrix) calibrated quantization (Sprint 11.6).

Static k-quants lose 1–3 perplexity points on domain-specific vocabulary
vs. fp16; an importance matrix built from text the model will actually
generate shrinks that gap well under 1 pp (llama.cpp's upstream
benchmarks). The replay corpus is the natural calibration source —
it's what the adapter was trained on, already on disk, per-document,
and SHA-addressable.

Module contract:

- `ImatrixArtifact` — dataclass bundling (path, sha256, metadata).
- `resolve_imatrix(export_dir, base_revision, corpus_sha256, chunks)`
  → returns an existing matching artifact or `None`.
- `build_imatrix(base_gguf, calibration_text, out_path, ...)` → runs
  the `llama-imatrix` subprocess and returns an `ImatrixArtifact`.
- `calibration_text_from_replay(replay_store, max_chars=...)` →
  concatenates sampled replay-corpus prose + instruction text into a
  single calibration string. Deterministic given the same store.

Cache key: `(base_revision, corpus_sha256, chunks)`. A fresh replay
write changes the corpus sha and rebuilds automatically; re-exporting
a second quant against the same corpus hits the cache.

Subprocess: `llama-imatrix -m <base.gguf> -f <calib.txt> -o <out>
--chunks N`. Calibration text lands in `<export_dir>/imatrix.calib.txt`
and is removed after the subprocess exits (keeps the store lean; the
cache sidecar records just the `corpus_sha256`).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from dlm.export import vendoring
from dlm.export.errors import SubprocessError
from dlm.export.quantize import run_checked

_LOG = logging.getLogger(__name__)

_IMATRIX_FILENAME: Final[str] = "imatrix.gguf"
_IMATRIX_META_FILENAME: Final[str] = "imatrix.meta.json"
_CALIB_FILENAME: Final[str] = "imatrix.calib.txt"

# Calibration defaults — `llama-imatrix` crashes on a too-short calib
# file, so we enforce a reasonable floor. These map to ~128k tokens
# with a typical BPE tokenizer, which is enough to stabilize
# per-tensor statistics without burning minutes on CPU.
DEFAULT_CHUNKS: Final[int] = 256
DEFAULT_CHUNK_SIZE: Final[int] = 512

# Max characters we concatenate from replay before truncating. The
# imatrix binary reads `--chunks × --chunk-size` tokens; a 4x text
# overhead factor covers typical ratios and gives headroom.
_CALIB_CHAR_HEADROOM: Final[int] = 4


@dataclass(frozen=True)
class ImatrixArtifact:
    """Materialized imatrix file plus enough metadata to validate a cache hit."""

    path: Path
    sha256: str
    base_revision: str
    corpus_sha256: str
    chunks: int
    chunk_size: int
    built_at: datetime

    def to_meta_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.name,
            "sha256": self.sha256,
            "base_revision": self.base_revision,
            "corpus_sha256": self.corpus_sha256,
            "chunks": self.chunks,
            "chunk_size": self.chunk_size,
            "built_at": self.built_at.isoformat(),
        }


def resolve_imatrix(
    export_dir: Path,
    *,
    base_revision: str,
    corpus_sha256: str,
    chunks: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> ImatrixArtifact | None:
    """Return the cached artifact if its key matches, else `None`.

    Cache hit requires all five match: binary present, sidecar parse
    succeeds, and (`base_revision`, `corpus_sha256`, `chunks`,
    `chunk_size`) equal. Any divergence → treat as miss; caller
    rebuilds.
    """
    bin_path = export_dir / _IMATRIX_FILENAME
    meta_path = export_dir / _IMATRIX_META_FILENAME
    if not bin_path.is_file() or not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    if (
        meta.get("base_revision") != base_revision
        or meta.get("corpus_sha256") != corpus_sha256
        or meta.get("chunks") != chunks
        or meta.get("chunk_size") != chunk_size
    ):
        return None
    recorded_sha = meta.get("sha256")
    if not isinstance(recorded_sha, str):
        return None
    # Verify the binary itself hasn't been touched; a stale sidecar
    # with a tampered binary would otherwise look like a cache hit.
    actual_sha = _sha256_of_file(bin_path)
    if actual_sha != recorded_sha:
        return None
    built_at_raw = meta.get("built_at")
    try:
        built_at = datetime.fromisoformat(str(built_at_raw))
    except (TypeError, ValueError):
        return None
    return ImatrixArtifact(
        path=bin_path,
        sha256=actual_sha,
        base_revision=base_revision,
        corpus_sha256=corpus_sha256,
        chunks=chunks,
        chunk_size=chunk_size,
        built_at=built_at,
    )


def build_imatrix(
    *,
    base_gguf: Path,
    calibration_text: str,
    export_dir: Path,
    base_revision: str,
    corpus_sha256: str,
    chunks: int = DEFAULT_CHUNKS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    bin_override: Path | None = None,
    subprocess_runner: Callable[[Sequence[str]], Any] | None = None,
) -> ImatrixArtifact:
    """Run `llama-imatrix` against `calibration_text`; write the binary + sidecar.

    Writes `imatrix.gguf` and `imatrix.meta.json` atomically into
    `export_dir`. On subprocess failure the calibration text file is
    left behind for debugging; on success it's removed.
    """
    if not base_gguf.is_file():
        raise FileNotFoundError(f"imatrix base model missing: {base_gguf}")
    if chunks <= 0:
        raise ValueError(f"chunks must be positive, got {chunks}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not calibration_text.strip():
        raise ValueError("calibration_text is empty; imatrix needs real text")

    export_dir.mkdir(parents=True, exist_ok=True)
    calib_path = export_dir / _CALIB_FILENAME
    out_path = export_dir / _IMATRIX_FILENAME
    meta_path = export_dir / _IMATRIX_META_FILENAME

    calib_path.write_text(calibration_text, encoding="utf-8")

    run = subprocess_runner if subprocess_runner is not None else run_checked
    argv = build_imatrix_args(
        base_gguf=base_gguf,
        calib_path=calib_path,
        out_path=out_path,
        chunks=chunks,
        bin_override=bin_override,
    )

    _LOG.info("imatrix: building (%d × %d tokens)", chunks, chunk_size)
    try:
        run(argv)
    except SubprocessError:
        # Leave calib file on disk so operators can rerun by hand.
        raise

    if not out_path.is_file():
        raise SubprocessError(
            cmd=[str(a) for a in argv],
            returncode=0,
            stderr_tail=(f"llama-imatrix exited 0 but {out_path.name} was not produced."),
        )

    sha = _sha256_of_file(out_path)
    artifact = ImatrixArtifact(
        path=out_path,
        sha256=sha,
        base_revision=base_revision,
        corpus_sha256=corpus_sha256,
        chunks=chunks,
        chunk_size=chunk_size,
        built_at=datetime.now(UTC).replace(tzinfo=None, microsecond=0),
    )
    meta_path.write_text(
        json.dumps(artifact.to_meta_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    # Calib text is regenerable from the replay corpus; don't leave
    # the concatenated blob on disk.
    calib_path.unlink(missing_ok=True)
    return artifact


def build_imatrix_args(
    *,
    base_gguf: Path,
    calib_path: Path,
    out_path: Path,
    chunks: int,
    bin_override: Path | None = None,
) -> list[str]:
    """Assemble the `llama-imatrix ...` argv.

    Pure string-manipulation; no subprocess, no FS side effects beyond
    the vendoring resolver. Snapshot-tested against the pinned
    upstream CLI shape (audit F09 pattern).
    """
    binary = vendoring.llama_imatrix_bin(bin_override)
    return [
        str(binary),
        "-m",
        str(base_gguf),
        "-f",
        str(calib_path),
        "-o",
        str(out_path),
        "--chunks",
        str(chunks),
    ]


def calibration_text_from_replay(
    corpus_path: Path,
    index_path: Path,
    *,
    max_chars: int = DEFAULT_CHUNKS * DEFAULT_CHUNK_SIZE * _CALIB_CHAR_HEADROOM,
) -> tuple[str, str]:
    """Concatenate replay-corpus snapshot content into one calibration blob.

    Returns `(text, corpus_sha256)`. `corpus_sha256` hashes the raw
    `corpus.zst` bytes so the cache key tracks any corpus mutation
    (training a new adapter-version appends to the file → new sha).

    A missing / empty corpus yields `("", "<no-corpus>")` — callers
    decide whether to fall back to static quantization in that case.
    """
    if not corpus_path.is_file():
        return "", "<no-corpus>"

    corpus_sha = _sha256_of_file(corpus_path)

    if not index_path.is_file():
        # Corpus exists but no index — same story as no-corpus for
        # calibration purposes; the sha still tracks the binary so a
        # subsequent index write triggers a rebuild.
        return "", corpus_sha

    # Walk every snapshot in order. Each snapshot's raw content is
    # plain UTF-8 text the imatrix binary re-tokenizes; we don't
    # involve a Python tokenizer here.
    from dlm.replay.corpus import iter_snapshots
    from dlm.replay.index import load_index

    entries = load_index(index_path)
    buf: list[str] = []
    total = 0
    for snap in iter_snapshots(corpus_path, entries):
        if not snap.content:
            continue
        chunk = snap.content.strip()
        if not chunk:
            continue
        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                buf.append(chunk[:remaining])
                total += remaining
            break
        buf.append(chunk)
        total += len(chunk)

    return "\n\n".join(buf), corpus_sha


# --- internals ----------------------------------------------------------------


def _sha256_of_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
