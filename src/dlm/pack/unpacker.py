"""Install a `.dlm.pack` into the local store (Sprint 14).

Public entry: `unpack(pack_path, *, home, force, out_dir)` → `Path`
(the restored `.dlm` file).

Flow:

1. Decompress the zstd stream + untar into a temp staging directory.
2. Assert layout — every entry in `REQUIRED_ENTRIES` is present.
3. Verify `CHECKSUMS.sha256` against file contents (tamper/corruption
   gate — audit F27).
4. Read `PACK_HEADER.json`; refuse versions newer than this tool
   supports; migrate older-but-known versions via
   `dlm.pack.migrations.dispatch.apply_pending`.
5. Read `manifest.json` to get `dlm_id`; compute the target store
   path at `$DLM_HOME/store/<dlm_id>/`.
6. If the target already exists and `force=False`, refuse.
7. Atomic rename staging `store/` → target store path.
8. Copy `dlm/<name>.dlm` to `out_dir` (or cwd) and return its path.

Missing tarball entries, checksum drift, or unknown pack-format
versions all surface as typed errors the CLI reporter maps to
specific remediation hints.
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dlm.pack.errors import (
    PackFormatVersionError,
    PackIntegrityError,
    PackLayoutError,
)
from dlm.pack.format import CURRENT_PACK_FORMAT_VERSION, PackHeader, PackManifest
from dlm.pack.integrity import verify_checksums
from dlm.pack.layout import (
    DLM_DIR,
    HEADER_FILENAME,
    MANIFEST_FILENAME,
    REQUIRED_ENTRIES,
    STORE_DIR,
)
from dlm.pack.migrations.dispatch import apply_pending

# Defense against zstd bombs: a small stream that claims a huge window
# size decompresses slowly and can exhaust RAM. 512 MiB covers every
# realistic dlm pack member (GGUFs chunk much smaller) while refusing
# pathological inputs. Python's zstandard defaults to 128 MiB already;
# we set it explicitly as a contract.
_MAX_ZSTD_WINDOW_BYTES: Final[int] = 512 * 1024 * 1024

# Defense against tar bombs: cap per-member size and total pack size.
# A single tar entry claiming to be 1 PB is a red flag; the largest
# legitimate member (a multi-gig base GGUF) stays well under 16 GiB.
# 64 GiB total is absurd for a `.dlm` store; flagging it costs nothing.
_MAX_TAR_MEMBER_BYTES: Final[int] = 16 * 1024**3
_MAX_PACK_DECOMPRESSED_BYTES: Final[int] = 64 * 1024**3


@dataclass(frozen=True)
class UnpackResult:
    """Return value of `unpack()` — what the CLI prints on success."""

    dlm_path: Path
    store_path: Path
    dlm_id: str
    applied_migrations: list[int]
    header: PackHeader


def unpack(
    pack_path: Path,
    *,
    home: Path | None = None,
    force: bool = False,
    out_dir: Path | None = None,
) -> UnpackResult:
    """Install `pack_path` into `$DLM_HOME/store/<dlm_id>/` and restore the `.dlm`.

    `home` overrides `$DLM_HOME` (test hook). `out_dir` is where the
    `.dlm` file is placed; defaults to the parent of `pack_path`.
    `force=True` allows overwriting an existing store (the prior store
    is removed after extraction validates).
    """
    from dlm.store.paths import dlm_home

    out_dir = out_dir or pack_path.parent

    with tempfile.TemporaryDirectory(prefix="dlm-unpack-") as tmp:
        staging = Path(tmp)

        _extract_tar_zstd(pack_path, staging)
        _assert_layout(staging)
        verify_checksums(staging)

        header = _read_header(staging)
        if header.pack_format_version > CURRENT_PACK_FORMAT_VERSION:
            raise PackFormatVersionError(
                detected=header.pack_format_version,
                supported=CURRENT_PACK_FORMAT_VERSION,
            )
        migrated_root, applied = apply_pending(staging, from_version=header.pack_format_version)

        pack_manifest = _read_manifest(migrated_root)
        home_resolved = dlm_home(home)
        target_store = home_resolved / "store" / pack_manifest.dlm_id

        target_store.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-swap install: stage the old store out of the way first
        # so an interrupted move doesn't leave the caller with "no store
        # at all" (audit-04 B6). `.old-<pid>` isolates concurrent
        # unpacks and keeps the original recoverable on step-2 failure.
        quarantine: Path | None = None
        if target_store.exists():
            if not force:
                raise PackIntegrityError(
                    relpath=str(target_store),
                    expected="<not-present>",
                    actual="<exists>",
                )
            import os

            quarantine = target_store.parent / f".{target_store.name}.old-{os.getpid()}"
            if quarantine.exists():
                shutil.rmtree(quarantine)
            target_store.rename(quarantine)

        try:
            shutil.move(str(migrated_root / STORE_DIR), str(target_store))
        except Exception:
            # Roll back: restore the quarantined old store so the caller
            # doesn't see a disappeared store on failure.
            if quarantine is not None and not target_store.exists():
                quarantine.rename(target_store)
                quarantine = None
            raise

        if quarantine is not None:
            shutil.rmtree(quarantine, ignore_errors=True)

        dlm_src = _find_dlm_file(migrated_root / DLM_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        dlm_dst = out_dir / dlm_src.name
        shutil.copy2(dlm_src, dlm_dst)

    return UnpackResult(
        dlm_path=dlm_dst,
        store_path=target_store,
        dlm_id=pack_manifest.dlm_id,
        applied_migrations=applied,
        header=header,
    )


# --- internals --------------------------------------------------------------


def _extract_tar_zstd(pack_path: Path, staging: Path) -> None:
    """Stream-decompress the pack's zstd+tar into `staging/` under DoS bounds.

    Three defenses (audit-04 B7):

    1. `max_window_size` on the zstd decompressor — refuses streams
       that claim to need > 512 MiB decompression state.
    2. Per-member size cap at iteration time — a tar entry claiming
       absurd `size` fails the layout gate before we touch disk.
    3. Total cumulative size cap — sum of member sizes must stay
       under 64 GiB or we refuse the whole archive.

    Bundled with the pre-existing `_is_unsafe_member` traversal guard
    and `filter="data"` on extraction.
    """
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor(max_window_size=_MAX_ZSTD_WINDOW_BYTES)
    # First pass: defense-in-depth name scan + size bounds. `tarfile`
    # iterates once so we open a fresh reader below to actually extract.
    total_size = 0
    # Audit-05 M5: refuse duplicate member names. `tar.extractall`
    # silently overwrites later-with-earlier duplicates, so an attacker
    # who repacks could inject a second `store/manifest.json` that slips
    # past the CHECKSUMS verify (we compute over the final on-disk bytes,
    # so two copies with identical content would pass the old check).
    seen: set[str] = set()
    with (
        pack_path.open("rb") as fh,
        dctx.stream_reader(fh) as reader,
        tarfile.open(fileobj=reader, mode="r|") as tar,
    ):
        for member in tar:
            if _is_unsafe_member(member.name):
                raise PackLayoutError(f"refusing to extract unsafe tar entry {member.name!r}")
            if member.name in seen:
                raise PackLayoutError(
                    f"duplicate tar entry {member.name!r}; pack may have been tampered with"
                )
            seen.add(member.name)
            if member.size > _MAX_TAR_MEMBER_BYTES:
                raise PackLayoutError(
                    f"tar entry {member.name!r} size {member.size} exceeds "
                    f"per-member cap {_MAX_TAR_MEMBER_BYTES}"
                )
            total_size += max(member.size, 0)
            if total_size > _MAX_PACK_DECOMPRESSED_BYTES:
                raise PackLayoutError(
                    f"pack total decompressed size exceeds cap "
                    f"{_MAX_PACK_DECOMPRESSED_BYTES} at entry {member.name!r}"
                )

    # Second pass: actually extract. Python 3.12+ defaults to the `data`
    # filter; we set it explicitly so path traversal + special-file
    # attacks are blocked on earlier versions too.
    with (
        pack_path.open("rb") as fh,
        dctx.stream_reader(fh) as reader,
        tarfile.open(fileobj=reader, mode="r|") as tar,
    ):
        tar.extractall(path=staging, filter="data")


def _is_unsafe_member(name: str) -> bool:
    """Reject absolute paths, drive letters, or `..` escape components."""
    if name.startswith("/") or name.startswith("\\"):
        return True
    parts = Path(name).parts
    return any(part == ".." for part in parts)


def _assert_layout(staging: Path) -> None:
    """Every required entry must be present post-extraction."""
    for entry in REQUIRED_ENTRIES:
        if not (staging / entry).exists():
            raise PackLayoutError(f"pack missing required entry {entry!r}; not a valid .dlm.pack")


def _read_header(staging: Path) -> PackHeader:
    """Parse `PACK_HEADER.json` into a validated `PackHeader`."""
    import json

    try:
        raw = json.loads((staging / HEADER_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackLayoutError(f"cannot read {HEADER_FILENAME}: {exc}") from exc
    return PackHeader.model_validate(raw)


def _read_manifest(staging: Path) -> PackManifest:
    """Parse `manifest.json` into a validated `PackManifest`."""
    import json

    try:
        raw = json.loads((staging / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackLayoutError(f"cannot read {MANIFEST_FILENAME}: {exc}") from exc
    return PackManifest.model_validate(raw)


def _find_dlm_file(dlm_dir: Path) -> Path:
    """Locate the packed `.dlm` source file.

    Spec: exactly one `*.dlm` file under `dlm/`. More than one ⇒
    packer bug (shouldn't happen); zero ⇒ layout error.
    """
    candidates = list(dlm_dir.glob("*.dlm"))
    if len(candidates) != 1:
        raise PackLayoutError(
            f"expected exactly one .dlm file under {DLM_DIR}/, found {len(candidates)}"
        )
    return candidates[0]
