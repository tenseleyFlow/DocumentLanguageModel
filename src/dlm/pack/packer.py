"""Assemble a `.dlm.pack` tarball (Sprint 14).

Public entry: `pack(dlm_path, *, out, include_exports, include_base,
include_logs, licensee_acceptance_url)` → `Path`.

Flow:

1. Parse the `.dlm` and resolve its `StorePath`.
2. Acquire the store's exclusive lock so no concurrent train/export
   tears a checkpoint mid-copy.
3. Copy the store tree (minus excluded subdirs) into a tmp staging
   directory, alongside the `.dlm` file and a `PACK_HEADER.json`.
4. Walk the staging dir, write `CHECKSUMS.sha256`, and compute the
   rollup into `PackManifest.content_sha256`.
5. Tar the staging dir under a zstd-10 stream into `out`.
6. Release the lock.

`include_base` on a `BaseModelSpec.redistributable=False` spec
refuses unless `licensee_acceptance_url` is supplied; the URL goes
into `PackHeader.licensee_acceptance_url` so `dlm show` of the pack
(future) can surface the provenance. Audit F21.

Pack-local zstd discipline (audit F18): the outer stream is zstd-10,
but individual pre-compressed blobs inside the store (`replay/
corpus.zst`, `*.gguf`) are already compressed — recompressing them
wastes CPU and inflates output. We let the outer stream pass them
through; zstd's compression on already-compressed input is near no-op
but the CPU cost is amortized over small metadata files.
"""

from __future__ import annotations

import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from dlm.pack.errors import BaseLicenseRefusedError
from dlm.pack.format import (
    CURRENT_PACK_FORMAT_VERSION,
    ContentType,
    PackHeader,
    PackManifest,
)
from dlm.pack.integrity import rollup_sha256, write_checksums
from dlm.pack.layout import (
    DLM_DIR,
    HEADER_FILENAME,
    MANIFEST_FILENAME,
    SHA256_FILENAME,
    STORE_DIR,
)

if TYPE_CHECKING:
    from dlm.store.paths import StorePath


@dataclass(frozen=True)
class PackResult:
    """Return value of `pack()` — what the CLI prints on success."""

    path: Path
    bytes_written: int
    content_type: str
    applied_licensee_url: str | None


def pack(
    dlm_path: Path,
    *,
    out: Path | None = None,
    include_exports: bool = False,
    include_base: bool = False,
    include_logs: bool = False,
    licensee_acceptance_url: str | None = None,
) -> PackResult:
    """Build a `.dlm.pack` from a `.dlm` and its store.

    `out` defaults to `<dlm_path>.pack` next to the source file. The
    store lock is held for the duration of the copy + tar phases;
    release happens before compression finalization.

    Raises `BaseLicenseRefusedError` if `include_base=True` targets a
    non-redistributable spec and `licensee_acceptance_url` is empty.
    """
    from dlm import __version__ as dlm_version
    from dlm.base_models import BASE_MODELS
    from dlm.doc.parser import parse_file
    from dlm.store.lock import exclusive
    from dlm.store.manifest import load_manifest
    from dlm.store.paths import for_dlm

    parsed = parse_file(dlm_path)
    store = for_dlm(parsed.frontmatter.dlm_id)
    manifest = load_manifest(store.manifest)

    # Gate: `--include-base` on non-redistributable specs without
    # a licensee URL refuses (audit F21).
    spec = BASE_MODELS.get(manifest.base_model)
    if (
        include_base
        and spec is not None
        and not spec.redistributable
        and not licensee_acceptance_url
    ):
        raise BaseLicenseRefusedError(
            base_key=manifest.base_model,
            license_url=spec.license_url,
        )

    out_path = out if out is not None else dlm_path.with_suffix(dlm_path.suffix + ".pack")

    with (
        exclusive(store.lock),
        tempfile.TemporaryDirectory(prefix="dlm-pack-") as tmp,
    ):
        staging = Path(tmp)
        _stage_tree(
            staging=staging,
            dlm_path=dlm_path,
            store=store,
            include_exports=include_exports,
            include_base=include_base,
            include_logs=include_logs,
        )

        content_type = _content_type(
            include_base=include_base,
            include_exports=include_exports,
            include_logs=include_logs,
        )
        header = PackHeader(
            pack_format_version=CURRENT_PACK_FORMAT_VERSION,
            created_at=datetime.now(UTC).replace(tzinfo=None, microsecond=0),
            tool_version=dlm_version,
            content_type=content_type,
            platform_hint=_platform_hint(),
            licensee_acceptance_url=licensee_acceptance_url,
        )
        (staging / HEADER_FILENAME).write_text(
            header.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

        # Checksums cover everything except itself and the manifest
        # (which we write after checksums so the rollup includes only
        # the real content, not the manifest's own sha256-of-sha256s).
        checksums = write_checksums(staging, exclude=(SHA256_FILENAME, MANIFEST_FILENAME))
        # `content_sha256` further excludes the header: header carries
        # `created_at` / `tool_version` / `platform_hint`, which vary
        # across hosts and invocations. Rolling them into the content
        # hash would make two packs of identical user content produce
        # different rollups and defeat byte-identical reproducibility
        # (audit-04 B5). CHECKSUMS.sha256 still covers the header file
        # for tamper detection.
        content_checksums = {k: v for k, v in checksums.items() if k != HEADER_FILENAME}
        manifest_model = PackManifest(
            dlm_id=parsed.frontmatter.dlm_id,
            base_model=manifest.base_model,
            base_model_revision=manifest.base_model_revision,
            base_model_sha256=manifest.base_model_sha256,
            adapter_version=manifest.adapter_version,
            entries={rel: (staging / rel).stat().st_size for rel in checksums},
            content_sha256=rollup_sha256(content_checksums),
        )
        (staging / MANIFEST_FILENAME).write_text(
            manifest_model.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

        _write_tar_zstd(staging, out_path)

    return PackResult(
        path=out_path,
        bytes_written=out_path.stat().st_size,
        content_type=content_type,
        applied_licensee_url=licensee_acceptance_url,
    )


# --- internals --------------------------------------------------------------


def _platform_hint() -> str:
    """`<os>-<accelerator>` — more informative than `sys.platform` alone.

    `sys.platform` collapses both "CUDA Linux" and "CPU-only Linux" to
    `"linux"`, which loses the information `dlm doctor` on the receiving
    host most needs: can this host reasonably resume training? Audit-04
    N2. Falls back to `sys.platform` if the hardware detector can't
    import torch (packaging context without torch installed).
    """
    try:
        from dlm.hardware.backend import detect
    except ImportError:  # pragma: no cover — dev env has torch
        return sys.platform
    try:
        backend = detect().value
    except Exception:
        backend = "unknown"
    return f"{sys.platform}-{backend}"


def _content_type(*, include_base: bool, include_exports: bool, include_logs: bool) -> ContentType:
    """Coarse label for `PackHeader.content_type`."""
    _ = include_logs  # logs don't flip the coarse label
    if include_base and include_exports:
        return "full"
    if include_base:
        return "no-exports"
    if include_exports:
        return "no-base"
    return "minimal"


def _stage_tree(
    *,
    staging: Path,
    dlm_path: Path,
    store: StorePath,
    include_exports: bool,
    include_base: bool,
    include_logs: bool,
) -> None:
    """Copy the `.dlm` + selected store subtrees into `staging/`."""
    # `.dlm` file lives at `dlm/<name>.dlm` inside the pack.
    dlm_dst = staging / DLM_DIR
    dlm_dst.mkdir(parents=True)
    shutil.copy2(dlm_path, dlm_dst / dlm_path.name)

    store_dst = staging / STORE_DIR
    store_dst.mkdir(parents=True)

    # Required tree: everything in the store root except exports/,
    # cache/, and logs/ (which are conditional). Walk shallow to
    # decide per-top-level entry; recurse for those we keep.
    for child in sorted(store.root.iterdir()):
        relpath = child.name
        if relpath == "exports" and not include_exports:
            continue
        if relpath == "cache" and not include_base:
            continue
        if relpath == "logs" and not include_logs:
            continue
        if relpath == store.lock.name:
            # The lockfile itself is state-of-this-process only.
            continue
        dest = store_dst / relpath
        if child.is_dir():
            shutil.copytree(child, dest, symlinks=False)
        else:
            shutil.copy2(child, dest)


def _normalize_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Strip host-specific metadata so two packs of identical content match byte-for-byte.

    Default `TarInfo` carries the file's `mtime`, owner `uid/gid/uname/
    gname` — all of which vary across hosts and invocations. For
    byte-identical packs (audit-04 B5), we zero them and pin `mode` to
    a canonical 0o644 (files) / 0o755 (dirs). Content bytes stay
    intact; only the tar entry *header* is normalized.
    """
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    if info.isdir():
        info.mode = 0o755
    else:
        info.mode = 0o644
    return info


def _write_tar_zstd(staging: Path, out_path: Path) -> None:
    """Tar the staging dir and stream through zstd level 10 to `out_path`."""
    import zstandard as zstd

    cctx = zstd.ZstdCompressor(level=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        out_path.open("wb") as fh,
        cctx.stream_writer(fh) as compressor,
        tarfile.open(fileobj=compressor, mode="w|") as tar,
    ):
        # Entries sorted so the tar stream is deterministic for
        # identical inputs — a property the DoD ratio test relies on.
        for path in sorted(staging.rglob("*")):
            arcname = path.relative_to(staging).as_posix()
            tar.add(path, arcname=arcname, recursive=False, filter=_normalize_tarinfo)
