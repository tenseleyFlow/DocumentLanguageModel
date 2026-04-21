"""Push orchestrator — pack (if needed) → dispatch to the resolved sink.

Input is a `.dlm` document (which we pack) or an already-packed
`.dlm.pack`. Output is the sink-specific upload confirmation; the CLI
formats it for the user.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dlm.pack.packer import pack
from dlm.share.errors import ShareError, SinkError
from dlm.share.sinks import SinkKind, SinkSpec, parse_source

_LOG = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None] | None


@dataclass(frozen=True)
class PushResult:
    """Outcome of `push()` — what the CLI prints on success.

    `destination` is the user-facing location. `sink_kind` lets the CLI
    pick the right follow-up hint (e.g. `dlm pull hf:...` for HF).
    `bytes_sent` aids progress reporting.
    """

    destination: str
    sink_kind: SinkKind
    bytes_sent: int
    detail: str = ""


def push(
    source_path: Path,
    destination: str,
    *,
    sign: bool = False,
    include_exports: bool = False,
    include_base: bool = False,
    include_logs: bool = False,
    licensee_acceptance_url: str | None = None,
    progress: ProgressCallback = None,
) -> PushResult:
    """Push `source_path` to `destination`.

    `source_path` may be either a `.dlm` (we'll pack it) or an
    already-packed `.dlm.pack`. `destination` is a source-string
    parsed by `parse_source` (hf:, https:, peer: disallowed for push
    — you don't push TO a peer — and local paths just copy).
    """
    spec = parse_source(destination)
    if spec.kind == SinkKind.PEER:
        raise ShareError(
            "push to peer:// is not supported — `dlm serve` hosts the pack, "
            "the other side pulls. Use `dlm serve <path>` instead."
        )

    pack_path, cleanup = _ensure_pack(
        source_path,
        include_exports=include_exports,
        include_base=include_base,
        include_logs=include_logs,
        licensee_acceptance_url=licensee_acceptance_url,
    )
    try:
        if sign:
            _sign_pack(pack_path)
        return _dispatch_push(pack_path, spec, progress=progress)
    finally:
        cleanup()


def _ensure_pack(
    source_path: Path,
    *,
    include_exports: bool,
    include_base: bool,
    include_logs: bool,
    licensee_acceptance_url: str | None,
) -> tuple[Path, Callable[[], None]]:
    """Return a usable `.dlm.pack` path + a cleanup callable.

    If the input is already a pack, we return it as-is with a no-op
    cleanup. If it's a `.dlm`, we pack into a temp file and the
    cleanup deletes it.
    """
    if source_path.suffix == ".pack" or source_path.name.endswith(".dlm.pack"):
        return source_path, _noop

    # It's a `.dlm` — pack into a temp file.
    tmp_dir = Path(tempfile.mkdtemp(prefix="dlm-push-"))
    tmp_pack = tmp_dir / f"{source_path.stem}.dlm.pack"
    result = pack(
        source_path,
        out=tmp_pack,
        include_exports=include_exports,
        include_base=include_base,
        include_logs=include_logs,
        licensee_acceptance_url=licensee_acceptance_url,
    )

    def _cleanup() -> None:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)

    return result.path, _cleanup


def _sign_pack(pack_path: Path) -> None:
    """Sign CHECKSUMS.sha256 inside the pack. Modifies the pack in place.

    The pack layout puts CHECKSUMS.sha256 inside the tarball; signing
    requires extracting, signing, and repacking. For v1 we take a
    simpler approach: sign the pack file itself and place the
    resulting `.minisig` sidecar next to it. Sinks that can carry an
    extra file (hf, url) upload both; peer mode serves both on
    separate paths. Users pulling inspect for the sidecar; if present
    and verified, trust is elevated.

    TODO (follow-up): embed the signature inside the tarball for
    single-artifact atomicity. For v1, sidecar ships.
    """
    from dlm.share.signing import MinisignNotAvailableError, sign_file

    try:
        sig_path = sign_file(pack_path, comment=f"dlm push {pack_path.name}")
    except MinisignNotAvailableError:
        # Propagate so the CLI can print a clean message with the
        # install hint; we don't silently proceed unsigned when --sign
        # was explicit.
        raise
    _LOG.info("push: signed %s → %s", pack_path, sig_path)


def _dispatch_push(pack_path: Path, spec: SinkSpec, *, progress: ProgressCallback) -> PushResult:
    if spec.kind == SinkKind.HF:
        from dlm.share.hf_sink import push_hf

        readme_fields = _collect_readme_fields(pack_path)
        summary = push_hf(
            pack_path,
            spec.target,
            readme_fields=readme_fields,
            progress=progress,
        )
        return PushResult(
            destination=f"hf:{spec.target}",
            sink_kind=SinkKind.HF,
            bytes_sent=pack_path.stat().st_size,
            detail=f"pack: {summary.pack_url}",
        )

    if spec.kind == SinkKind.URL:
        from dlm.share.url_sink import push_url

        push_url(pack_path, spec.target, progress=progress)
        return PushResult(
            destination=spec.target,
            sink_kind=SinkKind.URL,
            bytes_sent=pack_path.stat().st_size,
        )

    if spec.kind == SinkKind.LOCAL:
        import shutil

        dest = Path(spec.target).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pack_path, dest)
        return PushResult(
            destination=str(dest),
            sink_kind=SinkKind.LOCAL,
            bytes_sent=dest.stat().st_size,
        )

    raise SinkError(f"push: unsupported sink kind {spec.kind!r}")


def _collect_readme_fields(pack_path: Path) -> dict[str, str]:
    """Peek inside the pack header for README autopopulation fields.

    Best-effort — if the pack is malformed the upload still happens,
    the README just says "(unknown)". The pack's own integrity check
    will catch real corruption at pull time.
    """
    try:
        import tarfile

        import zstandard as zstd

        fields: dict[str, str] = {}
        with (
            pack_path.open("rb") as f,
            zstd.ZstdDecompressor().stream_reader(f) as r,
            tarfile.open(fileobj=r, mode="r|") as tar,
        ):
            for member in tar:
                if member.name.endswith("header.json"):
                    import json

                    data = tar.extractfile(member)
                    if data is not None:
                        header = json.loads(data.read().decode("utf-8"))
                        fields["dlm_id"] = str(header.get("dlm_id", ""))
                        fields["base_model"] = str(header.get("base_model", ""))
                        fields["adapter_version"] = str(header.get("adapter_version", ""))
                    break
        return fields
    except (OSError, ValueError, ImportError) as exc:
        _LOG.warning("push: couldn't read pack header for README: %s", exc)
        return {}


def _noop() -> None:
    pass
