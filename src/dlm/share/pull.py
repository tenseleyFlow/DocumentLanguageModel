"""Pull orchestrator — download from sink → verify checksums + signature → unpack.

The pull path is the mirror of push + the pack contract's verification:

1. Parse the source string.
2. Download the pack to a temp location.
3. If a `.minisig` sidecar is available (peer / hf / url), download it too.
4. Verify sha256 checksums via `dlm.pack.unpack` (it re-checksums
   every tar entry before installing).
5. Verify the signature (if present); record the trust status.
6. Install the pack via `dlm.pack.unpack`, which places the `.dlm`
   at `out_dir` and the store at `~/.dlm/store/<dlm_id>/`.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dlm.pack.unpacker import unpack as pack_unpack
from dlm.share.errors import ShareError, SinkError
from dlm.share.signing import VerifyResult, VerifyStatus, verify_signature
from dlm.share.sinks import SinkKind, SinkSpec, parse_source

_LOG = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None] | None


@dataclass(frozen=True)
class PullResult:
    """Outcome of `pull()` — what the CLI prints on success."""

    dlm_path: Path
    store_path: Path
    dlm_id: str
    source: str
    bytes_received: int
    verification: VerifyResult


def pull(
    source: str,
    *,
    out_dir: Path | None = None,
    force: bool = False,
    home: Path | None = None,
    progress: ProgressCallback = None,
) -> PullResult:
    """Pull + verify + unpack a pack from `source`.

    `out_dir` is where the restored `.dlm` lands (default: CWD).
    `force=True` overwrites an existing store with the same `dlm_id`.
    `home` overrides `$DLM_HOME` (test hook; matches `dlm.pack.unpack`).
    """
    spec = parse_source(source)
    out_dir = out_dir or Path.cwd()

    with tempfile.TemporaryDirectory(prefix="dlm-pull-") as tmp:
        staging = Path(tmp)
        pack_path = staging / "incoming.dlm.pack"
        sig_path = pack_path.with_suffix(pack_path.suffix + ".minisig")

        bytes_received = _dispatch_pull(
            spec, pack_path, sig_path, progress=progress
        )

        # Verify signature BEFORE unpack so users learn the trust
        # status even if unpack then fails for an unrelated reason.
        verification = verify_signature(pack_path, sig_path)
        _log_verification(source, verification)

        # Unpack (re-checksums every entry internally).
        unpacked = pack_unpack(
            pack_path,
            home=home,
            force=force,
            out_dir=out_dir,
        )

    return PullResult(
        dlm_path=unpacked.dlm_path,
        store_path=unpacked.store_path,
        dlm_id=unpacked.dlm_id,
        source=source,
        bytes_received=bytes_received,
        verification=verification,
    )


def _dispatch_pull(
    spec: SinkSpec,
    pack_path: Path,
    sig_path: Path,
    *,
    progress: ProgressCallback,
) -> int:
    """Download the pack (+ sig if available) to the staging paths."""
    if spec.kind == SinkKind.HF:
        from dlm.share.hf_sink import pull_hf

        bytes_received = pull_hf(spec.target, pack_path, progress=progress)
        # Try the sig sidecar best-effort; missing is fine (unsigned
        # packs are valid).
        _try_hf_sidecar(spec.target, sig_path)
        return bytes_received

    if spec.kind == SinkKind.URL:
        from dlm.share.url_sink import pull_url

        bytes_received = pull_url(spec.target, pack_path, progress=progress)
        _try_url_sidecar(spec.target, sig_path)
        return bytes_received

    if spec.kind == SinkKind.PEER:
        from dlm.share.peer import pull_peer

        bytes_received = pull_peer(spec.target, pack_path, progress=progress)
        _try_peer_sidecar(spec.target, sig_path)
        return bytes_received

    if spec.kind == SinkKind.LOCAL:
        import shutil

        src = Path(spec.target).expanduser().resolve()
        if not src.is_file():
            raise SinkError(f"local pull: source missing: {src}")
        shutil.copy2(src, pack_path)
        # Look for sibling sig file
        local_sig = src.with_suffix(src.suffix + ".minisig")
        if local_sig.is_file():
            shutil.copy2(local_sig, sig_path)
        return pack_path.stat().st_size

    raise ShareError(f"pull: unsupported sink kind {spec.kind!r}")


def _try_hf_sidecar(repo_id: str, sig_path: Path) -> None:
    """Best-effort fetch of `.minisig` sidecar from HF Hub."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError  # type: ignore[attr-defined]

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename="adapter.dlm.pack.minisig",
            repo_type="model",
        )
        import shutil

        shutil.copy2(downloaded, sig_path)
    except (HfHubHTTPError, ImportError):
        # Sidecar absent = pack is unsigned; no-op.
        pass


def _try_url_sidecar(url: str, sig_path: Path) -> None:
    """Best-effort fetch of `.minisig` sidecar alongside a URL pack."""
    import contextlib

    from dlm.share.url_sink import pull_url

    # Sidecar 404 = pack is unsigned; no-op.
    with contextlib.suppress(SinkError):
        pull_url(url + ".minisig", sig_path)


def _try_peer_sidecar(target: str, sig_path: Path) -> None:
    """Best-effort fetch of `.minisig` sidecar from a peer."""
    import contextlib

    from dlm.share.peer import pull_peer

    # Peer doesn't serve a sidecar = unsigned; no-op.
    with contextlib.suppress(SinkError):
        pull_peer(target + ".minisig", sig_path)


def _log_verification(source: str, result: VerifyResult) -> None:
    if result.status == VerifyStatus.VERIFIED:
        _LOG.info(
            "pull: verified signature from %s using %s", source, result.key_path
        )
    elif result.status == VerifyStatus.UNVERIFIED:
        _LOG.warning(
            "pull: signature present but could not verify (%s). "
            "Pack content integrity still checked via sha256.",
            result.detail,
        )
    else:
        _LOG.info("pull: no signature on %s (unsigned, checksums still verified)", source)
