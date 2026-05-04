"""Helpers shared across more than one cli/commands submodule.

Anything used by exactly one command lives in that command's own
submodule. This file is the explicit destination for true cross-command
helpers — keeping it small means the package's public surface (the
`__init__.py` re-exports) stays focused on commands themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _previously_accepted(store_manifest_path: Path) -> bool:
    """Return True iff the store manifest already holds a LicenseAcceptance.

    `dlm prompt`, `dlm export`, and `dlm repl` operate on an
    already-trained adapter; the gated-base license was accepted during
    training and persisted into `manifest.license_acceptance`. Replaying
    that acceptance here is correct; silently hardcoding
    `accept_license=True` is not — it would let a never-accepted gated
    base slip through.
    """
    if not store_manifest_path.exists():
        return False
    from dlm.store.errors import ManifestCorruptError
    from dlm.store.manifest import load_manifest

    try:
        manifest = load_manifest(store_manifest_path)
    except (ManifestCorruptError, OSError):
        # Narrow from bare `Exception` so programmer bugs propagate
        # instead of being silently treated as "no acceptance."
        return False
    return manifest.license_acceptance is not None


def _human_size(n: int) -> str:
    """Render a byte count as a 1-decimal human string (B / KB / MB / …)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n //= 1024
    return f"{n} PB"
