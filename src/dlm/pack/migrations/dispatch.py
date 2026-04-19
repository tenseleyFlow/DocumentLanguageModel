"""Pack-migration dispatcher — walk an unpacked tree to CURRENT_PACK_FORMAT_VERSION.

Invoked by `dlm.pack.unpacker.unpack` after checksum verification and
before the atomic install step. Mirrors the structure of
`dlm.doc.migrations.dispatch.apply_pending` but keyed on
`pack_format_version` and operating on a filesystem tree rather than a
YAML dict.

Returns the (possibly new) root path and the list of applied
from-versions. The unpacker installs whichever root the dispatcher
hands back, so migrators are free to relocate the tree.
"""

from __future__ import annotations

from pathlib import Path

from dlm.pack.errors import PackFormatVersionError
from dlm.pack.format import CURRENT_PACK_FORMAT_VERSION
from dlm.pack.migrations import PACK_MIGRATORS


def apply_pending(root: Path, *, from_version: int) -> tuple[Path, list[int]]:
    """Walk the pack tree at `root` up to `CURRENT_PACK_FORMAT_VERSION`.

    - Same-version → `(root, [])` no-op.
    - Newer-than-current → `PackFormatVersionError` (the unpacker's
      refuse-to-install gate; normally this is caught earlier by
      reading the header, but the dispatcher enforces it again for
      callers that bypass the header check).
    - Older-than-current → chain of registered migrators; raises
      `PackFormatVersionError` on a gap in the registry.
    """
    if from_version > CURRENT_PACK_FORMAT_VERSION:
        raise PackFormatVersionError(
            detected=from_version, supported=CURRENT_PACK_FORMAT_VERSION
        )

    applied: list[int] = []
    current_root = root
    current_version = from_version
    while current_version < CURRENT_PACK_FORMAT_VERSION:
        migrator = PACK_MIGRATORS.get(current_version)
        if migrator is None:
            # A gap here is structurally the same failure class as a
            # newer-than-supported version — the tool can't bridge this
            # pack forward. Reuse the same error type for one CLI-reporter
            # mapping.
            raise PackFormatVersionError(
                detected=from_version, supported=CURRENT_PACK_FORMAT_VERSION
            )
        current_root = migrator(current_root)
        applied.append(current_version)
        current_version += 1
    return current_root, applied
