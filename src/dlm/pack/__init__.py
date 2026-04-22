"""Portable `.dlm.pack` bundles — pack + unpack.

One `.dlm` + its store compressed into a single file for sharing,
backup, and reproducibility. The container is a zstd-compressed tarball
with a typed header, a Pydantic manifest, and a `CHECKSUMS.sha256` file
that covers every packed entry.

Public surface:

- `pack(dlm_path, out, options)` → `Path` written
- `unpack(pack_path, home, force)` → `Path` of restored `.dlm`
- `PackHeader`, `PackManifest` — typed models
- `PackError` hierarchy for CLI error mapping

Heavy imports (`tarfile`, `zstandard`) stay deferred to call sites so
`import dlm.pack` stays cheap in fast-path CLI invocations.
"""

from __future__ import annotations

from dlm.pack.errors import (
    BaseLicenseRefusedError,
    PackError,
    PackExecutableFileError,
    PackFormatVersionError,
    PackIntegrityError,
    PackLayoutError,
)
from dlm.pack.format import (
    CURRENT_PACK_FORMAT_VERSION,
    PackHeader,
    PackManifest,
)
from dlm.pack.layout import (
    DLM_DIR,
    HEADER_FILENAME,
    MANIFEST_FILENAME,
    SHA256_FILENAME,
    STORE_DIR,
)

__all__ = [
    "BaseLicenseRefusedError",
    "CURRENT_PACK_FORMAT_VERSION",
    "DLM_DIR",
    "HEADER_FILENAME",
    "MANIFEST_FILENAME",
    "PackError",
    "PackExecutableFileError",
    "PackFormatVersionError",
    "PackHeader",
    "PackIntegrityError",
    "PackLayoutError",
    "PackManifest",
    "SHA256_FILENAME",
    "STORE_DIR",
]
