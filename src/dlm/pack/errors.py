"""Typed errors for the `.dlm.pack` pipeline.

Each error class maps to a distinct CLI remediation message so the
reporter (`dlm.cli.reporter`) can route them to useful
one-liners without scraping exception text.
"""

from __future__ import annotations


class PackError(Exception):
    """Base for all `dlm.pack` errors."""


class PackFormatVersionError(PackError):
    """`PACK_HEADER.pack_format_version` is newer than this tool supports.

    Remediation: upgrade `dlm` or ask the pack author for a re-export.
    Carries detected + supported versions so the CLI can render a
    precise message.
    """

    def __init__(self, detected: int, supported: int) -> None:
        super().__init__(
            f"pack_format_version {detected} is newer than this tool's "
            f"supported version {supported}. Upgrade dlm."
        )
        self.detected = detected
        self.supported = supported


class PackIntegrityError(PackError):
    """`CHECKSUMS.sha256` disagrees with the actual file content.

    Either the pack was tampered with or the transfer corrupted it —
    both cases refuse to install. Carries the first offending relative
    path for diagnostics.
    """

    def __init__(self, relpath: str, expected: str, actual: str) -> None:
        super().__init__(
            f"checksum mismatch for {relpath}: expected {expected[:12]}…, got {actual[:12]}…"
        )
        self.relpath = relpath
        self.expected = expected
        self.actual = actual


class PackLayoutError(PackError):
    """Pack tarball is missing a required top-level entry.

    Required entries are listed in `dlm.pack.layout.REQUIRED_ENTRIES`.
    A valid pack has `PACK_HEADER.json`, `manifest.json`, `dlm/<name>.dlm`,
    and `CHECKSUMS.sha256` at minimum.
    """


class BaseLicenseRefusedError(PackError):
    """`pack --include-base` on a non-redistributable spec without the licensee flag.

    The `BaseModelSpec.redistributable=False` gate requires the user to
    pass `--i-am-the-licensee <url>` acknowledging they have
    separate acceptance with the upstream, and the URL is recorded in
    the pack header for provenance. Refusal message includes the
    license_url from the spec.
    """

    def __init__(self, base_key: str, license_url: str | None) -> None:
        suffix = f" Review at {license_url}." if license_url else ""
        super().__init__(
            f"base model {base_key!r} is non-redistributable; "
            "pass --i-am-the-licensee <url> with your acceptance URL to "
            f"include it in the pack.{suffix}"
        )
        self.base_key = base_key
        self.license_url = license_url
