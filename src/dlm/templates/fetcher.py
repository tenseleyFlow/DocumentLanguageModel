"""Remote template-gallery fetcher — scaffold only.

The sprint spec calls for a remote gallery hosted at
`github.com/<org>/dlm-templates` with `git clone --depth 1` into
`~/.dlm/templates-cache/` and signed-tag verification against a pinned
public key. Neither the upstream repo nor the signing key exists yet
(organization TBD per the sprint text), so this module intentionally
doesn't dial the network.

What lands here:

- `cache_dir()` — returns `~/.dlm/templates-cache/` (used by the CLI
  refresh path once upstream ships).
- `RemoteFetchUnavailable` — a typed error the CLI can raise so the
  offline-first fallback is always the calling pattern.
- `fetch_all()` — documented shape; raises `RemoteFetchUnavailable`
  with a pointer to the deferred-polish section of the sprint file.

Keeping this as a real, typed stub rather than a secret TODO lets the
CLI path (`dlm templates list --refresh`) report a clear error instead
of silently falling back to the bundled set.
"""

from __future__ import annotations

from pathlib import Path

from dlm.templates.errors import TemplateError


class RemoteFetchUnavailable(TemplateError):  # noqa: N818 — existing sentinel, not an unexpected failure
    """Raised by `fetch_all` until the upstream gallery repo is pinned.

    The CLI catches this and falls back to `list_bundled`, so end users
    still see the offline gallery. Callers who genuinely need remote
    fetch get a clear pointer to the sprint file's deferred-polish list.
    """


def cache_dir(home: Path | None = None) -> Path:
    """Return `~/.dlm/templates-cache/` (or a subpath of `home` if given).

    The directory is not created — callers decide when to materialize it.
    """
    root = home if home is not None else Path.home() / ".dlm"
    return root / "templates-cache"


def fetch_all(
    cache_dir_path: Path,  # noqa: ARG001 — reserved for the live implementation
    remote: str,  # noqa: ARG001
) -> None:
    """Refresh the template cache from the pinned upstream gallery.

    Signed-tag verification against the pinned public key is part of the
    contract (sprint DoD). Until the upstream repo is live and the key
    fingerprint is checked in under `src/dlm/templates/upstream_key.pem`,
    this always raises `RemoteFetchUnavailable`.
    """
    raise RemoteFetchUnavailable(
        "remote template gallery fetch is not wired yet — upstream repo "
        "and signing key are pending (Sprint 27 deferred polish).",
    )
