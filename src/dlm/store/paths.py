"""Path resolver for the content-addressed store.

`dlm_home()` answers "where does `~/.dlm/` live?" with the precedence:

    1. `$DLM_HOME` environment variable (if set).
    2. Platform default:
       - POSIX: `~/.dlm`
       - Windows: `%APPDATA%\\dlm`, else fall back to `~/.dlm`.

`for_dlm(dlm_id)` returns a `StorePath` anchored at
`{home}/store/{dlm_id}/` with typed accessors for every well-known file.

Callers that need CLI `--home` override should resolve it and pass the
value into `dlm_home(override=...)` — env + CLI precedence is enforced
in the CLI layer (Sprint 13).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dlm.store.layout import (
    ADAPTER_CURRENT_POINTER,
    ADAPTER_DIR,
    ADAPTER_VERSIONS_DIR,
    ALWAYS_CREATE_DIRS,
    CACHE_DIR,
    EXPORTS_DIR,
    LOCK_FILENAME,
    LOGS_DIR,
    MANIFEST_FILENAME,
    REPLAY_CORPUS_FILENAME,
    REPLAY_DIR,
    REPLAY_INDEX_FILENAME,
    TRAINING_STATE_FILENAME,
    TRAINING_STATE_SHA_FILENAME,
)

STORE_SUBDIR: Final = "store"

# Mirror the schema/parser grammar — adapter names must be
# path-safe and log-friendly. Fullmatch-enforced here so path helpers
# can't compose a `../escape`-style directory traversal.
_ADAPTER_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def _validate_adapter_name(name: str) -> None:
    if not _ADAPTER_NAME_RE.fullmatch(name):
        raise ValueError(
            f"adapter name {name!r} is not valid "
            f"(must match {_ADAPTER_NAME_RE.pattern})"
        )


def _current_os_name() -> str:
    """Return `os.name`. Indirected through a helper so tests can patch
    the NT vs POSIX branch without globally mutating `os.name` (which
    would break `pathlib` on the host).
    """
    return os.name


def dlm_home(override: Path | str | None = None) -> Path:
    """Resolve the DLM home directory.

    Precedence: explicit `override` → `$DLM_HOME` env var → platform
    default. Does NOT create the directory; use `ensure_home()` for that.
    """
    if override is not None:
        return Path(override).expanduser().resolve()

    env = os.environ.get("DLM_HOME")
    if env:
        return Path(env).expanduser().resolve()

    if _current_os_name() == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata).resolve() / "dlm"
    return Path.home() / ".dlm"


def ensure_home(override: Path | str | None = None) -> Path:
    """Create the DLM home and `store/` subdir if absent; return the home path."""
    home = dlm_home(override)
    (home / STORE_SUBDIR).mkdir(parents=True, exist_ok=True)
    return home


def for_dlm(dlm_id: str, *, home: Path | str | None = None) -> StorePath:
    """Return the `StorePath` for a given `dlm_id`.

    Does NOT check for existence; use `StorePath.exists()` or
    `StorePath.ensure_layout()` as needed.
    """
    if not dlm_id:
        raise ValueError("dlm_id must be non-empty")
    return StorePath(root=dlm_home(home) / STORE_SUBDIR / dlm_id)


@dataclass(frozen=True)
class StorePath:
    """Typed access to every well-known file under a store.

    Construct via `for_dlm(dlm_id)`; never instantiate directly except in
    tests that bypass the home resolver.
    """

    root: Path

    # --- top-level files ----------------------------------------------------

    @property
    def manifest(self) -> Path:
        return self.root / MANIFEST_FILENAME

    @property
    def lock(self) -> Path:
        return self.root / LOCK_FILENAME

    @property
    def training_state(self) -> Path:
        return self.root / TRAINING_STATE_FILENAME

    @property
    def training_state_sha(self) -> Path:
        return self.root / TRAINING_STATE_SHA_FILENAME

    # --- top-level dirs -----------------------------------------------------

    @property
    def adapter(self) -> Path:
        return self.root / ADAPTER_DIR

    @property
    def adapter_versions(self) -> Path:
        return self.adapter / ADAPTER_VERSIONS_DIR

    @property
    def adapter_current_pointer(self) -> Path:
        return self.adapter / ADAPTER_CURRENT_POINTER

    @property
    def replay(self) -> Path:
        return self.root / REPLAY_DIR

    @property
    def replay_corpus(self) -> Path:
        return self.replay / REPLAY_CORPUS_FILENAME

    @property
    def replay_index(self) -> Path:
        return self.replay / REPLAY_INDEX_FILENAME

    @property
    def exports(self) -> Path:
        return self.root / EXPORTS_DIR

    @property
    def cache(self) -> Path:
        return self.root / CACHE_DIR

    @property
    def logs(self) -> Path:
        return self.root / LOGS_DIR

    # --- computed helpers ---------------------------------------------------

    def adapter_version(self, version: int) -> Path:
        """Return `adapter/versions/vNNNN` (does NOT create it).

        Used by the flat single-adapter layout. Multi-adapter documents
        call `adapter_version_for(name, version)` instead.
        """
        if version < 1:
            raise ValueError(f"adapter versions are 1-indexed, got {version}")
        return self.adapter_versions / f"v{version:04d}"

    # --- multi-adapter helpers ---------------------------------------------
    #
    # For documents with `training.adapters`, each named adapter gets its
    # own nested directory tree: `adapter/<name>/{current.txt,versions/}`.
    # The flat methods above remain for single-adapter (default) docs; a
    # given store uses one shape or the other but never both.

    def adapter_dir_for(self, name: str) -> Path:
        """Return `adapter/<name>/` (does NOT create it)."""
        _validate_adapter_name(name)
        return self.adapter / name

    def adapter_versions_for(self, name: str) -> Path:
        """Return `adapter/<name>/versions/` (does NOT create it)."""
        return self.adapter_dir_for(name) / ADAPTER_VERSIONS_DIR

    def adapter_version_for(self, name: str, version: int) -> Path:
        """Return `adapter/<name>/versions/vNNNN/` (does NOT create it)."""
        if version < 1:
            raise ValueError(f"adapter versions are 1-indexed, got {version}")
        return self.adapter_versions_for(name) / f"v{version:04d}"

    def adapter_current_pointer_for(self, name: str) -> Path:
        """Return `adapter/<name>/current.txt` (does NOT create it)."""
        return self.adapter_dir_for(name) / ADAPTER_CURRENT_POINTER

    def ensure_adapter_layout(self, name: str) -> None:
        """Create `adapter/<name>/versions/` on demand. Idempotent."""
        self.adapter_versions_for(name).mkdir(parents=True, exist_ok=True)

    def resolve_current_adapter_for(self, name: str) -> Path | None:
        """Resolve `adapter/<name>/current.txt` to an absolute path, or None."""
        pointer = self.adapter_current_pointer_for(name)
        if not pointer.exists():
            return None
        rel = pointer.read_text(encoding="utf-8").strip()
        if not rel:
            return None
        resolved = (self.root / rel).resolve()
        try:
            resolved.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"adapter pointer {pointer} escapes store root: {rel}",
            ) from exc
        return resolved

    def set_current_adapter_for(self, name: str, version_dir: Path) -> None:
        """Atomically point `adapter/<name>/current.txt` at a version directory."""
        try:
            relative = version_dir.resolve().relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"adapter version {version_dir} is outside store root {self.root}",
            ) from exc
        from dlm.io.atomic import write_text as _atomic_write_text

        _atomic_write_text(
            self.adapter_current_pointer_for(name), f"{relative}\n"
        )

    def export_quant_dir(self, quant: str) -> Path:
        """Return `exports/<quant>/` (does NOT create it)."""
        if not quant:
            raise ValueError("quant must be non-empty")
        return self.exports / quant

    def cache_dir_for(self, slug: str) -> Path:
        """Return `cache/<slug>/` (does NOT create it).

        `slug` is a filesystem-safe identifier; callers are responsible
        for normalizing HF ids before passing.
        """
        if not slug:
            raise ValueError("slug must be non-empty")
        return self.cache / slug

    # --- existence / layout -------------------------------------------------

    def exists(self) -> bool:
        return self.root.exists()

    def ensure_layout(self) -> None:
        """Create the store root and always-on subdirs.

        Idempotent. Subdirs owned by other sprints (replay/, exports/, cache/)
        are created lazily by those sprints on first write.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ALWAYS_CREATE_DIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.adapter_versions.mkdir(parents=True, exist_ok=True)

    def resolve_current_adapter(self) -> Path | None:
        """Resolve `adapter/current.txt` to an absolute path, or None.

        The pointer file contains a single line: the relative path (from
        the store root) of the active adapter version directory. Missing
        or empty pointer files return `None`.
        """
        pointer = self.adapter_current_pointer
        if not pointer.exists():
            return None
        rel = pointer.read_text(encoding="utf-8").strip()
        if not rel:
            return None
        resolved = (self.root / rel).resolve()
        # Safety: the target must be inside the store root.
        try:
            resolved.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"adapter pointer {pointer} escapes store root: {rel}",
            ) from exc
        return resolved

    def set_current_adapter(self, version_dir: Path) -> None:
        """Atomically point `adapter/current.txt` at a version directory.

        The path written is relative to the store root so moving the
        store doesn't orphan the pointer.
        """
        try:
            relative = version_dir.resolve().relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"adapter version {version_dir} is outside store root {self.root}",
            ) from exc
        from dlm.io.atomic import write_text as _atomic_write_text

        _atomic_write_text(self.adapter_current_pointer, f"{relative}\n")
