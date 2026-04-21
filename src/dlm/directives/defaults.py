"""Curated default-exclude patterns applied unless opted out.

Everyone training on a codebase tree wants the same starter set gone:
VCS metadata, secrets, build artifacts, lockfiles, binaries. Shipping
these as defaults removes the "why is my adapter memorizing
package-lock.json" foot-gun from the common path.

`training.yaml` can opt out per-subtree via `exclude_defaults: false`
when a tree legitimately wants to train on (e.g.) generated code or
the `.git` dir itself. Defaults are a *starting point*, not a security
boundary — users with real secrets still need their own excludes.
"""

from __future__ import annotations

from typing import Final

DEFAULT_EXCLUDES: Final[tuple[str, ...]] = (
    # Version control metadata
    ".git/**",
    ".hg/**",
    ".svn/**",
    # Secrets / local config (best-effort; not a security boundary —
    # users with actual secrets need an explicit exclude list).
    ".env",
    ".env.*",
    "**/.env",
    "**/.env.*",
    "**/id_rsa",
    "**/id_ed25519",
    "**/*.pem",
    "**/*.key",
    "**/secrets.*",
    # Python artifacts
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    ".venv/**",
    "venv/**",
    ".tox/**",
    # Node / JS artifacts
    "node_modules/**",
    "**/*.min.js",
    "**/*.min.css",
    "**/*.map",
    # Rust / Go / Java / C / C++ compiled output
    "target/**",
    "**/*.rlib",
    "**/*.class",
    "**/*.jar",
    "**/*.o",
    "**/*.so",
    "**/*.dylib",
    "**/*.dll",
    # Build / dist trees
    "build/**",
    "dist/**",
    "__generated__/**",
    "generated/**",
    # Lockfiles — long, low training signal, noisy to diff
    "**/package-lock.json",
    "**/yarn.lock",
    "**/pnpm-lock.yaml",
    "**/Cargo.lock",
    "**/uv.lock",
    "**/poetry.lock",
    "**/Pipfile.lock",
    # Archives + non-image binaries. Image extensions (png/jpg/jpeg/gif/
    # webp/bmp/tiff) are intentionally *not* excluded — schema v10 made
    # them first-class training material via `SectionType.IMAGE`, and
    # `expand_sources` refuses to ingest them unless the caller supplies
    # a `BlobStore`. Text-only workflows that pass no blob store still
    # see zero images because the walker tallies them under
    # `skipped_image_no_store` instead of emitting Sections.
    "**/*.ico",
    "**/*.pdf",
    "**/*.zip",
    "**/*.tar",
    "**/*.gz",
    "**/*.xz",
    "**/*.bz2",
    "**/*.7z",
    "**/*.wasm",
    # dlm's own config — don't train on the training config
    ".dlm/**",
    "**/.dlm/**",
)
