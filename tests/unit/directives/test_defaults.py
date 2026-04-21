"""Default-exclude set — verify curated patterns match expected inputs.

This isn't trying to freeze the list; it's a behavioral pin that the
specific foot-guns the set is meant to catch (VCS, secrets, lockfiles)
stay caught. Regressions here are usually because someone narrowed a
pattern without updating the test.
"""

from __future__ import annotations

import pytest

from dlm.directives.defaults import DEFAULT_EXCLUDES
from dlm.directives.safety import _compile_glob


def _matches_any_default(path: str) -> bool:
    return any(_compile_glob(p).fullmatch(path) is not None for p in DEFAULT_EXCLUDES)


@pytest.mark.parametrize(
    "path",
    [
        # VCS
        ".git/HEAD",
        ".git/objects/aa/bb",
        ".hg/store/data/foo",
        ".svn/entries",
        # Secrets / local config
        ".env",
        ".env.production",
        "src/.env",
        "src/.env.local",
        "home/user/id_rsa",
        "home/user/id_ed25519",
        "configs/production.pem",
        "configs/service.key",
        "app/secrets.toml",
        # Python artifacts
        "src/__pycache__/foo.pyc",
        "a/b/__pycache__/c.pyo",
        ".venv/lib/python/site-packages/x.py",
        "venv/bin/activate",
        # Node
        "node_modules/react/index.js",
        "dist/bundle.min.js",
        "dist/styles.min.css",
        "dist/app.map",
        # Rust / compiled
        "target/release/app",
        "artifacts/libx.rlib",
        "build/class/App.class",
        "build/lib/app.jar",
        "src/libx.so",
        "src/libx.dylib",
        "src/app.dll",
        # Build / dist
        "build/output.txt",
        "dist/index.html",
        "__generated__/proto.py",
        "generated/api_client.py",
        # Lockfiles
        "package-lock.json",
        "subdir/yarn.lock",
        "Cargo.lock",
        "uv.lock",
        "poetry.lock",
        "Pipfile.lock",
        # Media / binaries
        "assets/logo.png",
        "docs/diagram.pdf",
        "release.zip",
        "assets/clip.wasm",
        # dlm config
        ".dlm/training.yaml",
        "src/.dlm/ignore",
    ],
)
def test_default_excludes_catch_known_traps(path: str) -> None:
    assert _matches_any_default(path), f"DEFAULT_EXCLUDES did not catch: {path}"


@pytest.mark.parametrize(
    "path",
    [
        "src/main.py",
        "docs/guide.md",
        "Makefile",
        "README.md",
        "src/utils/helpers.rs",
        "lib/auth.go",
        "app/models/user.java",  # source Java, not compiled
    ],
)
def test_default_excludes_leave_source_alone(path: str) -> None:
    assert not _matches_any_default(path), (
        f"DEFAULT_EXCLUDES wrongly caught: {path}"
    )
