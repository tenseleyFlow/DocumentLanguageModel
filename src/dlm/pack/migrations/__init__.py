"""Pack-format migrations (Sprint 14, audit F27).

Independent of Sprint 12b's `dlm.doc.migrations` — this registry
targets `pack_format_version` (the `.dlm.pack` container shape),
not `dlm_version` (the `.dlm` frontmatter schema). The contract is
identical in structure so the two read the same way.

Each time `CURRENT_PACK_FORMAT_VERSION` increments, a matching
`src/dlm/pack/migrations/vN.py` module registers a `migrate(root:
Path) -> Path` callable that rewrites the extracted pack tree in
place (or returns a new tmp root) before the unpacker installs it.

Coverage test at `tests/unit/pack/test_migrations.py::
TestCoverageEnforcement::test_migrators_span_required_range` refuses
a PR that bumps `CURRENT_PACK_FORMAT_VERSION` without registering a
matching migrator.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final

# `from_version` → migrator. Populated by `@register` side-effects at
# import time; per-version modules are imported from this file's tail.
PACK_MIGRATORS: Final[dict[int, Callable[[Path], Path]]] = {}


def register(
    from_version: int,
) -> Callable[[Callable[[Path], Path]], Callable[[Path], Path]]:
    """Decorator: bind a migrator to its `from_version` key.

    Double-registration is a programming bug (two modules claim the
    same migration step); raises `AssertionError` at import time so
    the bug surfaces immediately on the CI run that introduced it.
    """

    def decorator(fn: Callable[[Path], Path]) -> Callable[[Path], Path]:
        assert from_version not in PACK_MIGRATORS, (
            f"duplicate pack migrator for from_version={from_version}: "
            f"{PACK_MIGRATORS[from_version].__module__} vs {fn.__module__}"
        )
        PACK_MIGRATORS[from_version] = fn
        return fn

    return decorator


# Side-effect imports — `register` must be defined first.
from dlm.pack.migrations import v1  # noqa: E402, F401
