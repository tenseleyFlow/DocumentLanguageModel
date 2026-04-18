"""Golden-output registry for determinism tests.

Goldens are JSON blobs under `tests/golden/` keyed by `(name, torch_version)`
so bumping torch produces a new golden rather than silently invalidating
the old one (audit F19).

Contract:

- `assert_golden(actual, name)` compares `actual` against the stored
  golden; if no golden exists for the current torch version, the test
  fails with an instruction to regenerate.
- `pytest --update-goldens` regenerates instead of asserting; the
  `update_goldens` fixture (`conftest.py`) surfaces the flag.
- Golden content is deterministically serialized JSON (sorted keys,
  normalized floats).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GOLDEN_ROOT = Path(__file__).resolve().parent.parent / "golden"


class MissingGoldenError(AssertionError):
    """Raised when a golden isn't on disk and we're not regenerating."""


def golden_path(name: str, *, torch_version: str | None = None) -> Path:
    """Resolve the on-disk path for a golden by (name, torch_version)."""
    tv = torch_version if torch_version is not None else _current_torch_version()
    # Filesystem-safe: replace periods and pluses.
    safe_tv = tv.replace("/", "_").replace("+", "_").replace(" ", "_")
    safe_name = name.replace("/", "_")
    return GOLDEN_ROOT / f"{safe_name}.torch-{safe_tv}.json"


def load_golden(name: str, *, torch_version: str | None = None) -> Any:
    path = golden_path(name, torch_version=torch_version)
    if not path.exists():
        raise MissingGoldenError(
            f"No golden for {name!r} at {path}. "
            "Run with --update-goldens to regenerate after manual review.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


def assert_golden(
    actual: Any,
    name: str,
    *,
    update: bool = False,
    torch_version: str | None = None,
) -> None:
    """Compare `actual` against stored golden, or regenerate when `update`.

    `actual` must be JSON-serializable. Regeneration writes sorted-keys
    output so diffs are reviewable.
    """
    path = golden_path(name, torch_version=torch_version)

    if update:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, actual)
        return

    if not path.exists():
        raise MissingGoldenError(
            f"No golden for {name!r} at {path}. "
            "Run `pytest --update-goldens` after manual review to create it.",
        )

    expected = json.loads(path.read_text(encoding="utf-8"))
    if _canonical(expected) != _canonical(actual):
        diff_lines = [
            f"Golden mismatch for {name!r} (path: {path}).",
            "Expected:",
            json.dumps(expected, sort_keys=True, indent=2),
            "Actual:",
            json.dumps(actual, sort_keys=True, indent=2),
        ]
        raise AssertionError("\n".join(diff_lines))


# --- internals ---------------------------------------------------------------


def _current_torch_version() -> str:
    # Lazy import so callers that don't use goldens don't pay for torch.
    import torch

    return torch.__version__


def _canonical(value: Any) -> str:
    """Canonical JSON form for comparison."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, sort_keys=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
