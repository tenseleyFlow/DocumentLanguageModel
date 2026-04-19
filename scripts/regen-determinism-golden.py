#!/usr/bin/env python3
"""Regenerate Sprint 15's determinism goldens (audit F19).

Goldens are keyed by the runtime tuple
`(torch, transformers, peft, trl, bitsandbytes, platform_tag)`. Changing
any pinned version invalidates the recorded adapter SHA; the operator
must re-run this script after deliberate review to refresh the golden.

Flow:

1. Sample the current runtime versions via `capture_runtime_versions()`.
2. Train SmolLM2-135M from scratch twice against the same seed/doc.
3. Hash each run's `adapter_model.safetensors`; assert they match.
4. Write `tests/golden/determinism/<tuple>.json` with:
   - `adapter_sha256` — the matching hash
   - `pinned_versions` — the runtime tuple
   - `regenerated_at` — UTC timestamp
   - `dlm_sha256` — hash of the synthetic training doc (reproducible
     across runs when the factory's ULID seed is pinned)
5. Compare against the prior golden (if one existed) and print a diff.
6. Exit non-zero unless `--approve` is passed. The default is
   dry-run-and-report so a stray script invocation doesn't silently
   overwrite a baseline.

Usage:
    uv run python scripts/regen-determinism-golden.py           # dry run
    uv run python scripts/regen-determinism-golden.py --approve # write

The matching root-level `dlm.lock` (distinct from the per-store
`dlm.lock`) records which tuples have a checked-in golden. CI computes
the current golden and fails iff that lock asserts a tuple has a
golden but the on-disk file differs (catches silent drift on dep bump).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GOLDEN_DIR = _REPO_ROOT / "tests" / "golden" / "determinism"
_SYNTHETIC_DLM_ID = "01HRDGOLDEN" + "0" * 15  # 26 chars — stable across runs
_SEED = 42
_MAX_STEPS = 20


def _tuple_filename(versions: dict[str, str]) -> str:
    """Produce a filesystem-safe key from the version tuple.

    Ordering matters for determinism — keys are sorted so a reorder
    in `pinned_versions` doesn't produce a different filename.
    """
    parts = [f"{k}={versions[k]}" for k in sorted(versions)]
    parts.append(f"platform={platform.system().lower()}-{platform.machine()}")
    raw = "|".join(parts)
    # Keep the filename short + avoid shell-unfriendly characters.
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"tuple-{digest}.json"


def _hash_adapter(adapter_dir: Path) -> str:
    target = adapter_dir / "adapter_model.safetensors"
    if not target.is_file():
        raise FileNotFoundError(f"adapter weights missing: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_training(home: Path) -> Path:
    """Run one fresh training cycle under `home`. Return the adapter dir."""
    import os

    os.environ["DLM_HOME"] = str(home)

    from tests.fixtures.dlm_factory import make_dlm

    from dlm.base_models import resolve as resolve_base_model
    from dlm.doc.parser import parse_file
    from dlm.hardware import doctor
    from dlm.store.paths import for_dlm
    from dlm.train import run as run_training

    doc_path = home / "determinism.dlm"
    doc_path.write_text(
        make_dlm(base_model="smollm2-135m", dlm_id=_SYNTHETIC_DLM_ID),
        encoding="utf-8",
    )

    parsed = parse_file(doc_path)
    spec = resolve_base_model(parsed.frontmatter.base_model)
    plan = doctor().plan
    if plan is None:
        raise RuntimeError("doctor() returned no viable plan on this host")

    store = for_dlm(parsed.frontmatter.dlm_id)
    store.ensure_layout()

    run_training(
        store,
        parsed,
        spec,
        plan,
        mode="fresh",
        seed=_SEED,
        max_steps=_MAX_STEPS,
        lock_mode="update",  # we're deliberately (re)baselining
    )
    adapter = store.resolve_current_adapter()
    if adapter is None:
        raise RuntimeError("training finished but no current adapter is set")
    return adapter


def _current_versions() -> dict[str, str]:
    from dlm.train.state_sidecar import capture_runtime_versions

    versions = capture_runtime_versions()
    # Filter to str-valued keys so the output is stable across runs that
    # happen to lack an optional dep.
    return {k: v for k, v in versions.items() if isinstance(v, str)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Write the refreshed golden instead of dry-running.",
    )
    args = parser.parse_args()

    import tempfile

    versions = _current_versions()
    filename = _tuple_filename(versions)
    target = _GOLDEN_DIR / filename
    prior = None
    if target.is_file():
        try:
            prior = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior = None

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        adapter_a = _run_training(Path(a))
        sha_a = _hash_adapter(adapter_a)
        adapter_b = _run_training(Path(b))
        sha_b = _hash_adapter(adapter_b)

    if sha_a != sha_b:
        print(
            f"[ERROR] determinism broken: run-A={sha_a} run-B={sha_b}",
            file=sys.stderr,
        )
        return 2

    payload: dict[str, Any] = {
        "adapter_sha256": sha_a,
        "pinned_versions": versions,
        "platform": f"{platform.system().lower()}-{platform.machine()}",
        "regenerated_at": datetime.now(UTC).replace(tzinfo=None, microsecond=0).isoformat(),
        "dlm_id": _SYNTHETIC_DLM_ID,
        "seed": _SEED,
        "max_steps": _MAX_STEPS,
    }

    print(f"[ok] tuple determinism confirmed: adapter_sha={sha_a[:16]}…")
    if prior is not None:
        prior_sha = prior.get("adapter_sha256")
        if prior_sha != sha_a:
            print(
                f"[diff] prior={prior_sha} current={sha_a} delta=change",
                file=sys.stderr,
            )
        else:
            print("[diff] no change from prior golden")

    if not args.approve:
        print(
            f"[dry-run] pass --approve to write {target.relative_to(_REPO_ROOT)}",
        )
        return 1 if prior is None or prior.get("adapter_sha256") != sha_a else 0

    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[wrote] {target.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
