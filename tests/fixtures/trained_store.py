"""Session-scoped fixture that trains a tiny-model store once per test session.

Sprint 14.5 consumers (`tests/integration/train/`, `tests/integration/export/`,
`tests/integration/pack/`) all need "a store with a real trained adapter."
A session fixture amortizes the ~60-90s train cost across the slow-test run.

The fixture uses `dlm.train.run()` directly (not Typer CliRunner) so tests
can treat it as a clean dependency: any `dlm train` CLI regression belongs
in the one-cycle rewrite, not here. The CLI-level round trip is covered
by the pack test which exercises `dlm pack` / `dlm unpack` / `dlm prompt`
through the Typer stack.

Skips gracefully when:
- `torch` or `transformers` aren't importable (e.g. a CPU-only slim runner)
- The tiny-model fixture can't download / isn't cached
- `doctor().plan` returns None (no viable training plan on the host)
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

if TYPE_CHECKING:
    from dlm.store.paths import StorePath

# Small enough to keep wall-clock manageable on CPU (smollm2-135m: ~1s/step
# on a recent Mac, ~0.5s on a CI ubuntu runner). Exports and resumes can
# hash the resulting adapter either way.
_TRAIN_MAX_STEPS: Final[int] = 20
_TRAIN_SEED: Final[int] = 42


@dataclass(frozen=True)
class TrainedStoreHandle:
    """Result of one shared training run.

    `store` is a live `StorePath` (DLM_HOME is set to `home` for the
    fixture's lifetime). `dlm_id` is cached so tests don't need to
    re-parse the document.
    """

    doc: Path
    home: Path
    dlm_id: str
    store: StorePath


@pytest.fixture(scope="session")
def trained_store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TrainedStoreHandle]:
    """Train smollm2-135m once, yield handle with doc/home/store/dlm_id.

    Requires full model weights, not just the tokenizer. On a dev machine
    with a cold cache the first run takes ~2 minutes (download + 20-step
    train). CI's slow-test job pre-warms the HF cache so subsequent runs
    skip the download.
    """
    # Clear the autouse `_offline_hf_env` so snapshot_download / from_pretrained
    # can pull missing model weights. Restored at teardown so downstream
    # fast-path tests see the offline contract again.
    offline_vars = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    saved_env = {k: os.environ.pop(k, None) for k in offline_vars}

    try:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"torch/transformers unavailable: {exc}")

        try:
            from tests.fixtures.tiny_model import tiny_model_path

            tiny_model_path()  # force-resolve the snapshot (may download)
        except Exception as exc:
            pytest.skip(f"tiny-model fixture unavailable: {exc}")

        from dlm.base_models import resolve as resolve_base_model
        from dlm.doc.parser import parse_file
        from dlm.hardware import doctor
        from dlm.store.paths import for_dlm
        from dlm.train import run as run_training
        from tests.fixtures.dlm_factory import make_dlm

        plan = doctor().plan
        if plan is None:
            pytest.skip("doctor() returned no viable training plan on this host")

        home = tmp_path_factory.mktemp("dlm-trained-home")
        os.environ["DLM_HOME"] = str(home)

        doc = home / "smoke.dlm"
        doc.write_text(make_dlm(base_model="smollm2-135m"), encoding="utf-8")

        parsed = parse_file(doc)
        spec = resolve_base_model(parsed.frontmatter.base_model)
        store = for_dlm(parsed.frontmatter.dlm_id)
        store.ensure_layout()

        # Seed the initial manifest — `dlm init` owns this in the CLI
        # path, but the fixture skips `init` and calls `run_training`
        # directly. Missing manifest → ManifestCorruptError on load.
        from dlm.store.manifest import Manifest, save_manifest

        save_manifest(
            store.manifest,
            Manifest(
                dlm_id=parsed.frontmatter.dlm_id,
                base_model=parsed.frontmatter.base_model,
            ),
        )

        run_training(
            store,
            parsed,
            spec,
            plan,
            mode="fresh",
            seed=_TRAIN_SEED,
            max_steps=_TRAIN_MAX_STEPS,
        )

        yield TrainedStoreHandle(
            doc=doc,
            home=home,
            dlm_id=parsed.frontmatter.dlm_id,
            store=store,
        )
    finally:
        for key, value in saved_env.items():
            if value is not None:
                os.environ[key] = value
