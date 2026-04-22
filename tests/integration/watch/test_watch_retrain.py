"""Watch-mode retrain smoke (Sprint 25).

Edits an on-disk `.dlm`, drives one `do_one_cycle` directly, and
asserts the adapter version bumped. This skips the full `watchfiles`
event loop (that's driven by real FS notifications which are flaky
in CI) and instead exercises the cycle driver against a real
trained store.

Slow-marked; depends on the `trained_store` session fixture from
Sprint 14.5 which trains a tiny-model adapter once.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.fixtures.trained_store import TrainedStoreHandle


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("DLM_ENABLE_SLOW_INTEGRATION") != "1",
        reason="set DLM_ENABLE_SLOW_INTEGRATION=1 to opt in",
    ),
]


def test_watch_cycle_detects_new_content_and_retrains(  # pragma: no cover - slow path
    trained_store: TrainedStoreHandle,
    tmp_path: Path,
) -> None:
    """`do_one_cycle` on a doc with new content runs the trainer and bumps version."""
    from dlm.base_models import resolve as resolve_base_model
    from dlm.doc.parser import parse_file
    from dlm.hardware import doctor
    from dlm.train.trainer import run as trainer_run
    from dlm.watch.loop import do_one_cycle

    doc_path = trained_store.doc_path
    store = trained_store.store
    initial_adapter = store.resolve_current_adapter()
    assert initial_adapter is not None

    # Append a new section to the doc so the ChangeSet sees `new`.
    original = doc_path.read_text(encoding="utf-8")
    doc_path.write_text(
        original + "\n\n## Added by watch test\n\nThis is new content.\n",
        encoding="utf-8",
    )

    parsed = parse_file(doc_path)
    spec = resolve_base_model(parsed.frontmatter.base_model, accept_license=True)
    plan = doctor(training_config=parsed.frontmatter.training).plan
    if plan is None:
        pytest.skip("no viable plan on this host — watch retrain needs a real trainer")

    result = do_one_cycle(
        doc_path=doc_path,
        store=store,
        spec=spec,
        plan=plan,
        max_steps=1,
        reload_doc=parse_file,
        retrain=lambda store, parsed, spec, plan, *, max_steps: trainer_run(
            store, parsed, spec, plan, mode="resume", max_steps=max_steps
        ),
    )

    assert result.ran is True, "cycle did not run despite new content"
    assert result.new_sections >= 1

    # Adapter version should have bumped.
    post_adapter = store.resolve_current_adapter()
    assert post_adapter is not None
    assert post_adapter != initial_adapter, (
        f"adapter pointer did not advance: {initial_adapter} → {post_adapter}"
    )
