"""End-to-end: DAPT schedule on a prose-heavy doc trains cleanly.

Verifies three contracts rather than a rigorous "DAPT beats SFT on
loss" numerical claim — a 20-step run doesn't carry enough statistical
signal for that. Instead we check:

1. The CPT row fraction crosses the DAPT threshold on a prose-only doc.
2. `trainer.run()` completes end-to-end with `training.cpt.schedule=dapt`
   and commits a v0001 adapter under a fresh store.
3. The selected SFTConfig actually carries the DAPT overrides
   (`lr_scheduler_type=cosine_with_min_lr`, warmup_ratio=0.2,
   min_lr_rate=0.1) — proven via a lightweight spy on the
   `_build_real_trainer` seam.

Slow — driven by the weekly integration-slow CI job.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.slow


_PROSE_CORPUS = """
# Domain prose

This is a paragraph of domain-specific prose. The training corpus is
dominated by continuous text rather than question-answer pairs, which
is the regime where the DAPT schedule's longer warmup and non-zero
floor are designed to help.

## Another section

Additional prose content follows. Tokenizers often split uncommon
domain vocabulary into many subword pieces; the vocab gap report
surfaces this at the start of training so the user can decide whether
a different base model's tokenizer would fit the corpus better.

## Yet more prose

We add enough paragraphs here to push the row count above the 70%
threshold the CPT runtime uses to auto-select DAPT. Each paragraph
becomes one training row; a handful of them crosses that bar easily
on the tiny-model fixture.

## Continued text

The paragraphs carry meaningful content rather than lorem ipsum so the
resulting checkpoint is at least plausibly useful as a smoke test of
the DAPT training path.
""".strip()


@pytest.mark.slow
def test_dapt_schedule_fires_on_prose_doc(tmp_path_factory: pytest.TempPathFactory) -> None:  # type: ignore[no-untyped-def]
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"torch/transformers unavailable: {exc}")

    try:
        from tests.fixtures.tiny_model import tiny_model_path

        tiny_model_path()
    except Exception as exc:
        pytest.skip(f"tiny-model fixture unavailable: {exc}")

    from dlm.doc.parser import parse_file
    from dlm.store.manifest import Manifest, load_manifest, save_manifest
    from dlm.store.paths import for_dlm
    from dlm.train import run as run_training
    from dlm.train.cpt.runtime import cpt_row_fraction
    from tests.fixtures.dlm_factory import make_dlm, prose
    from tests.fixtures.planning import resolve_spec_and_plan

    # Offline env would block model weight downloads.
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(key, None)

    home = tmp_path_factory.mktemp("dlm-dapt-home")
    os.environ["DLM_HOME"] = str(home)
    doc_path: Path = home / "prose.dlm"
    doc_path.write_text(
        make_dlm(
            sections=[prose(_PROSE_CORPUS)],
            base_model="smollm2-135m",
            training_overrides={"cpt": {"schedule": "dapt"}},
        ),
        encoding="utf-8",
    )

    parsed = parse_file(doc_path)
    assert parsed.frontmatter.training.cpt.schedule == "dapt"

    # Sanity: prose-only doc must cross the auto threshold too.
    from dlm.data.sections_to_rows import sections_to_rows

    rows = sections_to_rows(list(parsed.sections))
    assert cpt_row_fraction(rows) == pytest.approx(1.0)

    spec, plan, _caps = resolve_spec_and_plan(parsed, accept_license=True)
    store = for_dlm(parsed.frontmatter.dlm_id)
    store.ensure_layout()
    save_manifest(
        store.manifest,
        Manifest(
            dlm_id=parsed.frontmatter.dlm_id,
            base_model=parsed.frontmatter.base_model,
        ),
    )

    # Spy on the SFTTrainer factory to capture the realized SFTConfig.
    import dlm.train.trainer as trainer_mod

    captured: dict[str, Any] = {}
    real_build = trainer_mod._build_real_trainer  # type: ignore[attr-defined]

    def _spy(*args: Any, **kwargs: Any) -> Any:
        trainer = real_build(*args, **kwargs)
        captured["args"] = trainer.args
        return trainer

    trainer_mod._build_real_trainer = _spy  # type: ignore[attr-defined]
    try:
        result = run_training(
            store,
            parsed,
            spec,
            plan,
            mode="fresh",
            seed=42,
            max_steps=10,
        )
    finally:
        trainer_mod._build_real_trainer = real_build  # type: ignore[attr-defined]

    assert result.adapter_version == 1
    assert store.adapter_version(1).is_dir()

    manifest = load_manifest(store.manifest)
    assert len(manifest.training_runs) == 1
    assert manifest.adapter_version == 1

    # DAPT overrides landed on the realized SFTConfig.
    args = captured["args"]
    assert args.lr_scheduler_type == "cosine_with_min_lr"
    assert args.warmup_ratio == pytest.approx(0.2)
    assert args.lr_scheduler_kwargs == {"min_lr_rate": pytest.approx(0.1)}
