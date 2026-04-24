"""Slow integration: self-teacher synth pass compounds after retraining.

This is Sprint 43's bootstrap proof. We reuse the shared tiny-model
fixture for the seed adapter, rewrite the copied document to prose-only,
run a real `dlm synth instructions --teacher self --apply` pass, retrain
on those generated sections, and then compare a second synth pass from
adapter v0001 vs v0002 with the real SwayJudge.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from dlm.cli.app import app
from dlm.doc.parser import ParsedDlm, parse_file
from dlm.doc.sections import Section, SectionType
from dlm.doc.serializer import serialize
from dlm.preference.judge import JudgeInvocationError, JudgeUnavailableError
from dlm.synth import SelfTeacher, build_synth_plan, filter_synth_plan

if TYPE_CHECKING:
    from tests.fixtures.trained_store import TrainedStoreHandle

    from dlm.store.paths import StorePath

pytestmark = pytest.mark.slow

_PROSE_BODY = (
    "DGEMM multiplies dense matrices. "
    "It computes C = alpha*A*B + beta*C. "
    "BLAS libraries use it for high-performance linear algebra.\n"
)


def _copy_fixture_store(
    trained_store: TrainedStoreHandle,
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, StorePath]:
    from dlm.store.manifest import load_manifest, save_manifest
    from dlm.store.paths import for_dlm

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("DLM_HOME", str(home))

    source_doc = trained_store.doc
    doc = home / source_doc.name
    shutil.copy2(source_doc, doc)

    parsed = parse_file(doc)
    store = for_dlm(parsed.frontmatter.dlm_id)
    shutil.copytree(trained_store.store.root, store.root, dirs_exist_ok=True)

    manifest = load_manifest(store.manifest)
    save_manifest(
        store.manifest,
        manifest.model_copy(update={"source_path": doc.resolve()}),
    )
    return doc, store


def _rewrite_doc_to_prose_only(doc: Path) -> None:
    parsed = parse_file(doc)
    rewritten = ParsedDlm(
        frontmatter=parsed.frontmatter.model_copy(update={"dlm_version": 15}),
        sections=(Section(type=SectionType.PROSE, content=_PROSE_BODY),),
    )
    doc.write_text(serialize(rewritten), encoding="utf-8")


def _synth_metrics(doc: Path, store: StorePath, *, adapter_version: int) -> tuple[int, float]:
    from dlm.preference import build_judge

    original = store.resolve_current_adapter()
    store.set_current_adapter(store.adapter_version(adapter_version))
    try:
        parsed = parse_file(doc)
        teacher = SelfTeacher(doc, backend="pytorch")
        plan = build_synth_plan(
            parsed,
            teacher,
            per_section=2,
            strategy="extraction",
            max_pairs=2,
            temperature=0.0,
        )
        try:
            judge = build_judge("sway", dlm_path=doc)
        except JudgeUnavailableError as exc:
            pytest.skip(f"sway judge unavailable for synth-cycle proof: {exc}")
        filtered = filter_synth_plan(plan, filter_kind="sway", judge=judge)
    except JudgeInvocationError as exc:
        pytest.skip(f"sway judge unavailable for synth-cycle proof: {exc}")
    finally:
        if original is not None:
            store.set_current_adapter(original)

    total_margin = sum(
        addition.judge_score.margin
        for addition in filtered.additions
        if addition.judge_score is not None
    )
    return len(filtered.additions), total_margin


@pytest.mark.slow
def test_self_teacher_cycle_improves_second_pass_sway_yield(
    trained_store: TrainedStoreHandle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlm.base_models import resolve as resolve_base_model
    from dlm.store.manifest import load_manifest
    from dlm.train import run as run_training

    doc, store = _copy_fixture_store(trained_store, tmp_path=tmp_path, monkeypatch=monkeypatch)
    _rewrite_doc_to_prose_only(doc)

    home = tmp_path / "home"
    runner = CliRunner()
    synth_result = runner.invoke(
        app,
        [
            "--home",
            str(home),
            "synth",
            "instructions",
            str(doc),
            "--teacher",
            "self",
            "--per-section",
            "2",
            "--max-pairs",
            "2",
            "--apply",
        ],
    )
    assert synth_result.exit_code == 0, synth_result.output

    mined_doc = parse_file(doc)
    auto_synth_sections = [
        section
        for section in mined_doc.sections
        if section.type is SectionType.INSTRUCTION and section.auto_synth
    ]
    assert auto_synth_sections, "expected the first self-teacher pass to write synth sections"

    spec = resolve_base_model(mined_doc.frontmatter.base_model, accept_license=True)
    result = run_training(
        store,
        mined_doc,
        spec,
        trained_store.plan,
        mode="fresh",
        lock_mode="ignore",
        capabilities=trained_store.capabilities,
        seed=42,
        max_steps=10,
    )
    assert result.adapter_version == 2

    manifest = load_manifest(store.manifest)
    assert manifest.adapter_version == 2
    assert len(manifest.training_runs) >= 2

    baseline_count, baseline_margin = _synth_metrics(doc, store, adapter_version=1)
    final_count, final_margin = _synth_metrics(doc, store, adapter_version=2)

    assert final_count > baseline_count, (
        "expected the second synth pass to yield more accepted additions "
        f"(baseline={baseline_count}, final={final_count})"
    )
    assert final_margin > baseline_margin, (
        "expected the second synth pass to improve total sway margin "
        f"(baseline={baseline_margin:.4f}, final={final_margin:.4f})"
    )
