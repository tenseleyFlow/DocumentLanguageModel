"""Slow integration: train → mine → train again improves held-out preference score.

This is Sprint 42's bootstrap-loop proof. We keep the mined candidates
deterministic (scripted backend + judge) so the test is stable, but the
two preference-training passes and the final held-out SwayJudge check are
real.
"""

from __future__ import annotations

import shutil
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from dlm.cli.app import app
from dlm.doc.parser import ParsedDlm, parse_file
from dlm.doc.serializer import serialize
from dlm.preference.judge import JudgeInvocationError, JudgeUnavailableError, PairScore, SwayJudge

if TYPE_CHECKING:
    from tests.fixtures.trained_store import TrainedStoreHandle

pytestmark = pytest.mark.slow

_EXTRA_BODY = """
::instruction::
### Q
What color is grass?
### A
Green.

::instruction::
### Q
What is 10 - 3?
### A
7.

::preference::
### Prompt
Is water wet?
### Chosen
Yes.
### Rejected
Water is generally considered wet in everyday language.
"""

_MINE_RESPONSES = {
    "What is 2 + 2?": ["4.", "The sum of two and two is four."],
    "What is the capital of France?": [
        "Paris.",
        "The capital of France is Paris.",
    ],
    "What color is grass?": ["Green.", "Grass is usually green."],
    "What is 10 - 3?": ["7.", "Ten minus three equals seven."],
}

_HELD_OUT_PAIRS = (
    ("What is 8 + 1?", "9.", "The result of adding eight and one is nine."),
    ("What color is snow?", "White.", "Snow is typically white in daylight."),
    ("What is the capital of Italy?", "Rome.", "The capital city of Italy is Rome."),
)


class _FakeMiningBackend:
    def __init__(self, responses: dict[str, list[str]]) -> None:
        self._responses = {prompt: deque(items) for prompt, items in responses.items()}

    def load(self, spec: object, store: object, *, adapter_name: str | None = None) -> None:
        _ = spec, store, adapter_name

    def generate(self, prompt: str, **_kwargs: object) -> str:
        return self._responses[prompt].popleft()

    def unload(self) -> None:
        return None


class _TerseJudge:
    name = "cli:terse-judge"
    suggested_threshold = 0.1

    def score_pair(self, prompt: str, candidate_a: str, candidate_b: str) -> PairScore:
        _ = prompt
        return PairScore(score_a=-float(len(candidate_a)), score_b=-float(len(candidate_b)))


def _copy_fixture_store(
    trained_store: TrainedStoreHandle,
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, object]:
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


def _prepare_doc_for_cycle(doc: Path) -> None:
    current = doc.read_text(encoding="utf-8")
    doc.write_text(current.rstrip() + "\n\n" + _EXTRA_BODY.lstrip(), encoding="utf-8")

    parsed = parse_file(doc)
    new_pref = parsed.frontmatter.training.preference.model_copy(
        update={"method": "orpo", "enabled": True}
    )
    new_training = parsed.frontmatter.training.model_copy(update={"preference": new_pref})
    rewritten = ParsedDlm(
        frontmatter=parsed.frontmatter.model_copy(update={"training": new_training}),
        sections=parsed.sections,
    )
    doc.write_text(serialize(rewritten), encoding="utf-8")


def _patch_mining(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dlm.inference.backends.select_backend",
        lambda *args, **kwargs: "pytorch",
    )
    monkeypatch.setattr(
        "dlm.inference.backends.build_backend",
        lambda *args, **kwargs: _FakeMiningBackend(_MINE_RESPONSES),
    )
    monkeypatch.setattr(
        "dlm.preference.build_judge",
        lambda *args, **kwargs: _TerseJudge(),
    )


def _mean_margin_for_version(doc: Path, store: object, version: int) -> float:
    target = store.adapter_version(version)
    original = store.resolve_current_adapter()
    assert original is not None
    store.set_current_adapter(target)
    try:
        judge = SwayJudge(doc)
        margins = [
            judge.score_pair(prompt, chosen, rejected).margin
            for prompt, chosen, rejected in _HELD_OUT_PAIRS
        ]
    except (JudgeUnavailableError, JudgeInvocationError) as exc:
        pytest.skip(f"sway judge unavailable for mine-cycle proof: {exc}")
    finally:
        store.set_current_adapter(original)
    return sum(margins) / len(margins)


@pytest.mark.slow
def test_preference_mine_cycle_improves_held_out_sway_margin(
    trained_store: TrainedStoreHandle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlm.base_models import resolve as resolve_base_model
    from dlm.doc.sections import SectionType
    from dlm.store.manifest import load_manifest
    from dlm.train.preference.phase_orchestrator import run_phases

    doc, store = _copy_fixture_store(trained_store, tmp_path=tmp_path, monkeypatch=monkeypatch)
    _prepare_doc_for_cycle(doc)

    parsed = parse_file(doc)
    spec = resolve_base_model(parsed.frontmatter.base_model, accept_license=True)
    plan = trained_store.plan
    capabilities = trained_store.capabilities

    baseline = run_phases(
        store,
        parsed,
        spec,
        plan,
        phase="preference",
        capabilities=capabilities,
        lock_mode="ignore",
        seed=42,
        max_steps=20,
    )
    assert [result.phase for result in baseline] == ["preference"]
    assert baseline[0].result.adapter_version == 2

    _patch_mining(monkeypatch)
    runner = CliRunner()
    mine_result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_path / "home"),
            "preference",
            "mine",
            str(doc),
            "--samples",
            "2",
            "--max-pairs",
            "4",
            "--apply",
        ],
    )
    assert mine_result.exit_code == 0, mine_result.output

    mined_doc = parse_file(doc)
    auto_mined_sections = [
        section
        for section in mined_doc.sections
        if section.type is SectionType.PREFERENCE and section.auto_mined
    ]
    assert len(auto_mined_sections) == 4

    final = run_phases(
        store,
        mined_doc,
        spec,
        plan,
        phase="preference",
        capabilities=capabilities,
        lock_mode="ignore",
        seed=42,
        max_steps=20,
    )
    assert [result.phase for result in final] == ["preference"]
    assert final[0].result.adapter_version == 3

    manifest = load_manifest(store.manifest)
    assert manifest.adapter_version == 3
    assert len(manifest.training_runs) >= 3

    baseline_margin = _mean_margin_for_version(doc, store, 2)
    final_margin = _mean_margin_for_version(doc, store, 3)
    assert final_margin > baseline_margin, (
        "expected final preference-tuned adapter to improve held-out sway margin "
        f"(baseline={baseline_margin:.4f}, final={final_margin:.4f})"
    )
