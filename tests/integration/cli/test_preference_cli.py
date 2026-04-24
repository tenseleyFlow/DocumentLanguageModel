"""CLI-level preference loop proof: mine/apply feeds the train path."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from dlm.base_models import BaseModelSpec
from dlm.cli.app import app
from dlm.doc.parser import parse_file
from dlm.doc.sections import SectionType
from dlm.preference.judge import PairScore
from dlm.store.manifest import Manifest, TrainingRunSummary, save_manifest
from dlm.store.paths import for_dlm

_DLM_ID = "01KPQ9X1000000000000000000"
_REV = "0123456789abcdef0123456789abcdef01234567"


def _write_doc(path: Path) -> None:
    path.write_text(
        "---\n"
        f"dlm_id: {_DLM_ID}\n"
        "dlm_version: 14\n"
        "base_model: smollm2-135m\n"
        "training:\n"
        "  seed: 42\n"
        "---\n"
        "::instruction::\n"
        "### Q\n"
        "What is DGEMM?\n"
        "### A\n"
        "A matrix multiply.\n",
        encoding="utf-8",
    )


def _write_manifest(home: Path, doc: Path, *, run_id: int = 7) -> None:
    store = for_dlm(_DLM_ID, home=home)
    store.ensure_layout()
    save_manifest(
        store.manifest,
        Manifest(
            dlm_id=_DLM_ID,
            base_model="smollm2-135m",
            base_model_revision=_REV,
            adapter_version=1,
            source_path=doc.resolve(),
            training_runs=[
                TrainingRunSummary(
                    run_id=run_id,
                    started_at=datetime(2026, 4, 23, 20, 0, 0),
                    ended_at=datetime(2026, 4, 23, 20, 1, 0),
                    adapter_version=1,
                    seed=123,
                    steps=12,
                )
            ],
        ),
    )


def _spec() -> BaseModelSpec:
    return BaseModelSpec.model_validate(
        {
            "key": "smollm2-135m",
            "hf_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "revision": _REV,
            "architecture": "LlamaForCausalLM",
            "params": 135_000_000,
            "target_modules": ["q_proj", "v_proj"],
            "template": "chatml",
            "gguf_arch": "llama",
            "tokenizer_pre": "default",
            "license_spdx": "Apache-2.0",
            "license_url": None,
            "requires_acceptance": False,
            "redistributable": True,
            "size_gb_fp16": 0.3,
            "context_length": 4096,
            "recommended_seq_len": 2048,
        }
    )


class _FakeBackend:
    def __init__(self, responses: dict[str, list[str]]) -> None:
        self._responses = {prompt: deque(items) for prompt, items in responses.items()}

    def load(self, spec: object, store: object, *, adapter_name: str | None = None) -> None:
        _ = spec, store, adapter_name

    def generate(self, prompt: str, **_kwargs: object) -> str:
        return self._responses[prompt].popleft()

    def unload(self) -> None:
        return None


class _FakeJudge:
    name = "stub:judge"
    suggested_threshold = 0.1

    def score_pair(self, prompt: str, candidate_a: str, candidate_b: str) -> PairScore:
        if prompt != "What is DGEMM?":
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return PairScore(score_a=0.1, score_b=0.9)


def _patch_mining_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dlm.base_models.resolve", lambda *args, **kwargs: _spec())
    monkeypatch.setattr(
        "dlm.hardware.doctor",
        lambda *args, **kwargs: SimpleNamespace(plan=object(), capabilities=object()),
    )
    monkeypatch.setattr(
        "dlm.inference.backends.select_backend",
        lambda *args, **kwargs: "pytorch",
    )
    monkeypatch.setattr(
        "dlm.inference.backends.build_backend",
        lambda *args, **kwargs: _FakeBackend({"What is DGEMM?": ["bad answer", "good answer"]}),
    )
    monkeypatch.setattr(
        "dlm.preference.build_judge",
        lambda *args, **kwargs: _FakeJudge(),
    )


def _capture_run_phases(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake(
        store: object, parsed: object, spec: object, plan: object, **kwargs: object
    ) -> list[object]:
        captured["store"] = store
        captured["parsed"] = parsed
        captured["spec"] = spec
        captured["plan"] = plan
        captured["kwargs"] = kwargs
        return [
            SimpleNamespace(
                phase="preference",
                result=SimpleNamespace(
                    adapter_version=2,
                    steps=12,
                    seed=42,
                    determinism=SimpleNamespace(class_="full"),
                    adapter_path=Path("/tmp/adapter-v0002"),
                    log_path=Path("/tmp/train.log"),
                    final_train_loss=None,
                ),
            )
        ]

    monkeypatch.setattr("dlm.train.preference.phase_orchestrator.run_phases", fake)
    return captured


@pytest.mark.parametrize(
    ("extra_train_args", "expected_include_auto_mined"),
    [
        ([], True),
        (["--no-mined"], False),
    ],
)
def test_mine_apply_then_train_surfaces_auto_mined_preferences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_train_args: list[str],
    expected_include_auto_mined: bool,
) -> None:
    home = tmp_path / "home"
    doc = tmp_path / "doc.dlm"
    _write_doc(doc)
    _write_manifest(home, doc)
    _patch_mining_runtime(monkeypatch)
    captured = _capture_run_phases(monkeypatch)

    runner = CliRunner()

    mine_result = runner.invoke(
        app,
        ["--home", str(home), "preference", "mine", str(doc), "--samples", "2", "--apply"],
    )
    assert mine_result.exit_code == 0, mine_result.output

    reloaded = parse_file(doc)
    assert any(
        section.type is SectionType.PREFERENCE and section.auto_mined
        for section in reloaded.sections
    )

    train_result = runner.invoke(
        app,
        ["--home", str(home), "train", str(doc), "--phase", "preference", *extra_train_args],
    )
    assert train_result.exit_code == 0, train_result.output

    parsed = captured["parsed"]
    sections = parsed.sections
    assert any(
        section.type is SectionType.PREFERENCE and section.auto_mined for section in sections
    )
    kwargs = captured["kwargs"]
    assert kwargs["include_auto_mined"] is expected_include_auto_mined
