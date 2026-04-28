"""CLI tests for `dlm preference` (Sprint 42)."""

from __future__ import annotations

import re
from collections import deque
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dlm.base_models import BaseModelSpec
from dlm.cli.app import app
from dlm.doc.parser import parse_file
from dlm.metrics.queries import preference_mining_for_run
from dlm.preference.judge import PairScore
from dlm.preference.pending import load_pending_plan
from dlm.store.manifest import Manifest, TrainingRunSummary, save_manifest
from dlm.store.paths import for_dlm

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_DLM_ID = "01KPQ9X1000000000000000000"
_REV = "0123456789abcdef0123456789abcdef01234567"


def _normalized_output(result: object) -> str:
    text = getattr(result, "output", "") + getattr(result, "stderr", "")
    return " ".join(_ANSI_RE.sub("", text).split())


def _write_doc(path: Path, *, body: str | None = None) -> None:
    payload = f"---\ndlm_id: {_DLM_ID}\ndlm_version: 14\nbase_model: smollm2-135m\n---\n"
    if body is None:
        body = "::instruction::\n### Q\nWhat is DGEMM?\n### A\nA matrix multiply.\n"
    path.write_text(payload + body, encoding="utf-8")


def _write_manifest(home: Path, doc: Path, *, run_id: int = 7) -> None:
    store = for_dlm(_DLM_ID, home=home)
    store.ensure_layout()
    save_manifest(
        store.manifest,
        Manifest(
            dlm_id=_DLM_ID,
            base_model="smollm2-135m",
            base_model_revision=_REV,
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
        self.loaded = False

    def load(self, spec: object, store: object, *, adapter_name: str | None = None) -> None:
        _ = spec, store, adapter_name
        self.loaded = True

    def generate(self, prompt: str, **_kwargs: object) -> str:
        return self._responses[prompt].popleft()

    def unload(self) -> None:
        self.loaded = False


class _FakeJudge:
    name = "stub:judge"
    suggested_threshold = 0.1

    def score_pair(self, prompt: str, candidate_a: str, candidate_b: str) -> PairScore:
        if prompt != "What is DGEMM?":
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return PairScore(score_a=0.1, score_b=0.9)


class _NamedFakeJudge(_FakeJudge):
    def __init__(self, name: str) -> None:
        self.name = name


def _patch_text_mining(
    monkeypatch: pytest.MonkeyPatch,
    *,
    responses: dict[str, list[str]],
    judge_names: dict[str, str] | None = None,
) -> None:
    monkeypatch.setattr("dlm.base_models.resolve", lambda *args, **kwargs: _spec())
    monkeypatch.setattr(
        "dlm.hardware.doctor",
        lambda: type("R", (), {"capabilities": object()})(),
    )
    monkeypatch.setattr(
        "dlm.inference.backends.select_backend",
        lambda *args, **kwargs: "pytorch",
    )
    monkeypatch.setattr(
        "dlm.inference.backends.build_backend",
        lambda *args, **kwargs: _FakeBackend(responses),
    )
    if judge_names is None:
        monkeypatch.setattr(
            "dlm.preference.judge.build_judge",
            lambda *args, **kwargs: _FakeJudge(),
        )
        return

    def _build_judge(ref: str, **_kwargs: object) -> _NamedFakeJudge:
        return _NamedFakeJudge(judge_names[ref])

    monkeypatch.setattr("dlm.preference.judge.build_judge", _build_judge)


class TestPreferenceCmd:
    def test_mine_stages_pending_plan_by_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_doc(doc)
        _write_manifest(home, doc)
        _patch_text_mining(
            monkeypatch,
            responses={"What is DGEMM?": ["bad answer", "good answer"]},
        )

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--home", str(home), "preference", "mine", str(doc), "--samples", "2"],
        )

        assert result.exit_code == 0, result.output
        normalized = _normalized_output(result)
        assert "preference mine plan: 1 add, 0 skip" in normalized
        assert "staged 1 mined preference section" in normalized

        pending = load_pending_plan(for_dlm(_DLM_ID, home=home))
        assert pending is not None
        assert len(pending.sections) == 1
        assert pending.sections[0].auto_mined is True

        rows = preference_mining_for_run(for_dlm(_DLM_ID, home=home).root, run_id=7)
        assert len(rows) == 1
        assert rows[0].judge_name == "stub:judge"
        assert rows[0].sample_count == 2
        assert rows[0].mined_pairs == 1
        assert rows[0].skipped_prompts == 0
        assert rows[0].write_mode == "staged"

    @pytest.mark.parametrize(
        ("judge_ref", "expected_name"),
        [
            ("sway", "sway:preference_judge"),
            ("hf:reward/model", "hf:reward/model"),
            ("cli:judge-bin --json", "cli:judge-bin --json"),
        ],
    )
    def test_mine_routes_explicit_judge_refs_through_cli_surface(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        judge_ref: str,
        expected_name: str,
    ) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_doc(doc)
        _write_manifest(home, doc)
        _patch_text_mining(
            monkeypatch,
            responses={"What is DGEMM?": ["bad answer", "good answer"]},
            judge_names={judge_ref: expected_name},
        )

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(home),
                "preference",
                "mine",
                str(doc),
                "--samples",
                "2",
                "--judge",
                judge_ref,
            ],
        )

        assert result.exit_code == 0, result.output
        rows = preference_mining_for_run(for_dlm(_DLM_ID, home=home).root, run_id=7)
        assert len(rows) == 1
        assert rows[0].judge_name == expected_name

    def test_apply_writes_staged_preferences_and_clears_pending(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_doc(doc)
        _write_manifest(home, doc)
        _patch_text_mining(
            monkeypatch,
            responses={"What is DGEMM?": ["bad answer", "good answer"]},
        )

        runner = CliRunner()
        mine_result = runner.invoke(
            app,
            ["--home", str(home), "preference", "mine", str(doc), "--samples", "2"],
        )
        assert mine_result.exit_code == 0, mine_result.output

        apply_result = runner.invoke(
            app,
            ["--home", str(home), "preference", "apply", str(doc)],
        )

        assert apply_result.exit_code == 0, apply_result.output
        normalized = _normalized_output(apply_result)
        assert "preference apply plan: 1 add, 0 skip" in normalized
        assert "wrote 1 section(s)" in normalized

        reloaded = parse_file(doc)
        assert any(section.auto_mined for section in reloaded.sections)
        assert load_pending_plan(for_dlm(_DLM_ID, home=home)) is None

    def test_mine_apply_writes_directly_without_staging(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_doc(doc)
        _write_manifest(home, doc)
        _patch_text_mining(
            monkeypatch,
            responses={"What is DGEMM?": ["bad answer", "good answer"]},
        )

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(home),
                "preference",
                "mine",
                str(doc),
                "--samples",
                "2",
                "--apply",
            ],
        )

        assert result.exit_code == 0, result.output
        reloaded = parse_file(doc)
        assert any(section.auto_mined for section in reloaded.sections)
        assert load_pending_plan(for_dlm(_DLM_ID, home=home)) is None

    def test_revert_strips_auto_mined_preferences(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_doc(doc)
        _write_manifest(home, doc)
        _patch_text_mining(
            monkeypatch,
            responses={"What is DGEMM?": ["bad answer", "good answer"]},
        )

        runner = CliRunner()
        apply_result = runner.invoke(
            app,
            [
                "--home",
                str(home),
                "preference",
                "mine",
                str(doc),
                "--samples",
                "2",
                "--apply",
            ],
        )
        assert apply_result.exit_code == 0, apply_result.output

        revert_result = runner.invoke(
            app,
            ["--home", str(home), "preference", "revert", str(doc)],
        )

        assert revert_result.exit_code == 0, revert_result.output
        assert "stripped 1 auto-mined preference section" in _normalized_output(revert_result)
        reloaded = parse_file(doc)
        assert not any(section.auto_mined for section in reloaded.sections)

    def test_list_reports_applied_and_pending_counts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_doc(doc)
        _write_manifest(home, doc)
        _patch_text_mining(
            monkeypatch,
            responses={"What is DGEMM?": ["bad answer", "good answer"]},
        )

        runner = CliRunner()
        stage_result = runner.invoke(
            app,
            ["--home", str(home), "preference", "mine", str(doc), "--samples", "2"],
        )
        assert stage_result.exit_code == 0, stage_result.output

        list_result = runner.invoke(
            app,
            ["--home", str(home), "preference", "list", str(doc)],
        )

        assert list_result.exit_code == 0, list_result.output
        normalized = _normalized_output(list_result)
        assert "applied auto-mined: 0" in normalized
        assert "staged pending: 1" in normalized
        assert "prompt=What is DGEMM?" in normalized

    def test_mine_requires_prior_training_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_doc(doc)
        _patch_text_mining(
            monkeypatch,
            responses={"What is DGEMM?": ["bad answer", "good answer"]},
        )

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--home", str(home), "preference", "mine", str(doc), "--samples", "2"],
        )

        assert result.exit_code == 1, result.output
        assert "requires a prior training run" in _normalized_output(result)
