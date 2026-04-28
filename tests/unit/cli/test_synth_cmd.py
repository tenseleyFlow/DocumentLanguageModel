"""CLI tests for `dlm synth` (Sprint 43)."""

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
from dlm.doc.sections import SectionType
from dlm.preference.judge import PairScore
from dlm.preference.pending import load_pending_plan as load_pending_preference_plan
from dlm.store.manifest import Manifest, TrainingRunSummary, save_manifest
from dlm.store.paths import for_dlm
from dlm.synth.pending import load_pending_plan

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_DLM_ID = "01KPQ9X1000000000000000000"
_REV = "0123456789abcdef0123456789abcdef01234567"


def _normalized_output(result: object) -> str:
    text = getattr(result, "output", "") + getattr(result, "stderr", "")
    return " ".join(_ANSI_RE.sub("", text).split())


def _write_synth_doc(path: Path) -> None:
    path.write_text(
        "---\n"
        f"dlm_id: {_DLM_ID}\n"
        "dlm_version: 15\n"
        "base_model: smollm2-135m\n"
        "---\n"
        "DGEMM multiplies two dense matrices and optionally accumulates the result.\n",
        encoding="utf-8",
    )


def _write_preference_doc(path: Path) -> None:
    path.write_text(
        "---\n"
        f"dlm_id: {_DLM_ID}\n"
        "dlm_version: 15\n"
        "base_model: smollm2-135m\n"
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
            source_path=doc.resolve(),
            training_runs=[
                TrainingRunSummary(
                    run_id=run_id,
                    started_at=datetime(2026, 4, 24, 12, 0, 0),
                    ended_at=datetime(2026, 4, 24, 12, 1, 0),
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


class _FakeTeacher:
    def __init__(self, name: str, payload: str) -> None:
        self.name = name
        self._payload = payload

    def generate(self, *_args: object, **_kwargs: object) -> str:
        return self._payload


class _FakeJudge:
    name = "sway:preference_judge"
    suggested_threshold = 0.1

    def score_pair(self, prompt: str, candidate_a: str, candidate_b: str) -> PairScore:
        _ = prompt, candidate_a, candidate_b
        return PairScore(score_a=0.8, score_b=0.2)


class _FakeBackend:
    def __init__(self, responses: dict[str, list[str]]) -> None:
        self._responses = {prompt: deque(items) for prompt, items in responses.items()}

    def load(self, spec: object, store: object, *, adapter_name: str | None = None) -> None:
        _ = spec, store, adapter_name

    def generate(self, prompt: str, **_kwargs: object) -> str:
        return self._responses[prompt].popleft()

    def unload(self) -> None:
        return None


def _patch_synth_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = {
        "self": ('[{"question":"What does DGEMM do?","answer":"It multiplies dense matrices."}]'),
        "hf:stub/model": (
            '[{"question":"When would you call DGEMM?","answer":"When you need a BLAS matrix multiplication."}]'
        ),
    }

    def _build_teacher(raw: str, **_kwargs: object) -> _FakeTeacher:
        payload = payloads.get(raw, payloads["self"])
        return _FakeTeacher(raw, payload)

    monkeypatch.setattr("dlm.synth.teachers.build_teacher", _build_teacher)
    monkeypatch.setattr("dlm.preference.judge.build_judge", lambda *args, **kwargs: _FakeJudge())


def _patch_preference_alias_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
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
        lambda *args, **kwargs: _FakeBackend({"What is DGEMM?": ["bad answer", "good answer"]}),
    )
    monkeypatch.setattr("dlm.preference.judge.build_judge", lambda *args, **kwargs: _FakeJudge())


class TestSynthCmd:
    def test_instructions_stage_pending_plan_by_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_synth_doc(doc)
        _patch_synth_runtime(monkeypatch)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--home", str(home), "synth", "instructions", str(doc), "--per-section", "1"],
        )

        assert result.exit_code == 0, result.output
        normalized = _normalized_output(result)
        assert "synth plan: 1 add, 0 skip" in normalized
        assert "synth filter: generated 1, dedup 1, judge passed 1, threshold 1" in normalized
        assert "staged 1 auto-synth instruction section" in normalized

        pending = load_pending_plan(for_dlm(_DLM_ID, home=home))
        assert pending is not None
        assert len(pending.sections) == 1
        assert pending.sections[0].auto_synth is True
        assert pending.sections[0].synth_teacher == "self"
        assert pending.sections[0].synth_strategy == "extraction"

    def test_apply_writes_auto_synth_sections_and_clears_pending(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_synth_doc(doc)
        _patch_synth_runtime(monkeypatch)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(home),
                "synth",
                "instructions",
                str(doc),
                "--per-section",
                "1",
                "--apply",
            ],
        )

        assert result.exit_code == 0, result.output
        normalized = _normalized_output(result)
        assert "synth apply plan: 1 add, 0 skip" in normalized
        assert "wrote 1 section(s)" in normalized
        assert load_pending_plan(for_dlm(_DLM_ID, home=home)) is None

        parsed = parse_file(doc)
        assert any(
            section.type is SectionType.INSTRUCTION and section.auto_synth
            for section in parsed.sections
        )

    def test_revert_strips_auto_synth_sections(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_synth_doc(doc)
        _patch_synth_runtime(monkeypatch)

        runner = CliRunner()
        apply_result = runner.invoke(
            app,
            [
                "--home",
                str(home),
                "synth",
                "instructions",
                str(doc),
                "--per-section",
                "1",
                "--apply",
            ],
        )
        assert apply_result.exit_code == 0, apply_result.output

        revert_result = runner.invoke(
            app,
            ["--home", str(home), "synth", "revert", str(doc)],
        )

        assert revert_result.exit_code == 0, revert_result.output
        assert "stripped 1 auto-synth instruction section" in _normalized_output(revert_result)

        parsed = parse_file(doc)
        assert not any(
            section.type is SectionType.INSTRUCTION and section.auto_synth
            for section in parsed.sections
        )

    def test_list_shows_counts_for_applied_and_pending_sections(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_synth_doc(doc)
        _patch_synth_runtime(monkeypatch)

        runner = CliRunner()
        apply_result = runner.invoke(
            app,
            [
                "--home",
                str(home),
                "synth",
                "instructions",
                str(doc),
                "--per-section",
                "1",
                "--strategy",
                "extraction",
                "--apply",
            ],
        )
        assert apply_result.exit_code == 0, apply_result.output

        stage_result = runner.invoke(
            app,
            [
                "--home",
                str(home),
                "synth",
                "instructions",
                str(doc),
                "--teacher",
                "hf:stub/model",
                "--per-section",
                "1",
                "--strategy",
                "expansion",
            ],
        )
        assert stage_result.exit_code == 0, stage_result.output

        list_result = runner.invoke(
            app,
            ["--home", str(home), "synth", "list", str(doc)],
        )

        assert list_result.exit_code == 0, list_result.output
        normalized = _normalized_output(list_result)
        source_id = parse_file(doc).sections[0].section_id
        assert "applied auto-synth: 1" in normalized
        assert "staged pending: 1" in normalized
        assert "self: 1" in normalized
        assert "hf:stub/model: 1" in normalized
        assert "extraction: 1" in normalized
        assert "expansion: 1" in normalized
        assert f"{source_id}: 1" in normalized

    def test_preferences_alias_routes_through_preference_mine(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_preference_doc(doc)
        _write_manifest(home, doc)
        _patch_preference_alias_runtime(monkeypatch)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(home),
                "synth",
                "preferences",
                str(doc),
                "--samples",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "preference mine plan: 1 add, 0 skip" in _normalized_output(result)
        pending = load_pending_preference_plan(for_dlm(_DLM_ID, home=home))
        assert pending is not None
        assert len(pending.sections) == 1
        assert pending.sections[0].auto_mined is True

    def test_openai_teacher_without_api_key_fails_cleanly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_synth_doc(doc)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(home),
                "synth",
                "instructions",
                str(doc),
                "--teacher",
                "openai:gpt-4o-mini",
                "--filter",
                "none",
            ],
        )

        assert result.exit_code == 1, result.output
        assert "requires $OPENAI_API_KEY to be set" in _normalized_output(result)

    def test_anthropic_teacher_without_api_key_fails_cleanly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        home = tmp_path / "home"
        doc = tmp_path / "doc.dlm"
        _write_synth_doc(doc)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(home),
                "synth",
                "instructions",
                str(doc),
                "--teacher",
                "anthropic:claude-3-5-haiku-latest",
                "--filter",
                "none",
            ],
        )

        assert result.exit_code == 1, result.output
        assert "requires $ANTHROPIC_API_KEY to be set" in _normalized_output(result)
