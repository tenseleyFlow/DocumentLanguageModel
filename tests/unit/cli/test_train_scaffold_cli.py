"""Audit-09 B1+B2+B3: CLI-level regression for `dlm train <dir>`.

These tests invoke `train_cmd` via `typer.testing.CliRunner` and monkey-
patch `run_phases` so the pre-train wiring gets exercised without the
cost of actually training a model. The previous slow integration test
(`tests/integration/directives/test_auto_scaffold_cycle.py`) hand-assembled
`store.ensure_layout() + save_manifest()` around the trainer call, which
hid the two blockers the auditor found:

- B1: `train_cmd`'s directory branch never called `save_manifest()`, so
  `trainer.run -> load_manifest` crashed with `ManifestCorruptError`.
- B2: the scaffolded `.dlm` carried `path: "."`, which anchored on
  `<target>/.dlm/` and ingested zero files from the user's tree.

The tests below would have failed before the fixes landed.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dlm.cli.app import app
from tests.fixtures.hardware_mocks import force_mps


def _captured() -> dict[str, Any]:
    return {}


def _install_capturing_fake(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    """Replace `run_phases` with a stub that records call args and
    returns `[]` (triggering the CLI's "no-op: nothing to train" path
    with exit code 0). The scaffold + manifest + expand_sources pipeline
    all run before this stub is reached.
    """
    from dlm.train.preference import phase_orchestrator as po

    def fake(
        store: Any,
        parsed: Any,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        captured["store"] = store
        captured["parsed"] = parsed
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(po, "run_phases", fake)


def _section_texts(sections: Iterable[Any]) -> str:
    return " ".join(s.content for s in sections)


class TestDlmTrainDirScaffold:
    def test_scaffold_provisions_store_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B1 regression: after `dlm train <dir>`, the store manifest
        exists. Without the fix, `ensure_layout()` ran but the manifest
        file was never written, so the next `load_manifest` crashed."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.md").write_text("# Alpha\nwidgets\n", encoding="utf-8")
        (corpus / "b.md").write_text("# Beta\ngadgets\n", encoding="utf-8")

        captured = _captured()
        _install_capturing_fake(monkeypatch, captured)

        with force_mps():
            runner = CliRunner()
            result = runner.invoke(
                app,
                [
                    "--home",
                    str(tmp_path / "home"),
                    "train",
                    str(corpus),
                    "--base",
                    "smollm2-135m",
                    "--include",
                    "**/*.md",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "store" in captured, f"run_phases never reached: {result.output}"
        manifest_path = captured["store"].manifest
        assert manifest_path.exists(), (
            f"B1: store manifest not written; next train would crash with "
            f"ManifestCorruptError. Output:\n{result.output}"
        )

    def test_scaffold_ingests_corpus_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B2 regression: the scaffolded `.dlm`, when `expand_sources`
        runs, must resolve to the user's target directory — not
        `<target>/.dlm/`. Without the fix, `path: "."` anchored on the
        `.dlm`'s parent (the `.dlm/` scaffold dir) so `file_count` was
        always 0."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "alpha.md").write_text("# Alpha\nalpha-unique-token\n", encoding="utf-8")
        (corpus / "beta.md").write_text("# Beta\nbeta-unique-token\n", encoding="utf-8")

        captured = _captured()
        _install_capturing_fake(monkeypatch, captured)

        with force_mps():
            runner = CliRunner()
            result = runner.invoke(
                app,
                [
                    "--home",
                    str(tmp_path / "home"),
                    "train",
                    str(corpus),
                    "--base",
                    "smollm2-135m",
                    "--include",
                    "**/*.md",
                ],
            )

        assert result.exit_code == 0, result.output

        # The trainer's `_materialize_directive_sources` calls
        # `expand_sources` with the .dlm's parent as base_path. Replay
        # that here against the scaffolded .dlm to verify the corpus
        # resolution ends at the user's tree.
        from dlm.directives import expand_sources
        from dlm.doc.parser import parse_file

        scaffolded = corpus / ".dlm" / "corpus.dlm"
        parsed_scaffold = parse_file(scaffolded)
        expanded = expand_sources(parsed_scaffold, base_path=scaffolded.parent)

        assert len(expanded.sections) == 2, (
            f"B2: expected 2 sections (alpha.md + beta.md), got "
            f"{len(expanded.sections)}. Provenance: {expanded.provenance}"
        )
        combined = _section_texts(expanded.sections)
        rendered = "\n".join(f"  {s.content[:80]!r}" for s in expanded.sections)
        assert "alpha-unique-token" in combined, "B2: alpha.md not ingested. got:\n" + rendered
        assert "beta-unique-token" in combined, "B2: beta.md not ingested. got:\n" + rendered
        assert expanded.provenance[0].file_count == 2
        assert expanded.provenance[0].total_bytes > 0

    def test_scaffold_frontmatter_carries_target_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B2 shape regression: the written `.dlm`'s `training.sources[0].path`
        must be the user's target directory (absolute, resolved), not
        the literal `.` that the old scaffold emitted."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "note.md").write_text("content\n", encoding="utf-8")

        captured = _captured()
        _install_capturing_fake(monkeypatch, captured)

        with force_mps():
            runner = CliRunner()
            result = runner.invoke(
                app,
                [
                    "--home",
                    str(tmp_path / "home"),
                    "train",
                    str(corpus),
                    "--base",
                    "smollm2-135m",
                    "--include",
                    "**/*.md",
                ],
            )

        assert result.exit_code == 0, result.output
        scaffolded = corpus / ".dlm" / "corpus.dlm"
        assert scaffolded.is_file(), result.output

        from dlm.doc.parser import parse_file

        parsed = parse_file(scaffolded)
        sources = parsed.frontmatter.training.sources
        assert sources is not None
        assert len(sources) == 1
        written_path = Path(sources[0].path)
        assert written_path.is_absolute(), (
            f"B2: scaffold wrote non-absolute path {sources[0].path!r}"
        )
        assert written_path.resolve() == corpus.resolve(), (
            f"B2: scaffold path {written_path} != corpus {corpus.resolve()}"
        )

    def test_resume_scaffold_does_not_overwrite_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Re-running `dlm train <dir>` after a first scaffold reuses
        the `.dlm` (scaffolded=False) and must not touch the manifest.
        Guards the `if not store.manifest.exists()` condition that
        protects existing training history."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "note.md").write_text("content\n", encoding="utf-8")

        captured_first = _captured()
        _install_capturing_fake(monkeypatch, captured_first)

        with force_mps():
            runner = CliRunner()
            r1 = runner.invoke(
                app,
                [
                    "--home",
                    str(tmp_path / "home"),
                    "train",
                    str(corpus),
                    "--base",
                    "smollm2-135m",
                    "--include",
                    "**/*.md",
                ],
            )
            assert r1.exit_code == 0, r1.output
            manifest_path = captured_first["store"].manifest
            assert manifest_path.exists()
            first_mtime = manifest_path.stat().st_mtime_ns

            # Second invocation — same corpus, no flags beyond what's
            # required to find the scaffolded .dlm.
            captured_second = _captured()
            _install_capturing_fake(monkeypatch, captured_second)
            r2 = runner.invoke(
                app,
                [
                    "--home",
                    str(tmp_path / "home"),
                    "train",
                    str(corpus),
                ],
            )

        assert r2.exit_code == 0, r2.output
        assert manifest_path.stat().st_mtime_ns == first_mtime, (
            "manifest was rewritten on the resume path; training history could be lost"
        )
