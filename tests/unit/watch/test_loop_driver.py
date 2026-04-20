"""Watch-loop cycle driver — dependency-injected with stub reloader + retrainer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from dlm.doc.sections import Section, SectionType
from dlm.store.manifest import Manifest
from dlm.watch.loop import do_one_cycle


def _section(content: str) -> Section:
    """`section_id` is derived from content + type; callers pick distinct content."""
    return Section(type=SectionType.PROSE, content=content)


def _store_stub(manifest: Manifest, tmp_path: Path) -> SimpleNamespace:
    mpath = tmp_path / "manifest.json"
    from dlm.store.manifest import save_manifest

    save_manifest(mpath, manifest)
    return SimpleNamespace(manifest=mpath)


class TestCycleSkipsWhenNoNewSections:
    def test_skips_when_all_sections_present(self, tmp_path: Path) -> None:
        """ChangeSet with empty `new` means we don't retrain."""
        sec = _section("alpha content")
        parsed = SimpleNamespace(sections=[sec])
        manifest = Manifest(
            dlm_id="01HZXY",
            base_model="smollm2-135m",
            content_hashes={sec.section_id: sec.section_id},
        )
        store = _store_stub(manifest, tmp_path)

        retrain = MagicMock()
        result = do_one_cycle(
            doc_path=tmp_path / "doc.dlm",
            store=store,  # type: ignore[arg-type]
            spec=MagicMock(),
            plan=MagicMock(),
            max_steps=100,
            reload_doc=lambda _p: parsed,  # type: ignore[arg-type,return-value]
            retrain=retrain,
        )

        assert result.ran is False
        assert result.new_sections == 0
        retrain.assert_not_called()


class TestCycleRunsOnChange:
    def test_fires_retrain_with_bounded_max_steps(self, tmp_path: Path) -> None:
        new_sec = _section("fresh content")
        parsed = SimpleNamespace(sections=[new_sec])
        manifest = Manifest(
            dlm_id="01HZXY",
            base_model="smollm2-135m",
            content_hashes={},  # empty — the section is new
        )
        store = _store_stub(manifest, tmp_path)

        fake_run_result = SimpleNamespace(
            final_train_loss=1.23,
            final_val_loss=1.10,
            steps=50,
        )
        retrain = MagicMock(return_value=fake_run_result)

        result = do_one_cycle(
            doc_path=tmp_path / "doc.dlm",
            store=store,  # type: ignore[arg-type]
            spec=MagicMock(),
            plan=MagicMock(),
            max_steps=75,
            reload_doc=lambda _p: parsed,  # type: ignore[arg-type,return-value]
            retrain=retrain,
        )

        assert result.ran is True
        assert result.new_sections == 1
        assert result.run_result is fake_run_result
        retrain.assert_called_once()
        _args, kwargs = retrain.call_args
        assert kwargs["max_steps"] == 75

    def test_reports_removed_sections(self, tmp_path: Path) -> None:
        """Sections dropped from the doc show up in CycleResult."""
        new_sec = _section("something new")
        parsed = SimpleNamespace(sections=[new_sec])
        manifest = Manifest(
            dlm_id="01HZXY",
            base_model="smollm2-135m",
            content_hashes={"old_sec": "old_sec"},
        )
        store = _store_stub(manifest, tmp_path)

        result = do_one_cycle(
            doc_path=tmp_path / "doc.dlm",
            store=store,  # type: ignore[arg-type]
            spec=MagicMock(),
            plan=MagicMock(),
            max_steps=100,
            reload_doc=lambda _p: parsed,  # type: ignore[arg-type,return-value]
            retrain=MagicMock(return_value=SimpleNamespace()),
        )

        assert result.removed_sections == 1
