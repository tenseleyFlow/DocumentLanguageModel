"""Tests for staged preference pending-plan helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dlm.doc.sections import Section, SectionType
from dlm.preference.pending import (
    PendingPreferencePlanError,
    _optional_float,
    _optional_int,
    _optional_str,
    _section_from_payload,
    clear_pending_plan,
    load_pending_plan,
    pending_plan_path,
    save_pending_plan,
)
from dlm.store.paths import for_dlm

_DLM_ID = "01KPQ9X1000000000000000000"


def _mined_pref(
    *,
    prompt: str = "question?",
    chosen: str = "better",
    rejected: str = "worse",
    run_id: int = 7,
) -> Section:
    body = f"### Prompt\n{prompt}\n### Chosen\n{chosen}\n### Rejected\n{rejected}"
    return Section(
        type=SectionType.PREFERENCE,
        content=body,
        start_line=12,
        adapter="tone",
        tags={"topic": "blas"},
        auto_mined=True,
        judge_name="sway:preference_judge",
        judge_score_chosen=0.9,
        judge_score_rejected=0.1,
        mined_at="2026-04-23T20:00:00Z",
        mined_run_id=run_id,
    )


def _image() -> Section:
    return Section(
        type=SectionType.IMAGE,
        content="A DGEMM block diagram.",
        media_path="diagram.png",
        media_alt="DGEMM diagram",
        media_blob_sha="ab" * 32,
    )


class TestPendingPlan:
    def test_pending_path_round_trip_and_clear(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        source_path = tmp_path / "doc.dlm"
        source_path.write_text("stub", encoding="utf-8")
        store = for_dlm(_DLM_ID, home=home)

        path = pending_plan_path(store)
        assert path == home / "store" / _DLM_ID / "preference" / "pending.json"

        saved = save_pending_plan(
            store,
            source_path=source_path,
            sections=[_mined_pref(), _image()],
        )
        raw = json.loads(path.read_text(encoding="utf-8"))
        loaded = load_pending_plan(store)

        assert saved.source_path == source_path.resolve()
        assert saved.created_at.endswith("Z")
        assert raw["schema_version"] == 1
        assert raw["source_path"] == str(source_path.resolve())
        assert loaded == saved
        assert clear_pending_plan(store) is True
        assert clear_pending_plan(store) is False
        assert load_pending_plan(store) is None

    def test_load_returns_none_when_plan_absent(self, tmp_path: Path) -> None:
        store = for_dlm(_DLM_ID, home=tmp_path / "home")

        assert load_pending_plan(store) is None

    def test_load_rejects_unreadable_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = for_dlm(_DLM_ID, home=tmp_path / "home")
        path = pending_plan_path(store)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

        def _raise(_self: Path, *, encoding: str) -> str:
            _ = encoding
            raise OSError("boom")

        monkeypatch.setattr(Path, "read_text", _raise)
        with pytest.raises(
            PendingPreferencePlanError, match="could not read staged preference plan"
        ):
            load_pending_plan(store)

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            (["not", "an", "object"], "must be a JSON object"),
            ({"schema_version": 2}, "unsupported staged preference plan schema_version=2"),
            (
                {"schema_version": 1, "created_at": "2026-04-24T20:00:00Z", "sections": []},
                "missing source_path",
            ),
            (
                {"schema_version": 1, "source_path": "/tmp/doc.dlm", "sections": []},
                "missing created_at",
            ),
            (
                {
                    "schema_version": 1,
                    "source_path": "/tmp/doc.dlm",
                    "created_at": "2026-04-24T20:00:00Z",
                },
                "missing sections",
            ),
            (
                {
                    "schema_version": 1,
                    "source_path": "/tmp/doc.dlm",
                    "created_at": "2026-04-24T20:00:00Z",
                    "sections": [{"content": "oops"}],
                },
                "invalid section payload at index 0",
            ),
        ],
    )
    def test_load_rejects_invalid_payloads(
        self, tmp_path: Path, payload: object, message: str
    ) -> None:
        store = for_dlm(_DLM_ID, home=tmp_path / "home")
        path = pending_plan_path(store)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(PendingPreferencePlanError, match=message):
            load_pending_plan(store)

    def test_load_rejects_invalid_json(self, tmp_path: Path) -> None:
        store = for_dlm(_DLM_ID, home=tmp_path / "home")
        path = pending_plan_path(store)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(
            PendingPreferencePlanError, match="staged preference plan is not valid JSON"
        ):
            load_pending_plan(store)


class TestPendingPayloadHelpers:
    def test_section_from_payload_validates_tags_and_optional_types(self) -> None:
        with pytest.raises(TypeError, match="expected object, got list"):
            _section_from_payload([])

        with pytest.raises(TypeError, match="tags must be an object"):
            _section_from_payload({"type": "preference", "content": "x", "tags": []})

        with pytest.raises(TypeError, match="tags keys and values must be strings"):
            _section_from_payload({"type": "preference", "content": "x", "tags": {"topic": 1}})

        with pytest.raises(TypeError, match="expected float or null"):
            _section_from_payload(
                {"type": "preference", "content": "x", "judge_score_chosen": True}
            )

        with pytest.raises(TypeError, match="expected int or null"):
            _section_from_payload({"type": "preference", "content": "x", "mined_run_id": True})

    def test_optional_helpers_accept_none_and_reject_wrong_types(self) -> None:
        assert _optional_str(None) is None
        assert _optional_str("ok") == "ok"
        assert _optional_float(None) is None
        assert _optional_float(1) == 1.0
        assert _optional_int(None) is None
        assert _optional_int(7) == 7

        with pytest.raises(TypeError, match="expected string or null"):
            _optional_str(7)

        with pytest.raises(TypeError, match="expected float or null"):
            _optional_float(True)

        with pytest.raises(TypeError, match="expected int or null"):
            _optional_int(True)
