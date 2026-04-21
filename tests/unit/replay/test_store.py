"""`ReplayStore` facade — append, load, sample_rows integration."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

from dlm.replay import ReplayStore, SectionSnapshot


def _store(tmp: Path) -> ReplayStore:
    return ReplayStore.at(tmp / "corpus.zst", tmp / "index.json")


def _snap(sid: str, section_type: str, content: str, *, added: datetime) -> SectionSnapshot:
    return SectionSnapshot(
        section_id=sid,
        section_type=section_type,  # type: ignore[arg-type]
        content=content,
        first_seen_at=added,
        last_seen_at=added,
    )


class TestStoreLifecycle:
    def test_append_persists_index(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        snap = _snap("a" * 16, "prose", "hello", added=datetime(2026, 1, 1))
        s.append(snap)
        # Re-open same paths and confirm index survived.
        again = _store(tmp_path)
        entries = again.load()
        assert len(entries) == 1
        assert entries[0].section_id == "a" * 16

    def test_append_many_batches(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        snaps = [
            _snap(f"{i:016x}", "prose", f"c{i}", added=datetime(2026, 1, 1) + timedelta(days=i))
            for i in range(5)
        ]
        entries = s.append_many(snaps)
        assert len(entries) == 5
        assert len(s.load()) == 5


class TestSampleRows:
    def test_prose_expands_to_text_row(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append(_snap("a" * 16, "prose", "hello", added=datetime(2026, 1, 1)))
        rows = s.sample_rows(k=1, now=datetime(2026, 4, 1), rng=random.Random(0))
        assert rows[0]["text"] == "hello"
        assert rows[0]["_dlm_section_id"].startswith("replay:")

    def test_instruction_expands_to_messages_rows(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        body = "### Q\nq1\n### A\na1\n\n### Q\nq2\n### A\na2"
        s.append(_snap("a" * 16, "instruction", body, added=datetime(2026, 1, 1)))
        rows = s.sample_rows(k=10, now=datetime(2026, 4, 1), rng=random.Random(0))
        assert len(rows) == 2
        assert all("messages" in r for r in rows)
        assert rows[0]["messages"][0]["content"] == "q1"

    def test_preference_expands_to_pref_rows(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        body = "### Prompt\np\n### Chosen\nc\n### Rejected\nr"
        s.append(_snap("a" * 16, "preference", body, added=datetime(2026, 1, 1)))
        rows = s.sample_rows(k=1, now=datetime(2026, 4, 1), rng=random.Random(0))
        assert rows[0] == {
            "prompt": "p",
            "chosen": "c",
            "rejected": "r",
            "_dlm_section_id": rows[0]["_dlm_section_id"],
        }

    def test_empty_prose_dropped(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append(_snap("a" * 16, "prose", "   \n", added=datetime(2026, 1, 1)))
        rows = s.sample_rows(k=1, now=datetime(2026, 4, 1), rng=random.Random(0))
        assert rows == []

    def test_replay_sid_prevents_collision(self, tmp_path: Path) -> None:
        """Replay row's `_dlm_section_id` must differ from the raw section id."""
        s = _store(tmp_path)
        raw_sid = "a" * 16
        s.append(_snap(raw_sid, "prose", "x", added=datetime(2026, 1, 1)))
        rows = s.sample_rows(k=1, now=datetime(2026, 4, 1), rng=random.Random(0))
        assert rows[0]["_dlm_section_id"] != raw_sid
        assert raw_sid in rows[0]["_dlm_section_id"]


_PREF_BODY_A = "### Prompt\nqA\n### Chosen\ncA\n### Rejected\nrA"
_PREF_BODY_B = "### Prompt\nqB\n### Chosen\ncB\n### Rejected\nrB"


class TestSamplePreferenceRows:
    def test_empty_corpus_returns_empty(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        rows = s.sample_preference_rows(k=5, now=datetime(2026, 4, 1), rng=random.Random(0))
        assert rows == []

    def test_corpus_with_no_preferences_returns_empty(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append(_snap("a" * 16, "prose", "hello", added=datetime(2026, 1, 1)))
        s.append(_snap("b" * 16, "instruction", "### Q\nq\n### A\na", added=datetime(2026, 1, 1)))
        rows = s.sample_preference_rows(k=2, now=datetime(2026, 4, 1), rng=random.Random(0))
        assert rows == []

    def test_filters_to_preferences_only(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append(_snap("a" * 16, "prose", "prose body", added=datetime(2026, 1, 1)))
        s.append(_snap("b" * 16, "preference", _PREF_BODY_A, added=datetime(2026, 1, 1)))
        s.append(_snap("c" * 16, "preference", _PREF_BODY_B, added=datetime(2026, 1, 2)))
        rows = s.sample_preference_rows(k=10, now=datetime(2026, 4, 1), rng=random.Random(0))
        assert len(rows) == 2
        assert {r["prompt"] for r in rows} == {"qA", "qB"}
        assert all("chosen" in r and "rejected" in r for r in rows)

    def test_k_smaller_than_available_samples_exactly_k(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        for i in range(5):
            sid = f"{i:016x}"
            body = f"### Prompt\nq{i}\n### Chosen\nc{i}\n### Rejected\nr{i}"
            s.append(_snap(sid, "preference", body, added=datetime(2026, 1, 1)))
        rows = s.sample_preference_rows(k=2, now=datetime(2026, 4, 1), rng=random.Random(0))
        assert len(rows) == 2

    def test_replay_sid_prefix_applied(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append(_snap("a" * 16, "preference", _PREF_BODY_A, added=datetime(2026, 1, 1)))
        rows = s.sample_preference_rows(k=1, now=datetime(2026, 4, 1), rng=random.Random(0))
        assert rows[0]["_dlm_section_id"].startswith("replay:")
