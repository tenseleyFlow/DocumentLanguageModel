"""`split` determinism, stability, and error contract."""

from __future__ import annotations

import pytest

from dlm.data.splitter import split


def _rows(n: int, sections: int = 5) -> list[dict[str, object]]:
    return [{"text": f"row{i}", "_dlm_section_id": f"sec{i % sections}"} for i in range(n)]


class TestDeterminism:
    def test_same_seed_same_split(self) -> None:
        rows = _rows(200)
        t1, v1 = split(rows, val_frac=0.2, seed=7)
        t2, v2 = split(rows, val_frac=0.2, seed=7)
        assert list(t1) == list(t2)
        assert list(v1) == list(v2)

    def test_different_seed_different_split(self) -> None:
        rows = _rows(200)
        t1, _ = split(rows, val_frac=0.2, seed=7)
        t2, _ = split(rows, val_frac=0.2, seed=8)
        assert list(t1) != list(t2)

    def test_val_frac_is_approximate(self) -> None:
        rows = _rows(1000)
        _, val = split(rows, val_frac=0.1, seed=42)
        # Hash-based assignment: within ~3% on 1000 rows.
        assert 70 <= len(val) <= 130


class TestStabilityAcrossAdds:
    def test_adding_new_section_does_not_reshuffle(self) -> None:
        original = _rows(100)
        train_before, val_before = split(original, val_frac=0.2, seed=42)

        # Add a new section's rows. Existing rows should keep their bucket.
        extended = original + [
            {"text": "new1", "_dlm_section_id": "brand-new-section"},
            {"text": "new2", "_dlm_section_id": "brand-new-section"},
        ]
        train_after, val_after = split(extended, val_frac=0.2, seed=42)

        orig_train_texts = {r["text"] for r in train_before}
        after_train_texts = {r["text"] for r in train_after}
        assert orig_train_texts.issubset(after_train_texts)

        orig_val_texts = {r["text"] for r in val_before}
        after_val_texts = {r["text"] for r in val_after}
        assert orig_val_texts.issubset(after_val_texts)

    def test_section_sub_index_spreads_qa_pairs(self) -> None:
        """Ten Q/A pairs from a single INSTRUCTION block should not all end
        up in the same bucket."""
        rows = [{"text": f"qa{i}", "_dlm_section_id": "only-one-section"} for i in range(50)]
        train, val = split(rows, val_frac=0.2, seed=42)
        # With 50 rows in a single section, val should have at least a few.
        assert len(val) > 0
        assert len(train) > 0


class TestContract:
    def test_val_frac_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="val_frac"):
            split(_rows(10), val_frac=0.0, seed=1)

    def test_val_frac_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="val_frac"):
            split(_rows(10), val_frac=1.0, seed=1)

    def test_row_without_section_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="_dlm_section_id"):
            split([{"text": "orphan"}], val_frac=0.1, seed=1)

    def test_empty_rows_produces_empty_datasets(self) -> None:
        train, val = split([], val_frac=0.1, seed=1)
        assert len(train) == 0
        assert len(val) == 0
