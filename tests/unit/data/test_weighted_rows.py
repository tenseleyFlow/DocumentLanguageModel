"""Tag-weighted row expansion — math + determinism + edge cases."""

from __future__ import annotations

from dlm.data.weighted_rows import (
    expand_rows_by_weight,
    resolve_row_weight,
    weight_distribution,
)


def _row(section_id: str, tags: dict[str, str] | None = None) -> dict:
    return {
        "text": f"body-{section_id}",
        "_dlm_section_id": section_id,
        "_dlm_row_tags": tags or {},
    }


class TestResolveRowWeight:
    def test_empty_weights_gives_unity(self) -> None:
        assert resolve_row_weight({"a": "b"}, {}) == 1.0

    def test_empty_tags_gives_unity(self) -> None:
        assert resolve_row_weight({}, {"a": {"b": 2.0}}) == 1.0

    def test_single_matching_tag_scales(self) -> None:
        assert resolve_row_weight({"docstring": "true"}, {"docstring": {"true": 2.5}}) == 2.5

    def test_multiple_keys_multiply(self) -> None:
        weights = {"a": {"x": 2.0}, "b": {"y": 0.5}}
        assert resolve_row_weight({"a": "x", "b": "y"}, weights) == 1.0

    def test_unmatched_value_does_not_scale(self) -> None:
        assert resolve_row_weight({"a": "z"}, {"a": {"x": 2.0}}) == 1.0

    def test_unmatched_key_does_not_scale(self) -> None:
        assert resolve_row_weight({"b": "x"}, {"a": {"x": 2.0}}) == 1.0


class TestExpandRowsByWeight:
    def test_empty_weights_returns_shallow_copy(self) -> None:
        rows = [_row("01"), _row("02")]
        out = expand_rows_by_weight(rows, {}, seed=42)
        assert out == rows
        assert out is not rows  # shallow copy

    def test_weight_one_is_noop(self) -> None:
        rows = [_row("01", {"k": "v"})]
        out = expand_rows_by_weight(rows, {"k": {"v": 1.0}}, seed=42)
        assert len(out) == 1

    def test_integer_weight_repeats(self) -> None:
        rows = [_row("01", {"k": "v"})]
        out = expand_rows_by_weight(rows, {"k": {"v": 3.0}}, seed=42)
        assert len(out) == 3
        # All copies share the same section_id.
        assert {r["_dlm_section_id"] for r in out} == {"01"}

    def test_zero_weight_drops(self) -> None:
        rows = [_row("01", {"k": "v"}), _row("02", {"other": "x"})]
        out = expand_rows_by_weight(rows, {"k": {"v": 0.0}}, seed=42)
        # Row 02 is untagged for `k` so it keeps weight 1.
        assert len(out) == 1
        assert out[0]["_dlm_section_id"] == "02"

    def test_fractional_weight_is_deterministic(self) -> None:
        rows = [_row(f"{i:02d}", {"k": "v"}) for i in range(100)]
        out1 = expand_rows_by_weight(rows, {"k": {"v": 0.5}}, seed=42)
        out2 = expand_rows_by_weight(rows, {"k": {"v": 0.5}}, seed=42)
        assert [r["_dlm_section_id"] for r in out1] == [r["_dlm_section_id"] for r in out2]

    def test_fractional_weight_approximates_probability(self) -> None:
        rows = [_row(f"{i:04d}", {"k": "v"}) for i in range(1000)]
        out = expand_rows_by_weight(rows, {"k": {"v": 0.5}}, seed=42)
        # 50% keep rate with 1000 rows should land within ±10% of 500.
        assert 450 <= len(out) <= 550

    def test_weight_between_one_and_two_includes_integer_plus_fractional(self) -> None:
        rows = [_row(f"{i:04d}", {"k": "v"}) for i in range(1000)]
        out = expand_rows_by_weight(rows, {"k": {"v": 1.5}}, seed=42)
        # Every row gets 1 copy unconditionally plus ~50% get a 2nd.
        assert 1450 <= len(out) <= 1550

    def test_different_seeds_yield_different_expansions(self) -> None:
        rows = [_row(f"{i:02d}", {"k": "v"}) for i in range(100)]
        out1 = expand_rows_by_weight(rows, {"k": {"v": 0.5}}, seed=42)
        out2 = expand_rows_by_weight(rows, {"k": {"v": 0.5}}, seed=43)
        # Not byte-identical — different seeds drive different Bernoulli rolls.
        ids1 = [r["_dlm_section_id"] for r in out1]
        ids2 = [r["_dlm_section_id"] for r in out2]
        assert ids1 != ids2

    def test_rows_without_tags_get_unity_weight(self) -> None:
        rows = [_row("01", {}), _row("02", {"k": "v"})]
        out = expand_rows_by_weight(rows, {"k": {"v": 3.0}}, seed=42)
        # Row 01 = 1 copy (unity). Row 02 = 3 copies.
        assert len(out) == 4

    def test_multiplicative_composition(self) -> None:
        rows = [_row("01", {"a": "x", "b": "y"})]
        weights = {"a": {"x": 2.0}, "b": {"y": 3.0}}
        out = expand_rows_by_weight(rows, weights, seed=42)
        # 2.0 × 3.0 = 6 copies.
        assert len(out) == 6


class TestWeightDistribution:
    def test_empty_rows_empty_dist(self) -> None:
        assert weight_distribution([]) == {}

    def test_untagged_rows_produce_empty_dist(self) -> None:
        rows = [_row("01"), _row("02")]
        assert weight_distribution(rows) == {}

    def test_counts_per_tag_value(self) -> None:
        rows = [
            _row("01", {"lang": "py", "gen": "true"}),
            _row("02", {"lang": "py", "gen": "false"}),
            _row("03", {"lang": "rs", "gen": "false"}),
        ]
        dist = weight_distribution(rows)
        assert dist == {
            "lang": {"py": 2, "rs": 1},
            "gen": {"true": 1, "false": 2},
        }
