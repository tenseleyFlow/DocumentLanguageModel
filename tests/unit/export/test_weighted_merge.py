"""Spec parser + validator for `--adapter-mix`."""

from __future__ import annotations

import pytest

from dlm.export.weighted_merge import (
    InvalidMixSpecError,
    MixEntry,
    parse_mix_spec,
    validate_mix_against_declared,
)


class TestParseMixSpec:
    def test_single_entry(self) -> None:
        out = parse_mix_spec("knowledge:1.0")
        assert out == [MixEntry(name="knowledge", weight=1.0)]

    def test_two_entries(self) -> None:
        out = parse_mix_spec("knowledge:1.0,tone:0.5")
        assert out == [
            MixEntry(name="knowledge", weight=1.0),
            MixEntry(name="tone", weight=0.5),
        ]

    def test_preserves_order(self) -> None:
        out = parse_mix_spec("tone:0.3,knowledge:0.7,snark:0.1")
        assert [e.name for e in out] == ["tone", "knowledge", "snark"]

    def test_strips_whitespace(self) -> None:
        out = parse_mix_spec("  knowledge : 1.0 , tone:0.5 ")
        assert out[0].name == "knowledge"
        assert out[0].weight == pytest.approx(1.0)

    def test_integer_weight_accepted(self) -> None:
        out = parse_mix_spec("knowledge:2")
        assert out[0].weight == pytest.approx(2.0)

    def test_zero_weight_allowed(self) -> None:
        # Zero-weight entries are degenerate but legal — easier to
        # refuse at the heavy path than complicate parsing.
        out = parse_mix_spec("knowledge:0")
        assert out[0].weight == 0.0


class TestParseMixSpecErrors:
    def test_empty_string_rejected(self) -> None:
        with pytest.raises(InvalidMixSpecError, match="empty spec"):
            parse_mix_spec("")

    def test_whitespace_only_rejected(self) -> None:
        with pytest.raises(InvalidMixSpecError, match="empty spec"):
            parse_mix_spec("   ")

    def test_missing_weight_rejected(self) -> None:
        with pytest.raises(InvalidMixSpecError, match="missing a weight"):
            parse_mix_spec("knowledge")

    def test_non_numeric_weight_rejected(self) -> None:
        with pytest.raises(InvalidMixSpecError, match="is not a number"):
            parse_mix_spec("knowledge:heavy")

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(InvalidMixSpecError, match="is negative"):
            parse_mix_spec("knowledge:-0.5")

    @pytest.mark.parametrize(
        "bad_name",
        ["Knowledge", "1tone", "tone-name", "_tone", "tone name"],
    )
    def test_invalid_name_rejected(self, bad_name: str) -> None:
        with pytest.raises(InvalidMixSpecError, match="is not valid"):
            parse_mix_spec(f"{bad_name}:1.0")

    def test_duplicate_name_rejected(self) -> None:
        with pytest.raises(InvalidMixSpecError, match="appears twice"):
            parse_mix_spec("knowledge:0.5,knowledge:0.5")

    def test_trailing_comma_rejected(self) -> None:
        with pytest.raises(InvalidMixSpecError, match="empty entry"):
            parse_mix_spec("knowledge:1.0,")


class TestValidateAgainstDeclared:
    def test_all_known_names_pass(self) -> None:
        entries = parse_mix_spec("knowledge:1.0,tone:0.5")
        validate_mix_against_declared(entries, {"knowledge", "tone"})  # no raise

    def test_unknown_name_rejected(self) -> None:
        entries = parse_mix_spec("knowledge:1.0,ghost:0.5")
        with pytest.raises(InvalidMixSpecError, match="ghost"):
            validate_mix_against_declared(entries, {"knowledge", "tone"})

    def test_error_lists_declared(self) -> None:
        entries = parse_mix_spec("ghost:0.5")
        with pytest.raises(InvalidMixSpecError, match="declared"):
            validate_mix_against_declared(entries, {"knowledge", "tone"})


class TestResolveFirstSourcePath:
    """Audit-07 B2 — ensure the helper used for tokenizer-copy points
    to a real committed adapter dir."""

    def test_returns_first_entry_path(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from dlm.export.weighted_merge import resolve_first_source_path
        from dlm.store.paths import StorePath

        store = StorePath(root=tmp_path / "s")
        store.ensure_layout()
        # Commit v0001 for both knowledge and tone.
        for name in ("knowledge", "tone"):
            store.ensure_adapter_layout(name)
            v1 = store.adapter_version_for(name, 1)
            v1.mkdir(parents=True)
            store.set_current_adapter_for(name, v1)

        entries = parse_mix_spec("knowledge:1.0,tone:0.5")
        out = resolve_first_source_path(store, entries)
        assert out == store.adapter_version_for("knowledge", 1).resolve()

    def test_empty_entries_rejected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from dlm.export.weighted_merge import resolve_first_source_path
        from dlm.store.paths import StorePath

        store = StorePath(root=tmp_path / "s")
        store.ensure_layout()
        with pytest.raises(InvalidMixSpecError, match="empty mix"):
            resolve_first_source_path(store, [])

    def test_missing_adapter_raises_export_error(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from dlm.export.errors import ExportError
        from dlm.export.weighted_merge import resolve_first_source_path
        from dlm.store.paths import StorePath

        store = StorePath(root=tmp_path / "s")
        store.ensure_layout()
        entries = parse_mix_spec("knowledge:1.0")
        with pytest.raises(ExportError, match="has no committed version"):
            resolve_first_source_path(store, entries)
