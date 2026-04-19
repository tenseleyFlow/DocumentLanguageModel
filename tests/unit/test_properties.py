"""Hypothesis property tests for invariants across dlm.* (audit-04 T4).

These aren't replacements for example-based tests — they're guardrails
for properties that hold for *all* inputs. Keep them cheap: the suite
runs inside the default pytest invocation, so each property's shrinking
budget is tuned small.
"""

from __future__ import annotations

import string
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dlm.export.errors import PreflightError, UnsafeMergeError
from dlm.export.merge import check_merge_safety
from dlm.export.plan import ExportPlan
from dlm.export.tokenizer_sync import read_gguf_vocab_size
from dlm.io.ulid import mint_ulid
from dlm.pack.integrity import rollup_sha256

# --- ULID ---------------------------------------------------------------------


class TestUlidMonotonicity:
    """Crockford ULIDs are 26 chars, time-prefix monotonic within a ms."""

    @given(n=st.integers(min_value=2, max_value=32))
    def test_each_ulid_is_26_chars(self, n: int) -> None:
        for _ in range(n):
            u = mint_ulid()
            assert len(u) == 26

    @given(n=st.integers(min_value=2, max_value=16))
    def test_two_ulids_minted_in_sequence_are_distinct(self, n: int) -> None:
        # Even at the same millisecond, random-component collisions should
        # be astronomically unlikely. Property: no duplicates across a run.
        seen = {mint_ulid() for _ in range(n)}
        assert len(seen) == n

    def test_alphabet_is_crockford(self) -> None:
        # Crockford base32 excludes I, L, O, U. 32 samples gives high
        # confidence that every alphabet slot has been observed.
        allowed = set(string.digits + "ABCDEFGHJKMNPQRSTVWXYZ")
        for _ in range(32):
            u = mint_ulid()
            assert set(u) <= allowed, f"ULID {u} has non-Crockford char"


# --- rollup_sha256 ------------------------------------------------------------


_hex64 = st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)
_relpath = st.text(
    alphabet=string.ascii_lowercase + string.digits + "/",
    min_size=1,
    max_size=20,
).filter(lambda s: ".." not in s and not s.startswith("/") and not s.endswith("/"))


class TestRollupDeterminism:
    """Rollup is content-addressable: input → same hex digest every time."""

    @given(st.dictionaries(_relpath, _hex64, min_size=1, max_size=12))
    def test_rollup_is_deterministic(self, d: dict[str, str]) -> None:
        assert rollup_sha256(d) == rollup_sha256(d)

    @given(st.dictionaries(_relpath, _hex64, min_size=2, max_size=12))
    def test_rollup_is_order_independent(self, d: dict[str, str]) -> None:
        # Reverse-ordered dict → same rollup (sort is the contract).
        reversed_d = dict(reversed(list(d.items())))
        assert rollup_sha256(d) == rollup_sha256(reversed_d)

    @given(st.dictionaries(_relpath, _hex64, min_size=1, max_size=8))
    def test_rollup_changes_with_content(self, d: dict[str, str]) -> None:
        base = rollup_sha256(d)
        tweaked = {**d, "extra_path": "0" * 64}
        assert rollup_sha256(tweaked) != base


# --- merge-safety truth table -------------------------------------------------


class TestMergeSafetyTruthTable:
    """Exhaustive `(merged, dequantize, was_qlora)` sweep.

    Contract (CLAUDE.md pitfall #3):
    - `merged=False`: always safe, regardless of other flags.
    - `merged=True, was_qlora=False`: safe (plain LoRA merge).
    - `merged=True, was_qlora=True, dequantize=False`: REFUSE.
    - `merged=True, was_qlora=True, dequantize=True`: safe (user confirmed).
    """

    @pytest.mark.parametrize("merged", [False, True])
    @pytest.mark.parametrize("dequantize", [False, True])
    @pytest.mark.parametrize("was_qlora", [False, True])
    def test_truth_table(self, merged: bool, dequantize: bool, was_qlora: bool) -> None:
        # `dequantize_confirmed=True` is meaningless without merged=True;
        # `ExportPlan.__post_init__` rejects the combo before we can test it.
        if dequantize and not merged:
            pytest.skip("invalid flag combination: --dequantize without --merged")
        plan = ExportPlan(merged=merged, dequantize_confirmed=dequantize)
        should_refuse = merged and was_qlora and not dequantize
        if should_refuse:
            with pytest.raises(UnsafeMergeError):
                check_merge_safety(plan, was_qlora=was_qlora)
        else:
            check_merge_safety(plan, was_qlora=was_qlora)  # no raise


# --- GGUF parser fuzz ---------------------------------------------------------


class TestGgufParserFuzz:
    """Feed random bytes at `read_gguf_vocab_size`; it must surface a typed
    `PreflightError`, never leak `struct.error` / `MemoryError` / a raw
    decoded string. Audit-04 T6 / B7 defense-in-depth.

    Reusing `tmp_path` across hypothesis iterations is deliberate — we
    overwrite the file each run, and spinning up a fresh dir per-sample
    would dominate the test runtime.
    """

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
    @given(
        payload=st.binary(min_size=0, max_size=256).filter(
            # Skip valid headers — fuzzing valid packets isn't the point.
            lambda b: not b.startswith(b"GGUF")
        )
    )
    def test_random_bytes_raise_preflight_error(self, payload: bytes, tmp_path: Path) -> None:
        path = tmp_path / "fuzz.gguf"
        path.write_bytes(payload)
        with pytest.raises(PreflightError):
            read_gguf_vocab_size(path)

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
    @given(body=st.binary(min_size=0, max_size=256))
    def test_random_body_after_magic_doesnt_crash(self, body: bytes, tmp_path: Path) -> None:
        """Even with valid magic, garbage body → typed error, no crash."""
        path = tmp_path / "fuzz.gguf"
        path.write_bytes(b"GGUF" + body)
        with pytest.raises(PreflightError):
            read_gguf_vocab_size(path)
