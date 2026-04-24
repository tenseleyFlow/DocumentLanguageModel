"""`--gpus` flag parsing + resolution."""

from __future__ import annotations

import pytest

from dlm.train.distributed.gpus import GpuSpec, UnsupportedGpuSpecError, parse_gpus


class TestParseGpus:
    def test_none_raises_empty(self) -> None:
        with pytest.raises(UnsupportedGpuSpecError, match="empty"):
            parse_gpus(None)  # type: ignore[arg-type]

    def test_all_case_insensitive(self) -> None:
        for value in ("all", "ALL", "All"):
            spec = parse_gpus(value)
            assert spec == GpuSpec(kind="all", value=None)

    def test_integer_count(self) -> None:
        assert parse_gpus("2") == GpuSpec(kind="count", value=2)
        assert parse_gpus("  4  ") == GpuSpec(kind="count", value=4)

    def test_comma_list(self) -> None:
        assert parse_gpus("0,1") == GpuSpec(kind="list", value=(0, 1))
        assert parse_gpus("0, 1, 3") == GpuSpec(kind="list", value=(0, 1, 3))

    def test_empty_raises(self) -> None:
        with pytest.raises(UnsupportedGpuSpecError, match="empty"):
            parse_gpus("")
        with pytest.raises(UnsupportedGpuSpecError, match="empty"):
            parse_gpus("   ")

    def test_negative_list_rejected(self) -> None:
        with pytest.raises(UnsupportedGpuSpecError, match="negative"):
            parse_gpus("0,-1")

    def test_non_integer_list_rejected(self) -> None:
        with pytest.raises(UnsupportedGpuSpecError, match="non-integer"):
            parse_gpus("0,foo,1")

    def test_empty_comma_list_rejected(self) -> None:
        with pytest.raises(UnsupportedGpuSpecError, match="is empty"):
            parse_gpus(", ,")

    def test_malformed_scalar_rejected(self) -> None:
        with pytest.raises(UnsupportedGpuSpecError, match="not `all`"):
            parse_gpus("xyz")


class TestResolveGpuSpec:
    def test_list_returns_requested_ids(self) -> None:
        spec = GpuSpec(kind="list", value=(0, 2))
        assert spec.resolve(device_count=4) == (0, 2)

    def test_all_returns_full_range(self) -> None:
        spec = GpuSpec(kind="all", value=None)
        assert spec.resolve(device_count=3) == (0, 1, 2)

    def test_count_returns_prefix(self) -> None:
        spec = GpuSpec(kind="count", value=2)
        assert spec.resolve(device_count=4) == (0, 1)

    def test_count_exceeding_visible_raises(self) -> None:
        spec = GpuSpec(kind="count", value=4)
        with pytest.raises(UnsupportedGpuSpecError, match="exceeds"):
            spec.resolve(device_count=2)

    def test_count_zero_rejected(self) -> None:
        spec = GpuSpec(kind="count", value=0)
        with pytest.raises(UnsupportedGpuSpecError, match=">= 1"):
            spec.resolve(device_count=4)

    def test_list_out_of_range_raises(self) -> None:
        spec = GpuSpec(kind="list", value=(0, 5))
        with pytest.raises(UnsupportedGpuSpecError, match="out-of-range"):
            spec.resolve(device_count=2)

    def test_list_duplicate_raises(self) -> None:
        spec = GpuSpec(kind="list", value=(0, 1, 1))
        with pytest.raises(UnsupportedGpuSpecError, match="duplicate"):
            spec.resolve(device_count=4)

    def test_no_visible_devices_raises(self) -> None:
        spec = GpuSpec(kind="all", value=None)
        with pytest.raises(UnsupportedGpuSpecError, match="at least 1"):
            spec.resolve(device_count=0)
