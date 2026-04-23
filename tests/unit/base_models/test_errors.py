"""Typed error formatting + ProbeReport aggregation."""

from __future__ import annotations

from dlm.base_models import (
    BaseModelError,
    GatedModelError,
    ProbeFailedError,
    ProbeReport,
    ProbeResult,
    UnknownBaseModelError,
)


class TestUnknownBaseModelError:
    def test_short_preview_without_tail(self) -> None:
        err = UnknownBaseModelError("nope", ("a", "b", "c"))
        msg = str(err)
        assert "nope" in msg
        assert "a, b, c" in msg
        assert "more" not in msg

    def test_long_preview_truncates_with_count(self) -> None:
        keys = tuple(f"k{i}" for i in range(12))
        err = UnknownBaseModelError("nope", keys)
        msg = str(err)
        assert "7 more" in msg
        assert "k0, k1, k2, k3, k4" in msg

    def test_hf_escape_hint_mentioned(self) -> None:
        err = UnknownBaseModelError("nope", ())
        assert "hf:org/name" in str(err)


class TestProbeFailedError:
    def test_message_lists_failures_only(self) -> None:
        results = [
            ProbeResult(name="architecture", passed=True, detail="ok"),
            ProbeResult(name="chat_template", passed=False, detail="missing"),
            ProbeResult(name="gguf_arch", passed=False, detail="unknown: foo"),
        ]
        err = ProbeFailedError("org/x", results)
        msg = str(err)
        assert "2 of 3" in msg
        assert "chat_template" in msg
        assert "gguf_arch" in msg
        assert "architecture" not in msg.split(": ", 1)[1]


class TestGatedModelError:
    def test_mentions_license_url(self) -> None:
        err = GatedModelError("meta-llama/Llama-3.2-1B-Instruct", "https://example.com/license")
        msg = str(err)
        assert "meta-llama/Llama-3.2-1B-Instruct" in msg
        assert "https://example.com/license" in msg
        assert "--i-accept-license" in msg
        assert "requires license acceptance" in msg

    def test_no_license_url_still_renders(self) -> None:
        err = GatedModelError("org/gated", None)
        msg = str(err)
        assert "org/gated" in msg


class TestProbeReport:
    def test_empty_report_passes_trivially(self) -> None:
        report = ProbeReport(hf_id="org/x", results=())
        assert report.passed is True
        assert report.failures == ()
        assert report.skipped == ()

    def test_all_pass(self) -> None:
        report = ProbeReport(
            hf_id="org/x",
            results=(
                ProbeResult(name="a", passed=True, detail="ok"),
                ProbeResult(name="b", passed=True, detail="ok"),
            ),
        )
        assert report.passed is True

    def test_skipped_counts_as_pass_for_aggregate(self) -> None:
        report = ProbeReport(
            hf_id="org/x",
            results=(
                ProbeResult(name="a", passed=True, detail="skipped: …", skipped=True),
                ProbeResult(name="b", passed=True, detail="ok"),
            ),
        )
        assert report.passed is True
        assert len(report.skipped) == 1

    def test_any_failure_fails_aggregate(self) -> None:
        report = ProbeReport(
            hf_id="org/x",
            results=(
                ProbeResult(name="a", passed=True, detail="ok"),
                ProbeResult(name="b", passed=False, detail="bad"),
            ),
        )
        assert report.passed is False
        assert len(report.failures) == 1


class TestHierarchy:
    def test_all_subclass_base_model_error(self) -> None:
        assert issubclass(UnknownBaseModelError, BaseModelError)
        assert issubclass(GatedModelError, BaseModelError)
        assert issubclass(ProbeFailedError, BaseModelError)
