"""Three-tier error reporter (Sprint 13)."""

from __future__ import annotations

import pytest

from dlm.cli.reporter import report_exception, run_with_reporter


class TestTier1ParseError:
    def test_parse_error_uses_parse_prefix(self, capsys: pytest.CaptureFixture[str]) -> None:
        from pathlib import Path

        from dlm.doc.errors import FrontmatterError

        exc = FrontmatterError("bad yaml", path=Path("x.dlm"), line=2, col=1)
        code = report_exception(exc)

        assert code == 1
        err = capsys.readouterr().err
        assert "parse:" in err
        assert "bad yaml" in err
        assert "x.dlm" in err


class TestTier2DomainError:
    def test_gated_model_uses_license_prefix(self, capsys: pytest.CaptureFixture[str]) -> None:
        from dlm.base_models.errors import GatedModelError

        exc = GatedModelError("meta-llama/Llama-3.2-1B", "https://example.com")
        code = report_exception(exc)

        assert code == 1
        err = capsys.readouterr().err
        assert "license:" in err

    def test_export_error_uses_export_prefix(self, capsys: pytest.CaptureFixture[str]) -> None:
        from dlm.export.errors import UnsafeMergeError

        exc = UnsafeMergeError("needs --dequantize")
        code = report_exception(exc)

        assert code == 1
        err = capsys.readouterr().err
        assert "export:" in err


class TestPrefixMapping:
    """Each known module prefix gets a distinct colored label."""

    def test_doc_error_prefix(self, capsys: pytest.CaptureFixture[str]) -> None:
        from dlm.doc.errors import UnsupportedMigrationError

        # NB: DlmParseError subclass — tier 1 beats prefix mapping, so this
        # lands under "parse:", not "doc:". The branch is here to prove the
        # inheritance path wins.
        exc = UnsupportedMigrationError("no migrator")
        report_exception(exc)
        err = capsys.readouterr().err
        assert "parse:" in err

    def test_store_error_prefix(self, capsys: pytest.CaptureFixture[str]) -> None:
        from dlm.store.errors import ManifestCorruptError

        exc = ManifestCorruptError(__import__("pathlib").Path("m.json"), "bad")
        report_exception(exc)
        err = capsys.readouterr().err
        assert "store:" in err

    def test_train_error_prefix(self, capsys: pytest.CaptureFixture[str]) -> None:
        from dlm.train.errors import DiskSpaceError

        exc = DiskSpaceError(required_bytes=100_000_000_000, free_bytes=1_000)
        report_exception(exc)
        err = capsys.readouterr().err
        assert "train:" in err

    def test_inference_error_prefix(self, capsys: pytest.CaptureFixture[str]) -> None:
        from dlm.inference.errors import AdapterNotFoundError

        exc = AdapterNotFoundError("no adapter")
        report_exception(exc)
        err = capsys.readouterr().err
        assert "inference:" in err

    def test_base_model_unknown_prefix(self, capsys: pytest.CaptureFixture[str]) -> None:
        from dlm.base_models.errors import UnknownBaseModelError

        exc = UnknownBaseModelError("no-such-key", ("qwen2.5-1.5b", "smollm2-135m"))
        report_exception(exc)
        err = capsys.readouterr().err
        assert "base_model:" in err


class TestTier3Uncaught:
    def test_unknown_exception_gets_verbose_hint(self, capsys: pytest.CaptureFixture[str]) -> None:
        exc = RuntimeError("something went wrong")
        code = report_exception(exc)

        assert code == 1
        err = capsys.readouterr().err
        assert "RuntimeError" in err
        assert "--verbose" in err

    def test_verbose_env_surfaces_traceback(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DLM_VERBOSE", "1")
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            report_exception(exc)
        err = capsys.readouterr().err
        # Traceback present.
        assert "boom" in err


class TestRunWrapper:
    def test_system_exit_code_propagated(self) -> None:
        def app() -> None:
            raise SystemExit(7)

        assert run_with_reporter(app) == 7

    def test_normal_exit_returns_zero(self) -> None:
        def app() -> None:
            return None

        assert run_with_reporter(app) == 0

    def test_keyboard_interrupt_returns_130(self) -> None:
        def app() -> None:
            raise KeyboardInterrupt

        assert run_with_reporter(app) == 130

    def test_unexpected_exception_caught(self, capsys: pytest.CaptureFixture[str]) -> None:
        def app() -> None:
            raise ValueError("nope")

        code = run_with_reporter(app)
        assert code == 1
        err = capsys.readouterr().err
        assert "ValueError" in err

    def test_system_exit_none_returns_zero(self) -> None:
        def app() -> None:
            raise SystemExit(None)

        assert run_with_reporter(app) == 0

    def test_system_exit_string_prints_and_fails(self, capsys: pytest.CaptureFixture[str]) -> None:
        def app() -> None:
            raise SystemExit("goodbye")

        assert run_with_reporter(app) == 1
        assert "goodbye" in capsys.readouterr().err
