"""Backend selection: auto / pytorch / mlx routing + refusal modes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dlm.inference.backends.select import (
    UnsupportedBackendError,
    build_backend,
    select_backend,
)


class TestSelectBackendExplicit:
    def test_pytorch_always_selected(self) -> None:
        # Explicit pytorch never triggers the mlx probe.
        with patch("dlm.inference.backends.select.mlx_available") as m_avail:
            assert select_backend("pytorch") == "pytorch"
            m_avail.assert_not_called()

    def test_mlx_on_apple_silicon_with_mlx_installed(self) -> None:
        with (
            patch("dlm.inference.backends.select.is_apple_silicon", return_value=True),
            patch("dlm.inference.backends.select.mlx_available", return_value=True),
        ):
            assert select_backend("mlx") == "mlx"

    def test_mlx_off_platform_raises_clear_error(self) -> None:
        with patch("dlm.inference.backends.select.is_apple_silicon", return_value=False):
            with pytest.raises(UnsupportedBackendError, match="Apple Silicon"):
                select_backend("mlx")

    def test_mlx_apple_silicon_without_extra_raises(self) -> None:
        with (
            patch("dlm.inference.backends.select.is_apple_silicon", return_value=True),
            patch("dlm.inference.backends.select.mlx_available", return_value=False),
        ):
            with pytest.raises(UnsupportedBackendError, match="mlx extra"):
                select_backend("mlx")


class TestSelectBackendAuto:
    def test_auto_picks_mlx_when_available(self) -> None:
        with patch("dlm.inference.backends.select.mlx_available", return_value=True):
            assert select_backend("auto") == "mlx"

    def test_auto_falls_back_to_pytorch_when_mlx_absent(self) -> None:
        with patch("dlm.inference.backends.select.mlx_available", return_value=False):
            assert select_backend("auto") == "pytorch"

    def test_auto_on_non_darwin_never_imports_mlx(self) -> None:
        # `mlx_available()` short-circuits on is_apple_silicon=False,
        # so auto on Linux gracefully lands on pytorch without probing
        # the mlx/mlx_lm modules.
        with patch("dlm.inference.backends.select.is_apple_silicon", return_value=False):
            assert select_backend("auto") == "pytorch"


class TestBuildBackend:
    def test_pytorch_returns_pytorch_backend(self) -> None:
        from dlm.inference.backends.pytorch_backend import PyTorchBackend

        backend = build_backend("pytorch", MagicMock())
        assert isinstance(backend, PyTorchBackend)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown backend"):
            build_backend("haskell", MagicMock())  # type: ignore[arg-type]


class TestMlxAvailableDoesNotImportMlx:
    def test_mlx_available_off_platform_short_circuits(self) -> None:
        # On non-darwin, mlx_available returns False without calling
        # importlib.util.find_spec — guaranteed by the early return.
        from dlm.inference.backends import select as sel

        with (
            patch.object(sel, "is_apple_silicon", return_value=False),
            patch.object(sel.importlib.util, "find_spec") as m_find,
        ):
            assert sel.mlx_available() is False
            m_find.assert_not_called()
