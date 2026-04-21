"""Loader module smoke tests."""

from __future__ import annotations


def test_loader_module_imports() -> None:
    import dlm.train.loader as loader

    assert loader.__name__ == "dlm.train.loader"
