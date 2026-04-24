"""Direct coverage for package-level version fallback wiring."""

from __future__ import annotations

import runpy
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import patch

_INIT_PATH = Path(__file__).resolve().parents[2] / "src" / "dlm" / "__init__.py"


def test_package_init_reads_installed_version() -> None:
    with patch("importlib.metadata.version", return_value="1.2.3"):
        module_globals = runpy.run_path(str(_INIT_PATH))

    assert module_globals["__version__"] == "1.2.3"


def test_package_init_falls_back_when_package_metadata_is_missing() -> None:
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
        module_globals = runpy.run_path(str(_INIT_PATH))

    assert module_globals["__version__"] == "0.0.0+unknown"
