"""Live `mlx-serve` export smoke using the Sprint 14.5 trained store."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from tests.integration.export._runtime_smoke import cleared_offline_env, require_loopback_bind

if TYPE_CHECKING:
    from tests.fixtures.trained_store import TrainedStoreHandle

pytestmark = pytest.mark.slow


@pytest.mark.slow
def test_export_target_mlx_serve_smokes_live(trained_store: TrainedStoreHandle) -> None:
    from dlm.inference.backends.select import is_apple_silicon, mlx_available

    require_loopback_bind()
    if not is_apple_silicon():
        pytest.skip("requires Apple Silicon (darwin-arm64)")
    if not mlx_available():
        pytest.skip("requires the mlx extra (`uv sync --extra mlx`)")

    from dlm.cli.app import app
    from dlm.export.manifest import load_export_manifest
    from dlm.store.manifest import load_manifest

    os.environ["DLM_HOME"] = str(trained_store.home)

    with cleared_offline_env():
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "export",
                str(trained_store.doc),
                "--target",
                "mlx-serve",
            ],
        )

    assert result.exit_code == 0, result.output

    export_dir = trained_store.store.exports / "mlx-serve"
    manifest = load_export_manifest(export_dir)
    store_manifest = load_manifest(trained_store.store.manifest)

    assert (export_dir / "mlx_serve_launch.sh").is_file()
    assert (export_dir / "adapter").is_dir()
    assert manifest.target == "mlx-serve"
    assert store_manifest.exports, "store export summary missing"
    assert store_manifest.exports[-1].target == "mlx-serve"
    assert store_manifest.exports[-1].smoke_output_first_line
