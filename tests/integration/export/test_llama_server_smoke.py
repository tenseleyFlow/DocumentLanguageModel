"""Live `llama-server` export smoke using the Sprint 14.5 trained store."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from tests.integration.export._runtime_smoke import (
    cleared_offline_env,
    require_loopback_bind,
    vendor_server_built,
)

if TYPE_CHECKING:
    from tests.fixtures.trained_store import TrainedStoreHandle

pytestmark = pytest.mark.slow


@pytest.mark.slow
def test_export_target_llama_server_smokes_live(trained_store: TrainedStoreHandle) -> None:
    require_loopback_bind()
    if not vendor_server_built():
        pytest.skip(
            "vendored llama-server binary not built; run `scripts/bump-llama-cpp.sh build --with-server`."
        )

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
                "llama-server",
                "--quant",
                "Q4_K_M",
                "--no-imatrix",
            ],
        )

    assert result.exit_code == 0, result.output

    export_dir = trained_store.store.export_quant_dir("Q4_K_M")
    manifest = load_export_manifest(export_dir)
    store_manifest = load_manifest(trained_store.store.manifest)

    assert (export_dir / "llama-server_launch.sh").is_file()
    assert (export_dir / "chat-template.jinja").is_file()
    assert manifest.target == "llama-server"
    assert store_manifest.exports, "store export summary missing"
    assert store_manifest.exports[-1].target == "llama-server"
    assert store_manifest.exports[-1].smoke_output_first_line
