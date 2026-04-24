"""Live `vllm` export smoke using the Sprint 14.5 trained store."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from tests.integration.export._runtime_smoke import (
    cleared_offline_env,
    require_loopback_bind,
    vllm_smoke_skip_reason,
)

if TYPE_CHECKING:
    from tests.fixtures.trained_store import TrainedStoreHandle

_VLLM_SKIP_REASON = vllm_smoke_skip_reason()

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(_VLLM_SKIP_REASON is not None, reason=_VLLM_SKIP_REASON or ""),
]


@pytest.mark.slow
def test_export_target_vllm_smokes_live(trained_store: TrainedStoreHandle) -> None:
    require_loopback_bind()

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
                "vllm",
            ],
        )

    assert result.exit_code == 0, result.output

    export_dir = trained_store.store.exports / "vllm"
    manifest = load_export_manifest(export_dir)
    store_manifest = load_manifest(trained_store.store.manifest)

    assert (export_dir / "vllm_launch.sh").is_file()
    assert (export_dir / "vllm_config.json").is_file()
    assert (export_dir / "adapters" / "adapter").is_dir()
    assert manifest.target == "vllm"
    assert store_manifest.exports, "store export summary missing"
    assert store_manifest.exports[-1].target == "vllm"
    assert store_manifest.exports[-1].smoke_output_first_line
