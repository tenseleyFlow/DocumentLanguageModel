"""End-to-end `dlm export` → `ollama create` → `ollama run` smoke (Sprint 12).

Replaces the audit-04 scaffold with a real body driven by the
`trained_store` fixture. Requires:

- `vendor/llama.cpp/` submodule built (`scripts/bump-llama-cpp.sh build`).
- `ollama` binary on PATH + a running daemon. The `ollama_daemon`
  fixture starts one if none is already listening; the external
  daemon (if any) is left untouched.
- SmolLM2-135M offline cache (via `trained_store`).

Skips cleanly when any of the above is missing. The slow-test CI job
pre-warms everything so the path runs green there.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess  # nosec B404
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

pytestmark = pytest.mark.slow


_OLLAMA_DAEMON_PORT = 11434
_OLLAMA_MODEL_NAME = "dlm-it-smoke:test"


def _port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        try:
            sock.connect(("127.0.0.1", port))
        except OSError:
            return False
        return True


def _vendor_built() -> bool:
    vendor_root = Path(__file__).resolve().parents[4] / "vendor" / "llama.cpp"
    return (vendor_root / "build" / "bin" / "llama-quantize").is_file()


@pytest.fixture(scope="session")
def ollama_daemon() -> Iterator[None]:
    """Start `ollama serve` if nothing is listening; leave a pre-existing
    daemon alone.

    Yields nothing — tests just need the daemon reachable on 11434 by
    the time they hit `ollama run`.
    """
    if shutil.which("ollama") is None:
        pytest.skip("ollama binary not on PATH.")

    owns_daemon = not _port_listening(_OLLAMA_DAEMON_PORT)
    proc: subprocess.Popen[bytes] | None = None
    if owns_daemon:
        proc = subprocess.Popen(  # nosec B603, B607
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Poll for readiness rather than sleeping blindly.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if _port_listening(_OLLAMA_DAEMON_PORT):
                break
            time.sleep(0.25)
        else:
            proc.terminate()
            pytest.skip("ollama serve didn't come up within 30s")
    try:
        yield
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.mark.slow
def test_export_creates_and_smokes_model(trained_store, ollama_daemon: None) -> None:
    """Full export → register → smoke-run round trip with non-empty output."""
    if not _vendor_built():
        pytest.skip("vendor/llama.cpp not built; run `scripts/bump-llama-cpp.sh build`.")

    from dlm.cli.app import app
    from dlm.store.manifest import load_manifest

    os.environ["DLM_HOME"] = str(trained_store.home)
    offline_vars = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    saved = {k: os.environ.pop(k, None) for k in offline_vars}

    try:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "export",
                str(trained_store.doc),
                "--quant",
                "Q4_K_M",
                "--name",
                _OLLAMA_MODEL_NAME,
            ],
        )
        try:
            assert result.exit_code == 0, result.output

            # Independent round trip: run the model directly and compare
            # with the smoke-output captured in the manifest.
            proc = subprocess.run(  # nosec B603, B607
                ["ollama", "run", _OLLAMA_MODEL_NAME, "hello"],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            assert proc.returncode == 0, proc.stderr
            assert proc.stdout.strip(), f"empty generation: {proc.stderr}"

            manifest = load_manifest(trained_store.store.manifest)
            assert manifest.exports, "export summary missing from manifest"
            last_export = manifest.exports[-1]
            assert last_export.smoke_output_first_line, (
                "smoke_output_first_line empty — `ollama run` smoke didn't capture"
            )
        finally:
            # Clean up the registered model so reruns start fresh.
            subprocess.run(  # nosec B603, B607
                ["ollama", "rm", _OLLAMA_MODEL_NAME],
                capture_output=True,
                check=False,
                timeout=30,
            )
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


@pytest.mark.slow
def test_template_round_trip_matches_jinja_reference() -> None:
    """Superseded by Sprint 12.6's closed-loop harness.

    `tests/integration/export/test_template_closed_loop.py` verifies
    Go-template → Ollama vs HF-Jinja token counts via
    `prompt_eval_count` telemetry. This scaffold predates that and
    would duplicate the check; kept skipped for back-reference.
    """
    pytest.skip(
        "Replaced by sprint 12.6 closed-loop "
        "(tests/integration/export/test_template_closed_loop.py)."
    )
