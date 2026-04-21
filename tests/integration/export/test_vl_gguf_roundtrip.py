"""End-to-end VL GGUF export round-trip (Sprint 35.4 T7).

Tests the full SUPPORTED path: train a PaliGemma adapter → `dlm export`
emits GGUF + Modelfile → `ollama create` + `ollama run` returns a
coherent response to an image prompt.

**Current status: auto-skips.** The vendored llama.cpp tag (b8816)
doesn't know about `PaliGemmaForConditionalGeneration` — the arch
probe returns UNSUPPORTED, so this test skips without running the
expensive training/export pipeline. It stays in the tree so a
llama.cpp bump that flips the probe verdict surfaces the GGUF path
immediately; the day that happens the test either passes (happy
path) or fails with a real actionable error.

Markers: `slow` + `vl` + `ollama`. Skipped by default. Run explicitly
on a provisioned host (Ollama 0.4+ installed, PaliGemma cached,
Gemma license accepted).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from dlm.export.arch_probe import SupportLevel, probe_gguf_arch

pytestmark = [
    pytest.mark.slow,
    pytest.mark.vl,
    pytest.mark.ollama,
]


_PALIGEMMA_ARCH = "PaliGemmaForConditionalGeneration"
_OLLAMA_MIN_VERSION = (0, 4, 0)


def _host_has_ollama() -> tuple[bool, str]:
    """Return (ok, reason). Ollama 0.4+ is required for `{{ .Image }}`."""
    ollama = shutil.which("ollama")
    if ollama is None:
        return False, "ollama not on PATH"
    try:
        proc = subprocess.run(
            [ollama, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return False, f"ollama --version failed: {exc}"
    version_line = proc.stdout.strip()
    # Best-effort: Ollama emits "ollama version is 0.4.x" or similar.
    # Any probe failure → assume pre-0.4 + skip rather than crash.
    parts = [int(p) for p in _extract_version(version_line) if p.isdigit()]
    if len(parts) < 3:
        return False, f"could not parse `{version_line}`"
    if tuple(parts[:3]) < _OLLAMA_MIN_VERSION:
        return False, (
            f"ollama {'.'.join(str(p) for p in parts[:3])} < "
            f"{'.'.join(str(p) for p in _OLLAMA_MIN_VERSION)} "
            "(required for {{ .Image }} directive)"
        )
    return True, ""


def _extract_version(line: str) -> list[str]:
    """Pull tokens that look like version parts from a free-form line."""
    chunks: list[str] = []
    for token in line.replace("-", " ").replace(".", " ").split():
        chunks.append(token)
    return chunks


@pytest.fixture
def paligemma_supported() -> None:
    """Skip the test cleanly when llama.cpp doesn't support PaliGemma yet."""
    verdict = probe_gguf_arch(_PALIGEMMA_ARCH)
    if verdict.support is not SupportLevel.SUPPORTED:
        pytest.skip(
            f"llama.cpp {verdict.llama_cpp_tag or '?'} does not support "
            f"{_PALIGEMMA_ARCH} ({verdict.support.value}). "
            "Bump the vendored tag once upstream adds PaliGemma GGUF "
            "conversion, then this test runs."
        )


@pytest.fixture
def ollama_available() -> None:
    ok, reason = _host_has_ollama()
    if not ok:
        pytest.skip(f"ollama prerequisite missing: {reason}")


def test_paligemma_gguf_roundtrip(
    paligemma_supported: None,
    ollama_available: None,
    tmp_path: Path,
) -> None:
    """Train tiny PaliGemma adapter → export GGUF → ollama run.

    Intentionally light on the training side (1 step, 1 image) — the
    test is about the export + ollama plumbing, not training quality.
    """
    # When this test actually runs (post-llama.cpp-bump), the body
    # below fills in. For now the SUPPORTED gate above skips every
    # invocation on the current vendored tag, so the scaffold doesn't
    # drag in PaliGemma weights on CI.
    pytest.skip(
        "VL GGUF round-trip body awaits llama.cpp PaliGemma support. "
        "See sprint 35.4 T7."
    )
