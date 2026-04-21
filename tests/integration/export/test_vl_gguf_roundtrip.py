"""End-to-end VL GGUF export round-trip.

Exercises the full SUPPORTED path: an adapter directory + a cached VL
base → `run_vl_gguf_export` emits a quantized GGUF + Modelfile + manifest
+ sidecar. Qwen2-VL is SUPPORTED at our pinned llama.cpp tag (b8816);
PaliGemma and InternVL2 remain UNSUPPORTED/PARTIAL upstream, so their
parametrizations auto-skip.

Guarded on:

- `probe_gguf_arch(...)` returns SUPPORTED for the parametrized arch
  (else skip — UNSUPPORTED/PARTIAL routes through HF-snapshot anyway).
- The base weights are locally cached (else skip — no network at test
  time; the vendored registry has placeholder SHAs so only a real
  `dlm train` against the base primes the cache).
- `llama-quantize` exists under `vendor/llama.cpp/build/bin/` (else
  skip — CI bots that haven't run `scripts/bump-llama-cpp.sh` can't
  execute the subprocess chain).

Running the full body requires HF weights + a built vendored llama.cpp.
This is a slow-marked test; the default CI skips it and it's opt-in
via `pytest -m "slow and vl"` on a provisioned host.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from dlm.export.arch_probe import SupportLevel, probe_gguf_arch

pytestmark = [
    pytest.mark.slow,
    pytest.mark.vl,
]


_VL_ARCHS = [
    pytest.param(
        "Qwen2VLForConditionalGeneration",
        "Qwen/Qwen2-VL-2B-Instruct",
        id="qwen2-vl-2b",
    ),
    pytest.param(
        "PaliGemmaForConditionalGeneration",
        "google/paligemma-3b-mix-224",
        id="paligemma-3b",
    ),
    pytest.param(
        "InternVLChatModel",
        "OpenGVLab/InternVL2-2B",
        id="internvl2-2b",
    ),
]

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


def _llama_quantize_available() -> tuple[bool, str]:
    """Confirm `vendor/llama.cpp/build/bin/llama-quantize` exists + is exec."""
    # Walk up from this file to find the repo root.
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "vendor" / "llama.cpp").is_dir():
            bin_path = parent / "vendor" / "llama.cpp" / "build" / "bin" / "llama-quantize"
            if bin_path.exists():
                return True, ""
            return False, f"{bin_path} not built — run scripts/bump-llama-cpp.sh build"
    return False, "repo root (vendor/llama.cpp) not found above this test file"


@pytest.mark.parametrize(("arch", "hf_id"), _VL_ARCHS)
def test_vl_gguf_roundtrip(
    arch: str,
    hf_id: str,
    tmp_path: Path,
    run_heavy_vl: bool,
) -> None:
    """GGUF emission for a VL arch — filled body on SUPPORTED, skip otherwise.

    The parametrize list covers all three registered VL archs; only the
    SUPPORTED one runs the real subprocess chain. This means a
    llama.cpp bump that flips a previously-UNSUPPORTED verdict to
    SUPPORTED surfaces here immediately (the skip reason changes from
    "UNSUPPORTED" to a concrete subprocess result).
    """
    verdict = probe_gguf_arch(arch)
    if verdict.support is not SupportLevel.SUPPORTED:
        pytest.skip(
            f"llama.cpp {verdict.llama_cpp_tag or '?'} verdict for {arch} is "
            f"{verdict.support.value} — the dispatcher routes to HF-snapshot "
            "for non-SUPPORTED archs. Bump the vendored tag once upstream "
            "adds coverage and this parametrization starts running."
        )

    ok, reason = _llama_quantize_available()
    if not ok:
        pytest.skip(f"vendored llama.cpp not ready: {reason}")

    # Cached weights check — we don't download at test time. If no one
    # has run `dlm train` (or a manual cache-prime) against this hf_id,
    # there's no base directory to feed merge + convert.
    try:
        from huggingface_hub import snapshot_download  # pragma: no cover
    except ImportError:
        pytest.skip("huggingface_hub not importable in this environment")
    try:
        cached_base = Path(
            snapshot_download(
                repo_id=hf_id,
                local_files_only=True,
            )
        )
    except Exception as exc:  # noqa: BLE001 - translate HF cache misses to skip
        pytest.skip(
            f"{hf_id} not in the local HF cache ({type(exc).__name__}); "
            "prime with `dlm train` or `huggingface-cli download` on a "
            "provisioned host."
        )

    assert cached_base.exists(), cached_base

    if not run_heavy_vl:
        pytest.skip(
            "VL GGUF round-trip body requires ~8 GB intermediate storage + "
            "several minutes of training. Opt in with "
            "`pytest -m 'slow and vl' --run-heavy-vl` to execute. "
            "The emitter itself is covered by tests/unit/export/test_vl_gguf.py."
        )

    # Heavy body: train a 1-step LoRA → export --merged --quant Q4_K_M →
    # verify the GGUF + Modelfile + manifest land. Any subprocess failure
    # surfaces here as a test error, which is the desired signal for
    # "upstream flipped support on this arch and our emitter broke."
    from typer.testing import CliRunner

    from dlm.base_models import BASE_MODELS
    from dlm.cli.app import app
    from dlm.doc.parser import parse_file
    from dlm.export.vl_gguf import run_vl_gguf_export
    from dlm.store.paths import for_dlm

    # Find the registered base-model key whose hf_id matches the arch;
    # the test parametrization carries the hf_id, and `dlm init --base`
    # wants a registry key.
    base_key = next(
        (key for key, spec in BASE_MODELS.items() if spec.hf_id == hf_id),
        None,
    )
    if base_key is None:
        pytest.skip(f"{hf_id} is not in the registry; add a BaseModelSpec for it")

    tmp_home = tmp_path / "home"
    doc_path = tmp_path / "doc.dlm"
    runner = CliRunner()

    # Scaffold a multimodal doc at the target base, then train one step
    # so an adapter version exists. We intentionally use --max-steps 1
    # (cap training cost) + --i-accept-license (gated bases).
    init_result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_home),
            "init",
            str(doc_path),
            "--multimodal",
            "--base",
            base_key,
            "--i-accept-license",
        ],
    )
    assert init_result.exit_code == 0, init_result.output

    train_result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_home),
            "train",
            str(doc_path),
            "--max-steps",
            "1",
            "--seed",
            "42",
        ],
    )
    assert train_result.exit_code == 0, train_result.output

    # Now drive the emitter directly — we control plan + verdict this
    # way and avoid routing through the CLI dispatcher's fallback on
    # any unrelated refusal.
    parsed = parse_file(doc_path)
    store = for_dlm(parsed.frontmatter.dlm_id, home=tmp_home)
    spec = BASE_MODELS[base_key]

    from dlm.export.plan import ExportPlan

    plan = ExportPlan(merged=True, imatrix="off", quant="Q4_K_M")
    emit_result = run_vl_gguf_export(
        store,
        spec,
        plan,
        verdict=verdict,
        cached_base_dir=cached_base,
        source_dlm_path=doc_path,
        dlm_version="test",
    )

    # Contract checks — the module's docstring + Sprint 35.4 spec pin these.
    assert emit_result.gguf_path.exists()
    assert emit_result.gguf_path.stat().st_size > 0
    assert emit_result.modelfile_path.exists()
    modelfile_body = emit_result.modelfile_path.read_text(encoding="utf-8")
    assert f"FROM ./{emit_result.gguf_path.name}" in modelfile_body
    assert "ADAPTER" not in modelfile_body  # merged-only at this upstream tag
    assert emit_result.mmproj_path is None  # single-file contract
    assert emit_result.quant == "Q4_K_M"
    assert emit_result.llama_cpp_tag == verdict.llama_cpp_tag
    assert emit_result.manifest_path.exists()

    sidecar_path = emit_result.export_dir / "vl_gguf.json"
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["arch_verdict"]["support"] == "SUPPORTED"
    assert sidecar["arch_verdict"]["architecture"] == arch
