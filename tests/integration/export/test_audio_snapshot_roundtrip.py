"""End-to-end audio HF-snapshot export round-trip.

Audio-language bases (Qwen2-Audio-7B at our pinned set) don't have a
llama.cpp conversion path — the dispatcher routes them unconditionally
through `run_audio_snapshot_export`, which writes a tar-able HF
snapshot (adapter + processor + manifest + README) under
`exports/hf-audio-snapshot/`. This test exercises that path end-to-end
on a real trained adapter:

1. Scaffold a doc via `dlm init --audio` + drop a 0.5 s WAV clip.
2. Train one step with `dlm.train.run` so a real adapter lands in the
   store and the processor caches locally.
3. Invoke `dlm export` via the CLI — the dispatcher branches on
   `spec.modality == "audio-language"` and runs the snapshot emitter.
4. Assert the layout (manifest + README + adapter dir + processor
   dir), parse the manifest, and run `verify_artifacts` — the same
   content-hash check a recipient would run after downloading.

Skips cleanly when prerequisites are missing (CUDA/MPS, Qwen2-Audio
weights cached, soundfile importable, license accepted). Slow-marked;
opt-in via `pytest -m "slow and audio"`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

pytestmark = [pytest.mark.slow, pytest.mark.audio]


def _write_wav(path: Path, *, sample_rate: int = 16_000, seconds: float = 0.5) -> None:
    """Write a mono float32 sine wave so soundfile can decode it."""
    import soundfile as sf  # type: ignore[import-untyped]

    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(round(seconds * sample_rate))
    t = np.linspace(0.0, seconds, num_samples, dtype=np.float32)
    data = np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    sf.write(str(path), data, sample_rate, subtype="FLOAT")


def _scaffold_audio_doc(tmp_home: Path, workdir: Path) -> Path:
    from dlm.cli.app import app

    doc = workdir / "doc.dlm"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_home),
            "init",
            str(doc),
            "--audio",
            "--i-accept-license",
        ],
    )
    assert result.exit_code == 0, result.output
    return doc


def _host_has_audio_prerequisites() -> tuple[bool, str]:
    """Return (ok, skip_reason) — only run when the host can host Qwen2-Audio."""
    try:
        import torch
    except ImportError:
        return False, "torch unavailable"
    has_accelerator = torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if not has_accelerator:
        return False, "no CUDA or MPS accelerator"
    try:
        import soundfile  # noqa: F401
    except ImportError:
        return False, "soundfile unavailable (run `uv sync --extra audio`)"
    try:
        from transformers import AutoProcessor
    except ImportError:
        return False, "transformers unavailable"
    try:
        AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            "Qwen/Qwen2-Audio-7B-Instruct",
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001 — any local-cache miss skips cleanly
        return False, f"qwen2-audio not cached locally: {type(exc).__name__}"
    return True, ""


@pytest.fixture
def audio_prereqs() -> None:
    ok, reason = _host_has_audio_prerequisites()
    if not ok:
        pytest.skip(reason)


def test_audio_snapshot_export_roundtrip(  # pragma: no cover — slow + audio
    tmp_path: Path,
    audio_prereqs: None,
) -> None:
    """Train 1 step on Qwen2-Audio → `dlm export` → verify snapshot integrity."""
    from dlm.cli.app import app
    from dlm.doc.parser import parse_file
    from dlm.export.audio_snapshot import (
        AUDIO_SNAPSHOT_SUBDIR,
        SNAPSHOT_MANIFEST_FILENAME,
        SNAPSHOT_README_FILENAME,
        load_audio_snapshot_manifest,
        verify_artifacts,
    )
    from dlm.store.paths import for_dlm

    tmp_home = tmp_path / "home"
    workdir = tmp_path / "corpus"
    workdir.mkdir()

    doc = _scaffold_audio_doc(tmp_home, workdir)
    # Scaffold ships with `::audio path="clips/your-clip.wav" ...::`; drop
    # a real WAV there so ingestion finds something to tokenize.
    _write_wav(workdir / "clips" / "your-clip.wav")

    parsed = parse_file(doc)
    store = for_dlm(parsed.frontmatter.dlm_id, home=tmp_home)

    # One training step via the CLI (the public surface) so the
    # adapter + processor both land in the store the same way a user
    # would produce them. --max-cycles 1 caps to a single cycle.
    runner = CliRunner()
    train_result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_home),
            "train",
            str(doc),
            "--max-steps",
            "1",
            "--seed",
            "42",
        ],
    )
    assert train_result.exit_code == 0, train_result.output
    assert store.resolve_current_adapter() is not None

    # Drive `dlm export` through the CLI so the dispatcher branch runs
    # (spec.modality=="audio-language" → _dispatch_audio_snapshot_export
    # → run_audio_snapshot_export). GGUF-only flags like --quant are
    # accepted but produce a banner; we omit them to keep the output
    # clean. --skip-ollama isn't applicable to the snapshot path (the
    # snapshot dispatcher doesn't invoke ollama), but passing it is a
    # no-op.
    result = runner.invoke(
        app,
        [
            "--home",
            str(tmp_home),
            "export",
            str(doc),
            "--no-smoke",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "HF snapshot written" in (result.output or "") + (result.stderr or "")

    # Snapshot lives at the registered subdir.
    export_dir = store.exports / AUDIO_SNAPSHOT_SUBDIR
    assert export_dir.is_dir()
    assert (export_dir / SNAPSHOT_MANIFEST_FILENAME).is_file()
    assert (export_dir / SNAPSHOT_README_FILENAME).is_file()
    assert (export_dir / "adapter" / "adapter_config.json").is_file()
    assert (export_dir / "adapter" / "adapter_model.safetensors").is_file()

    # Processor dir contains at least the feature extractor + tokenizer
    # configs (`save_pretrained` writes several JSON files, exact set
    # is transformers-version-dependent). Assert the dir is populated.
    processor_dir = export_dir / "processor"
    assert processor_dir.is_dir(), "processor/ missing — audio snapshot unloadable"
    processor_files = list(processor_dir.iterdir())
    assert processor_files, "processor/ is empty — audio snapshot unloadable"

    # Manifest round-trips + verify_artifacts passes: content-hash
    # integrity of every file the manifest lists.
    manifest = load_audio_snapshot_manifest(export_dir)
    assert manifest.export_target == "hf_snapshot"
    assert manifest.modality == "audio-language"
    assert manifest.base_model_hf_id == "Qwen/Qwen2-Audio-7B-Instruct"
    assert manifest.base_model_architecture == "Qwen2AudioForConditionalGeneration"
    assert manifest.adapter_version == 1
    # The preprocessor params come from the registry spec; assert they
    # match Qwen2-Audio's pinned values (16 kHz, 30s cap).
    assert manifest.sample_rate == 16_000
    assert manifest.max_length_seconds == 30.0
    assert manifest.audio_token == "<|AUDIO|>"

    verify_artifacts(export_dir, manifest)  # raises on any drift

    # Finally, a second `dlm export` invocation overwrites in place
    # (the snapshot dispatcher re-resolves the current adapter each
    # time; no stale-file cross-contamination). This protects a
    # train-retrain-re-export cadence.
    result2 = runner.invoke(
        app,
        [
            "--home",
            str(tmp_home),
            "export",
            str(doc),
            "--no-smoke",
        ],
    )
    assert result2.exit_code == 0, result2.output
    manifest_redux = load_audio_snapshot_manifest(export_dir)
    verify_artifacts(export_dir, manifest_redux)
