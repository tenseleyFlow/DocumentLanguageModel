"""End-to-end audio training cycle on Qwen2-Audio-7B (Sprint 35.2 T12).

Parallel to `tests/integration/vl/test_vl_training.py`. Exercises the
live audio-language path end-to-end:

1. `dlm init --audio` scaffolds a schema-v11 doc + the `.dlm` store.
2. A 0.5 s mono 16 kHz WAV clip is dropped at the scaffold's
   placeholder path with a matching `<stem>.txt` transcript.
3. `dlm.train.run()` walks:
     - `load_base_model` dispatches to `Qwen2AudioForConditionalGeneration`
     - `load_processor` loads the Qwen2-Audio processor
     - `BlobStore.put` for the audio file
     - `sections_to_rows` emits `{audio_blob_sha, audio_path, text}`
     - `AudioLmCollator` decodes the waveform + runs the processor
     - SFTTrainer runs 1 step and commits the adapter
4. Asserts the adapter directory exists, manifest records the run,
   and the blob store has the ingested clip sha'd under `blobs/`.

Markers: `slow` + `audio`. Skipped by default. Run explicitly via
`pytest -m "slow and audio" -k audio_training` on a host with:
- ~30 GB free disk (Qwen2-Audio fp16 weights)
- Qwen2-Audio cached locally (`huggingface-cli download Qwen/Qwen2-Audio-7B-Instruct`)
- Qwen2-Audio license accepted (`HF_TOKEN` exported)
- CUDA ≥ SM 8.0 with 24+ GB VRAM (fp16) or Apple Silicon with ≥ 32 GB
  unified memory (the 7B audio weights do not fit 16 GB fp16)

Skips cleanly when any prerequisite is missing so the marker alone is
enough to select it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

pytestmark = [pytest.mark.slow, pytest.mark.audio]


def _write_wav(path: Path, *, sample_rate: int = 16_000, seconds: float = 0.5) -> None:
    """Write a mono float32 sine wave so soundfile can decode it."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(round(seconds * sample_rate))
    t = np.linspace(0.0, seconds, num_samples, dtype=np.float32)
    data = np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    sf.write(str(path), data, sample_rate, subtype="FLOAT")


def _scaffold_audio_doc(tmp_home: Path, workdir: Path) -> Path:
    """Run `dlm init --audio` into `workdir/doc.dlm`."""
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
        AutoProcessor.from_pretrained(
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


def test_qwen2_audio_one_cycle_end_to_end(  # pragma: no cover — slow + audio
    tmp_path: Path,
    audio_prereqs: None,
) -> None:
    """Full audio cycle: init → ingest wav → train 1 step → verify adapter."""
    import dlm.train as dlm_train
    from dlm.doc.parser import parse_file
    from dlm.store.manifest import load_manifest
    from dlm.store.paths import for_dlm

    tmp_home = tmp_path / "home"
    workdir = tmp_path / "corpus"
    workdir.mkdir()

    doc = _scaffold_audio_doc(tmp_home, workdir)

    # The scaffold ships with `::audio path="clips/your-clip.wav"
    # transcript="Transcript of the audio clip."::`. Drop a real WAV
    # at that relative location so the trainer has something to ingest.
    # The inline transcript in the fence takes priority over any
    # sibling `<stem>.txt` — both are acceptable grammars.
    _write_wav(workdir / "clips" / "your-clip.wav")

    parsed = parse_file(doc)
    store = for_dlm(parsed.frontmatter.dlm_id, home=tmp_home)

    # Cap steps to 1 so the test completes on commodity hardware.
    result = dlm_train.run(
        doc,
        mode="fresh",
        seed=42,
        max_steps=1,
        home=tmp_home,
    )
    assert result is not None

    # Adapter committed under v0001/.
    adapter_dir = store.resolve_current_adapter()
    assert adapter_dir is not None, "no current-adapter pointer after audio train"
    assert (adapter_dir / "adapter_config.json").is_file()
    assert (adapter_dir / "adapter_model.safetensors").is_file()

    # Manifest records the audio run.
    manifest = load_manifest(store.manifest)
    assert len(manifest.training_runs) == 1
    assert manifest.training_runs[0].steps >= 1

    # Blob store ingested the clip — one blob under blobs/<prefix>/.
    blob_files = list(store.blob_dir.rglob("*"))
    blob_regular = [p for p in blob_files if p.is_file()]
    assert len(blob_regular) == 1, f"expected one blob, got {blob_regular}"

    # Finite-weight gate: sanity check the adapter weights didn't
    # collapse to NaN during the single step.
    import safetensors.torch as st

    weights: dict[str, Any] = st.load_file(str(adapter_dir / "adapter_model.safetensors"))
    assert weights, "adapter_model.safetensors is empty"
    import torch

    for name, tensor in weights.items():
        assert torch.isfinite(tensor).all(), f"{name} has non-finite weights"
