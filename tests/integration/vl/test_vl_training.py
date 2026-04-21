"""End-to-end VL training cycle on PaliGemma-3B-mix-224 (Sprint 35 v1, T12).

Exercises the live VL path end-to-end:

1. `dlm init --multimodal` scaffolds a schema-v10 doc + the `.dlm` store.
2. A 2x2 RGB PNG is dropped into the scaffolded directory under the
   placeholder filename the scaffold emits.
3. `dlm.train.run()` walks:
     - `load_base_model` (dispatches to `AutoModelForImageTextToText`)
     - `load_processor`
     - `BlobStore.put` for the image
     - `sections_to_rows` emits `{images: [PIL], text: "<image>\\n..."}`
     - TRL 1.2 auto-selects `DataCollatorForVisionLanguageModeling`
     - SFTTrainer runs 1 step and commits the adapter
4. Asserts the adapter directory exists, manifest records the run,
   and the blob store has the ingested image sha'd under `blobs/`.

Markers: `slow` + `vl`. Skipped by default (`-m "not slow and not vl"`
in `pyproject.toml`). Run explicitly via `pytest -m "slow and vl" -k vl_training`
on a host with:
- ~10 GB free disk (PaliGemma fp16 weights)
- PaliGemma cached locally via `huggingface-cli download google/paligemma-3b-mix-224`
- Gemma license accepted (`HF_TOKEN` exported)
- CUDA ≥ SM 8.0 with 12+ GB VRAM, or Apple Silicon with ≥ 16 GB unified memory

Skips cleanly when any prerequisite is missing so the marker alone is
enough to select it — the test itself refuses to run on an under-
provisioned host rather than erroring out mid-training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from typer.testing import CliRunner

pytestmark = [pytest.mark.slow, pytest.mark.vl]


def _write_pixel(path: Path, color: tuple[int, int, int] = (64, 128, 255)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=color).save(path, format="PNG")


def _scaffold_multimodal_doc(tmp_home: Path, workdir: Path) -> Path:
    """Run `dlm init --multimodal` into `workdir/doc.dlm`."""
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
            "--multimodal",
            "--i-accept-license",
        ],
    )
    assert result.exit_code == 0, result.output
    return doc


def _host_has_vl_prerequisites() -> tuple[bool, str]:
    """Return (ok, skip_reason) — only run when the host can host PaliGemma."""
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
        from transformers import AutoProcessor
    except ImportError:
        return False, "transformers unavailable"
    try:
        AutoProcessor.from_pretrained(
            "google/paligemma-3b-mix-224",
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001 — any local-cache miss skips cleanly
        return False, f"paligemma not cached locally: {type(exc).__name__}"
    return True, ""


@pytest.fixture
def vl_prereqs() -> None:
    ok, reason = _host_has_vl_prerequisites()
    if not ok:
        pytest.skip(reason)


def test_paligemma_one_cycle_end_to_end(  # pragma: no cover — slow + vl
    tmp_path: Path,
    vl_prereqs: None,
) -> None:
    """Full VL cycle: init → ingest image → train 1 step → verify adapter."""
    import dlm.train as dlm_train
    from dlm.doc.parser import parse_file
    from dlm.store.manifest import load_manifest
    from dlm.store.paths import for_dlm

    tmp_home = tmp_path / "home"
    workdir = tmp_path / "corpus"
    workdir.mkdir()

    doc = _scaffold_multimodal_doc(tmp_home, workdir)

    # The scaffold ships with `::image path="figures/your-image.png"::`.
    # Drop a real 4×4 PNG at that relative location so `dlm train`
    # actually finds a file to ingest.
    _write_pixel(workdir / "figures" / "your-image.png")

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
    assert adapter_dir is not None, "no current-adapter pointer after VL train"
    assert (adapter_dir / "adapter_config.json").is_file()
    assert (adapter_dir / "adapter_model.safetensors").is_file()

    # Manifest records the VL run.
    manifest = load_manifest(store.manifest)
    assert len(manifest.training_runs) == 1
    assert manifest.training_runs[0].steps >= 1

    # Blob store ingested the image — one blob under blobs/<prefix>/.
    blob_files = list(store.blob_dir.rglob("*"))
    blob_regular = [p for p in blob_files if p.is_file()]
    assert len(blob_regular) == 1, f"expected one blob, got {blob_regular}"

    # Finite-weight gate: sanity check the adapter weights didn't
    # collapse to NaN during the single step (pitfall #5 family bug).
    import safetensors.torch as st

    weights: dict[str, Any] = st.load_file(str(adapter_dir / "adapter_model.safetensors"))
    assert weights, "adapter_model.safetensors is empty"
    import torch

    for name, tensor in weights.items():
        assert torch.isfinite(tensor).all(), f"{name} has non-finite weights"
