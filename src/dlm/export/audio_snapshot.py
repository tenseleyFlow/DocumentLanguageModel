"""HF-snapshot export for audio-language bases (Sprint 35.2).

Parallel to `vl_snapshot.py`. GGUF conversion for audio architectures
isn't on llama.cpp's roadmap, so Sprint 35.2 emits an HF snapshot: a
self-contained directory a recipient can load with
`AutoProcessor.from_pretrained` + architecture-class `from_pretrained`
+ `PeftModel.from_pretrained`.

Layout under `exports/hf-audio-snapshot/`:

    adapter/                 # PEFT adapter dir (copy of the current store adapter)
    processor/               # processor config + tokenizer + feature-extractor files
    snapshot_manifest.json   # see AudioSnapshotManifest
    README.md                # load instructions for the recipient

The base weights are NOT copied — pinned by `hf_id` + `revision` in
the manifest. Qwen2-Audio-7B is ~15 GB fp16 and marked
`redistributable=False` for our pack path; the same policy applies to
the snapshot so users don't accidentally ship base weights through
the pack system downstream.

The architectural symmetry to VL is intentional: both paths write a
`snapshot_manifest.json` with overlapping fields (`created_at`,
`adapter_version`, `artifacts`). A future refactor can extract a
shared base once a third media modality lands.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from dlm.export.errors import ExportError, ExportManifestError
from dlm.export.manifest import ExportArtifact, build_artifact, compute_sha256
from dlm.io.atomic import write_text

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.store.paths import StorePath

AUDIO_SNAPSHOT_SUBDIR = "hf-audio-snapshot"
SNAPSHOT_MANIFEST_FILENAME = "snapshot_manifest.json"
SNAPSHOT_README_FILENAME = "README.md"


class AudioSnapshotManifest(BaseModel):
    """Self-describing record of one audio HF-snapshot export.

    Parallel to `VlSnapshotManifest` but scoped to the audio path.
    `export_target="hf_snapshot"` matches the VL manifest so downstream
    tooling can discriminate on `modality` alone when iterating over
    all exports in a store.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    export_target: Literal["hf_snapshot"] = "hf_snapshot"
    created_at: datetime
    created_by: str = Field(..., description="dlm version that wrote this manifest.")
    base_model_hf_id: str
    base_model_revision: str
    base_model_architecture: str
    modality: Literal["audio-language"] = "audio-language"
    audio_token: str
    num_audio_tokens: int
    sample_rate: int
    max_length_seconds: float
    adapter_version: int = Field(..., ge=1)
    adapter_name: str | None = None
    rationale: str = Field(
        default=(
            "Audio-language architectures are not on the llama.cpp "
            "roadmap; Sprint 35.2 emits an HF-snapshot fallback so "
            "users can share trained adapters without waiting for "
            "upstream GGUF support."
        ),
    )
    artifacts: list[ExportArtifact] = Field(default_factory=list)


@dataclass(frozen=True)
class AudioSnapshotResult:
    """Return value of `run_audio_snapshot_export` — what the CLI prints."""

    export_dir: Path
    manifest_path: Path
    readme_path: Path
    adapter_dir: Path
    processor_dir: Path
    artifacts: list[Path]


def run_audio_snapshot_export(
    store: StorePath,
    spec: BaseModelSpec,
    *,
    adapter_name: str | None = None,
    adapter_path_override: Path | None = None,
    dlm_version: str = "dlm-0",
    processor: object | None = None,
) -> AudioSnapshotResult:
    """Emit an audio HF-snapshot export under `exports/hf-audio-snapshot/`.

    Resolves the adapter dir, copies it into the export directory,
    saves the processor (if supplied) under `processor/`, writes the
    manifest + README, and returns the layout paths.

    `processor=None` lets callers skip the processor save (tests, dry
    runs). Production paths pass an `AutoProcessor` loaded via
    `dlm.train.loader.load_processor`.
    """
    if spec.modality != "audio-language":
        raise ExportError(
            f"run_audio_snapshot_export: {spec.key!r} is modality={spec.modality!r}; "
            "only audio-language bases go through the audio HF-snapshot path"
        )
    if spec.audio_preprocessor_plan is None:
        raise ExportError(
            f"run_audio_snapshot_export: {spec.key!r} has modality='audio-language' "
            "but no audio_preprocessor_plan (this is a schema bug — file an issue)"
        )

    adapter_path, adapter_version = _resolve_adapter_for_export(
        store=store,
        adapter_name=adapter_name,
        adapter_path_override=adapter_path_override,
    )

    export_dir = store.exports / AUDIO_SNAPSHOT_SUBDIR
    export_dir.mkdir(parents=True, exist_ok=True)

    adapter_out = export_dir / "adapter"
    _copy_adapter_dir(adapter_path, adapter_out)

    processor_out = export_dir / "processor"
    if processor is not None:
        processor_out.mkdir(parents=True, exist_ok=True)
        save = getattr(processor, "save_pretrained", None)
        if callable(save):
            save(str(processor_out))

    artifacts: list[Path] = []
    for path in sorted(export_dir.rglob("*")):
        if path.is_file() and path.name not in (
            SNAPSHOT_MANIFEST_FILENAME,
            SNAPSHOT_README_FILENAME,
        ):
            artifacts.append(path)

    plan = spec.audio_preprocessor_plan
    manifest = AudioSnapshotManifest(
        created_at=_utc_now(),
        created_by=dlm_version,
        base_model_hf_id=spec.hf_id,
        base_model_revision=spec.revision,
        base_model_architecture=spec.architecture,
        audio_token=plan.audio_token,
        num_audio_tokens=plan.num_audio_tokens,
        sample_rate=plan.sample_rate,
        max_length_seconds=plan.max_length_seconds,
        adapter_version=adapter_version,
        adapter_name=adapter_name,
        artifacts=[build_artifact(export_dir, p) for p in artifacts],
    )
    manifest_path = _save_manifest(export_dir, manifest)
    readme_path = _write_readme(export_dir, spec=spec, manifest=manifest)

    return AudioSnapshotResult(
        export_dir=export_dir,
        manifest_path=manifest_path,
        readme_path=readme_path,
        adapter_dir=adapter_out,
        processor_dir=processor_out,
        artifacts=artifacts,
    )


# --- internals ---------------------------------------------------------------


def _resolve_adapter_for_export(
    *,
    store: StorePath,
    adapter_name: str | None,
    adapter_path_override: Path | None,
) -> tuple[Path, int]:
    """Return (adapter_dir, version) for the export."""
    if adapter_path_override is not None:
        if not adapter_path_override.exists():
            raise ExportError(f"adapter_path_override {adapter_path_override} does not exist")
        return adapter_path_override, _version_from_dir_name(adapter_path_override)

    if adapter_name is None:
        resolved = store.resolve_current_adapter()
        pointer = store.adapter_current_pointer
    else:
        resolved = store.resolve_current_adapter_for(adapter_name)
        pointer = store.adapter_current_pointer_for(adapter_name)

    if resolved is None or not resolved.exists():
        raise ExportError(f"no current adapter under {pointer}; run `dlm train` before exporting.")
    return resolved, _version_from_dir_name(resolved)


def _version_from_dir_name(path: Path) -> int:
    stem = path.name
    if not stem.startswith("v") or not stem[1:].isdigit():
        return 1
    return int(stem[1:])


def _copy_adapter_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _save_manifest(export_dir: Path, manifest: AudioSnapshotManifest) -> Path:
    path = export_dir / SNAPSHOT_MANIFEST_FILENAME
    payload = manifest.model_dump(mode="json")
    blob = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    write_text(path, blob)
    return path


def _write_readme(
    export_dir: Path,
    *,
    spec: BaseModelSpec,
    manifest: AudioSnapshotManifest,
) -> Path:
    """Write a human-readable load-instruction file for the audio snapshot."""
    path = export_dir / SNAPSHOT_README_FILENAME
    body = (
        f"# HF-audio-snapshot export\n"
        f"\n"
        f"Target: **{spec.key}** ({spec.hf_id} @ {spec.revision[:12]}…)\n"
        f"Adapter version: v{manifest.adapter_version:04d}"
        f"{f' ({manifest.adapter_name})' if manifest.adapter_name else ''}\n"
        f"\n"
        f"## Load this snapshot\n"
        f"\n"
        f"```python\n"
        f"from transformers import AutoProcessor, {spec.architecture}\n"
        f"from peft import PeftModel\n"
        f"\n"
        f"base = {spec.architecture}.from_pretrained(\n"
        f'    "{spec.hf_id}", revision="{spec.revision}",\n'
        f")\n"
        f'model = PeftModel.from_pretrained(base, "./adapter")\n'
        f'processor = AutoProcessor.from_pretrained("./processor")\n'
        f"```\n"
        f"\n"
        f"## Audio input shape\n"
        f"\n"
        f"- Sample rate: {manifest.sample_rate} Hz (required; mismatches refused)\n"
        f"- Max length: {manifest.max_length_seconds} s (longer clips truncated)\n"
        f"- Placeholder token: `{manifest.audio_token}` "
        f"(expands to {manifest.num_audio_tokens} tokens per clip)\n"
        f"\n"
        f"## Why HF snapshot (not GGUF)\n"
        f"\n"
        f"Audio-language architectures are not currently supported by\n"
        f"`llama.cpp`. The HF-snapshot path gives you a portable adapter\n"
        f"directory that loads on any PyTorch + transformers install.\n"
    )
    write_text(path, body)
    return path


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)


def load_audio_snapshot_manifest(export_dir: Path) -> AudioSnapshotManifest:
    """Read + validate `<export_dir>/snapshot_manifest.json`."""
    path = export_dir / SNAPSHOT_MANIFEST_FILENAME
    if not path.exists():
        raise ExportManifestError(f"missing {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportManifestError(f"cannot parse {path}: {exc}") from exc
    try:
        return AudioSnapshotManifest.model_validate(data)
    except Exception as exc:
        raise ExportManifestError(f"{path} has invalid shape: {exc}") from exc


def verify_artifacts(export_dir: Path, manifest: AudioSnapshotManifest) -> None:
    """Re-hash each declared artifact and raise on mismatch."""
    for entry in manifest.artifacts:
        on_disk = export_dir / entry.path
        if not on_disk.exists():
            raise ExportManifestError(f"missing declared artifact: {on_disk}")
        actual = compute_sha256(on_disk)
        if actual != entry.sha256:
            raise ExportManifestError(
                f"sha256 mismatch for {entry.path}: "
                f"manifest={entry.sha256[:12]}… disk={actual[:12]}…"
            )
