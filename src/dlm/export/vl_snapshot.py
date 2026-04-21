"""HF-snapshot export for vision-language bases.

GGUF conversion for VL architectures is in flux upstream
(`convert_hf_to_gguf.py` doesn't register PaliGemma / Qwen2-VL /
InternVL2 consistently yet), so this path refuses to emit GGUF and
writes an HF snapshot instead: a self-contained directory that a
downstream user can load with `AutoProcessor.from_pretrained` +
`AutoModelForImageTextToText.from_pretrained` + `PeftModel.from_pretrained`.

Layout under `exports/hf-snapshot/`:

    adapter/                 # PEFT adapter dir (copy of the current store adapter)
    processor/               # processor config + tokenizer files
    snapshot_manifest.json   # see VlSnapshotManifest
    README.md                # how-to-load instructions for the recipient

The BASE model weights are NOT copied — they're pinned by `hf_id`
+ `revision` in the manifest. Recipients download the base once,
keyed on that revision. Bundling the base would balloon pack size
to 6 GB+ and conflicts with Gemma / Llama licensing for most VL
bases (they are `redistributable=False`).

The `export_target` field on the manifest is the load-bearing flag:
a future GGUF path will write `export_target="gguf"` instead.
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

VL_SNAPSHOT_SUBDIR = "hf-snapshot"
SNAPSHOT_MANIFEST_FILENAME = "snapshot_manifest.json"
SNAPSHOT_README_FILENAME = "README.md"


class VlSnapshotManifest(BaseModel):
    """Self-describing record of one HF-snapshot export.

    Parallel to `ExportManifest` but scoped to the VL path. No
    `quant` / `llama_cpp_tag` — the snapshot doesn't run llama.cpp.
    `export_target` is the discriminator: a future GGUF path adds a
    `"gguf"` branch to the same file when upstream converter support
    lands.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    export_target: Literal["hf_snapshot"] = "hf_snapshot"
    created_at: datetime
    created_by: str = Field(..., description="dlm version that wrote this manifest.")
    base_model_hf_id: str
    base_model_revision: str
    base_model_architecture: str
    modality: Literal["vision-language"] = "vision-language"
    image_token: str
    num_image_tokens: int
    target_size: tuple[int, int]
    adapter_version: int = Field(..., ge=1)
    adapter_name: str | None = None
    rationale: str = Field(
        default=(
            "Vision-language architectures in llama.cpp are in flux; "
            "this build emits an HF-snapshot fallback. A future release "
            "will add a GGUF path when upstream converter support "
            "stabilizes."
        ),
    )
    artifacts: list[ExportArtifact] = Field(default_factory=list)


@dataclass(frozen=True)
class VlSnapshotResult:
    """Return value of `run_vl_snapshot_export` — what the CLI prints."""

    export_dir: Path
    manifest_path: Path
    readme_path: Path
    adapter_dir: Path
    processor_dir: Path
    artifacts: list[Path]


def run_vl_snapshot_export(
    store: StorePath,
    spec: BaseModelSpec,
    *,
    adapter_name: str | None = None,
    adapter_path_override: Path | None = None,
    dlm_version: str = "dlm-0",
    processor: object | None = None,
) -> VlSnapshotResult:
    """Emit a VL HF-snapshot export under `exports/hf-snapshot/`.

    Resolves the adapter dir, copies it into the export directory,
    saves the processor (if supplied) under `processor/`, writes the
    manifest + README, and returns the layout paths.

    `processor=None` lets callers skip the processor save (tests, dry
    runs). Production paths pass an `AutoProcessor` loaded via
    `dlm.train.loader.load_processor`.
    """
    if spec.modality != "vision-language":
        raise ExportError(
            f"run_vl_snapshot_export: {spec.key!r} is modality={spec.modality!r}; "
            "only vision-language bases go through the HF-snapshot path"
        )
    if spec.vl_preprocessor_plan is None:
        raise ExportError(
            f"run_vl_snapshot_export: {spec.key!r} has modality='vision-language' "
            "but no vl_preprocessor_plan (this is a schema bug — file an issue)"
        )

    adapter_path, adapter_version = _resolve_adapter_for_export(
        store=store,
        adapter_name=adapter_name,
        adapter_path_override=adapter_path_override,
    )

    export_dir = store.exports / VL_SNAPSHOT_SUBDIR
    export_dir.mkdir(parents=True, exist_ok=True)

    adapter_out = export_dir / "adapter"
    _copy_adapter_dir(adapter_path, adapter_out)

    processor_out = export_dir / "processor"
    if processor is not None:
        processor_out.mkdir(parents=True, exist_ok=True)
        # HF processors implement `save_pretrained(dir)`. The method
        # writes tokenizer + image_processor configs + the processor
        # config itself — everything a recipient needs to re-hydrate.
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

    manifest = VlSnapshotManifest(
        created_at=_utc_now(),
        created_by=dlm_version,
        base_model_hf_id=spec.hf_id,
        base_model_revision=spec.revision,
        base_model_architecture=spec.architecture,
        image_token=spec.vl_preprocessor_plan.image_token,
        num_image_tokens=spec.vl_preprocessor_plan.num_image_tokens,
        target_size=spec.vl_preprocessor_plan.target_size,
        adapter_version=adapter_version,
        adapter_name=adapter_name,
        artifacts=[build_artifact(export_dir, p) for p in artifacts],
    )
    manifest_path = _save_manifest(export_dir, manifest)
    readme_path = _write_readme(export_dir, spec=spec, manifest=manifest)

    return VlSnapshotResult(
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
    """Return (adapter_dir, version) for the export.

    Mirrors the dispatch logic in `run_export` but minus the GGUF-
    specific concerns. Extracted so unit tests can exercise the
    refusal messages without touching the rest of the pipeline.
    """
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
    """Parse the `vNNNN` suffix on an adapter version directory name."""
    stem = path.name
    if not stem.startswith("v") or not stem[1:].isdigit():
        # Ephemeral merged-adapter dir won't match vNNNN; degrade to 1
        # rather than refuse — the export still works, the version is
        # just cosmetic in the manifest.
        return 1
    return int(stem[1:])


def _copy_adapter_dir(src: Path, dst: Path) -> None:
    """Copy the PEFT adapter directory into the export tree.

    `shutil.copytree` with `dirs_exist_ok=False` so repeat exports
    don't silently mix versions. Callers that want to overwrite
    delete `exports/hf-snapshot/` first.
    """
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _save_manifest(export_dir: Path, manifest: VlSnapshotManifest) -> Path:
    path = export_dir / SNAPSHOT_MANIFEST_FILENAME
    payload = manifest.model_dump(mode="json")
    blob = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    write_text(path, blob)
    return path


def _write_readme(
    export_dir: Path,
    *,
    spec: BaseModelSpec,
    manifest: VlSnapshotManifest,
) -> Path:
    """Write a human-readable load-instruction file.

    Recipients of the snapshot directory use this to understand what's
    in the tarball without opening the manifest JSON. Kept terse; the
    manifest is the authoritative record.
    """
    path = export_dir / SNAPSHOT_README_FILENAME
    body = (
        f"# HF-snapshot export\n"
        f"\n"
        f"Target: **{spec.key}** ({spec.hf_id} @ {spec.revision[:12]}…)\n"
        f"Adapter version: v{manifest.adapter_version:04d}"
        f"{f' ({manifest.adapter_name})' if manifest.adapter_name else ''}\n"
        f"\n"
        f"## Load this snapshot\n"
        f"\n"
        f"```python\n"
        f"from transformers import AutoModelForImageTextToText, AutoProcessor\n"
        f"from peft import PeftModel\n"
        f"\n"
        f"base = AutoModelForImageTextToText.from_pretrained(\n"
        f'    "{spec.hf_id}", revision="{spec.revision}",\n'
        f")\n"
        f'model = PeftModel.from_pretrained(base, "./adapter")\n'
        f'processor = AutoProcessor.from_pretrained("./processor")\n'
        f"```\n"
        f"\n"
        f"## Why HF snapshot (not GGUF)\n"
        f"\n"
        f"Vision-language converter support in `llama.cpp` is in flux.\n"
        f"A future release adds GGUF export when upstream stabilizes.\n"
    )
    write_text(path, body)
    return path


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)


def load_vl_snapshot_manifest(export_dir: Path) -> VlSnapshotManifest:
    """Read + validate `<export_dir>/snapshot_manifest.json`."""
    path = export_dir / SNAPSHOT_MANIFEST_FILENAME
    if not path.exists():
        raise ExportManifestError(f"missing {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportManifestError(f"cannot parse {path}: {exc}") from exc
    try:
        return VlSnapshotManifest.model_validate(data)
    except Exception as exc:
        raise ExportManifestError(f"{path} has invalid shape: {exc}") from exc


def verify_artifacts(export_dir: Path, manifest: VlSnapshotManifest) -> None:
    """Re-hash each declared artifact and raise on mismatch.

    Used by a downstream `dlm verify` pass to make sure the snapshot
    wasn't truncated in transit. Cheap — the snapshot is adapter-size,
    not base-size.
    """
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
