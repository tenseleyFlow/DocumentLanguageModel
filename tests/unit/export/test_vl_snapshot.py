"""VL HF-snapshot export (Sprint 35 v1) — manifest shape + layout.

Covers:

- `run_vl_snapshot_export` refuses text-modality specs.
- Writes adapter + manifest + README to the right paths.
- Manifest carries export_target="hf_snapshot" + the base's
  VlPreprocessorPlan params.
- `verify_artifacts` round-trips (no tampering → no raise).
- `load_vl_snapshot_manifest` deserializes what `_save_manifest` wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dlm.base_models.schema import BaseModelSpec, VlPreprocessorPlan
from dlm.export.errors import ExportError, ExportManifestError
from dlm.export.vl_snapshot import (
    SNAPSHOT_MANIFEST_FILENAME,
    SNAPSHOT_README_FILENAME,
    VL_SNAPSHOT_SUBDIR,
    VlSnapshotManifest,
    load_vl_snapshot_manifest,
    run_vl_snapshot_export,
    verify_artifacts,
)
from dlm.store.paths import for_dlm

_VALID_ULID = "01KPMGSTNGSTTSTTSTTSTTSTVA"


def _vl_spec(**overrides: object) -> BaseModelSpec:
    base_kwargs: dict[str, object] = {
        "key": "paligemma-test",
        "hf_id": "google/paligemma-test",
        "revision": "b" * 40,
        "architecture": "PaliGemmaForConditionalGeneration",
        "params": 3_000_000_000,
        "target_modules": ["q_proj"],
        "template": "paligemma",
        "gguf_arch": "paligemma",
        "tokenizer_pre": "gemma",
        "license_spdx": "Other",
        "redistributable": False,
        "size_gb_fp16": 6.0,
        "context_length": 8192,
        "recommended_seq_len": 2048,
        "modality": "vision-language",
        "vl_preprocessor_plan": VlPreprocessorPlan(
            target_size=(224, 224),
            image_token="<image>",
            num_image_tokens=256,
        ),
    }
    base_kwargs.update(overrides)
    return BaseModelSpec(**base_kwargs)  # type: ignore[arg-type]


def _text_spec() -> BaseModelSpec:
    return BaseModelSpec(
        key="text-base",
        hf_id="org/text-base",
        revision="a" * 40,
        architecture="LlamaForCausalLM",
        params=1_000_000,
        target_modules=["q_proj"],
        template="chatml",
        gguf_arch="llama",
        tokenizer_pre="llama-bpe",
        license_spdx="Apache-2.0",
        redistributable=True,
        size_gb_fp16=0.5,
        context_length=4096,
        recommended_seq_len=1024,
    )


@pytest.fixture
def populated_store(tmp_path: Path):
    """StorePath with a fake adapter at `adapter/versions/v0001/`."""
    store = for_dlm(_VALID_ULID, home=tmp_path)
    store.ensure_layout()
    v1 = store.adapter_version(1)
    v1.mkdir(parents=True, exist_ok=True)
    (v1 / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    (v1 / "adapter_model.safetensors").write_bytes(b"fake adapter bytes")
    store.set_current_adapter(v1)
    return store


class TestRefusals:
    def test_text_spec_refused(self, populated_store) -> None:
        with pytest.raises(ExportError, match="only vision-language bases"):
            run_vl_snapshot_export(populated_store, _text_spec())

    def test_missing_vl_preprocessor_plan_refused(self, populated_store) -> None:
        spec = _vl_spec()
        object.__setattr__(spec, "vl_preprocessor_plan", None)
        with pytest.raises(ExportError, match="no vl_preprocessor_plan"):
            run_vl_snapshot_export(populated_store, spec)

    def test_missing_adapter_refused(self, tmp_path: Path) -> None:
        store = for_dlm(_VALID_ULID, home=tmp_path)
        store.ensure_layout()
        with pytest.raises(ExportError, match="no current adapter"):
            run_vl_snapshot_export(store, _vl_spec())


class TestSnapshotLayout:
    def test_export_dir_under_exports_hf_snapshot(self, populated_store) -> None:
        result = run_vl_snapshot_export(populated_store, _vl_spec())
        assert result.export_dir.name == VL_SNAPSHOT_SUBDIR
        assert result.export_dir.parent == populated_store.exports

    def test_adapter_files_copied(self, populated_store) -> None:
        result = run_vl_snapshot_export(populated_store, _vl_spec())
        assert (result.adapter_dir / "adapter_config.json").exists()
        assert (
            result.adapter_dir / "adapter_model.safetensors"
        ).read_bytes() == b"fake adapter bytes"

    def test_manifest_and_readme_written(self, populated_store) -> None:
        result = run_vl_snapshot_export(populated_store, _vl_spec())
        assert result.manifest_path.name == SNAPSHOT_MANIFEST_FILENAME
        assert result.readme_path.name == SNAPSHOT_README_FILENAME
        assert result.manifest_path.exists()
        assert result.readme_path.exists()

    def test_repeat_export_overwrites_adapter(self, populated_store) -> None:
        # First export.
        run_vl_snapshot_export(populated_store, _vl_spec())
        # Mutate the source adapter and re-export — the copy should reflect it.
        v1 = populated_store.adapter_version(1)
        (v1 / "adapter_model.safetensors").write_bytes(b"new bytes")
        result = run_vl_snapshot_export(populated_store, _vl_spec())
        assert (result.adapter_dir / "adapter_model.safetensors").read_bytes() == b"new bytes"

    def test_named_adapter_export_uses_named_current_pointer(self, populated_store) -> None:
        named = populated_store.adapter_version_for("knowledge", 7)
        named.mkdir(parents=True, exist_ok=True)
        (named / "adapter_config.json").write_text('{"r": 32}', encoding="utf-8")
        (named / "adapter_model.safetensors").write_bytes(b"named bytes")
        populated_store.set_current_adapter_for("knowledge", named)

        result = run_vl_snapshot_export(
            populated_store,
            _vl_spec(),
            adapter_name="knowledge",
        )

        assert (result.adapter_dir / "adapter_model.safetensors").read_bytes() == b"named bytes"
        manifest = load_vl_snapshot_manifest(result.export_dir)
        assert manifest.adapter_version == 7
        assert manifest.adapter_name == "knowledge"

    def test_adapter_override_uses_provided_dir(self, populated_store, tmp_path: Path) -> None:
        override = tmp_path / "merged-adapter"
        override.mkdir()
        (override / "adapter_model.safetensors").write_bytes(b"override bytes")

        result = run_vl_snapshot_export(
            populated_store,
            _vl_spec(),
            adapter_path_override=override,
        )

        assert (result.adapter_dir / "adapter_model.safetensors").read_bytes() == b"override bytes"
        manifest = load_vl_snapshot_manifest(result.export_dir)
        assert manifest.adapter_version == 1

    def test_missing_adapter_override_refused(self, populated_store, tmp_path: Path) -> None:
        with pytest.raises(ExportError, match="adapter_path_override .* does not exist"):
            run_vl_snapshot_export(
                populated_store,
                _vl_spec(),
                adapter_path_override=tmp_path / "missing",
            )

    def test_processor_save_pretrained_writes_processor_artifact(self, populated_store) -> None:
        class _Processor:
            def save_pretrained(self, out_dir: str) -> None:
                Path(out_dir, "processor_config.json").write_text("{}", encoding="utf-8")

        result = run_vl_snapshot_export(populated_store, _vl_spec(), processor=_Processor())

        assert (result.processor_dir / "processor_config.json").exists()
        manifest = load_vl_snapshot_manifest(result.export_dir)
        paths = {entry.path for entry in manifest.artifacts}
        assert "processor/processor_config.json" in paths

    def test_noncallable_processor_save_is_ignored(self, populated_store) -> None:
        class _Processor:
            save_pretrained = "not-callable"

        result = run_vl_snapshot_export(populated_store, _vl_spec(), processor=_Processor())

        assert result.processor_dir.exists()
        assert not any(result.processor_dir.iterdir())


class TestManifestContent:
    def test_export_target_is_hf_snapshot(self, populated_store) -> None:
        run_vl_snapshot_export(populated_store, _vl_spec())
        manifest = load_vl_snapshot_manifest(populated_store.exports / VL_SNAPSHOT_SUBDIR)
        assert manifest.export_target == "hf_snapshot"
        assert manifest.modality == "vision-language"

    def test_base_pinned_in_manifest(self, populated_store) -> None:
        run_vl_snapshot_export(populated_store, _vl_spec())
        manifest = load_vl_snapshot_manifest(populated_store.exports / VL_SNAPSHOT_SUBDIR)
        assert manifest.base_model_hf_id == "google/paligemma-test"
        assert manifest.base_model_revision == "b" * 40
        assert manifest.base_model_architecture == "PaliGemmaForConditionalGeneration"

    def test_preprocessor_params_pinned(self, populated_store) -> None:
        run_vl_snapshot_export(populated_store, _vl_spec())
        manifest = load_vl_snapshot_manifest(populated_store.exports / VL_SNAPSHOT_SUBDIR)
        assert manifest.image_token == "<image>"
        assert manifest.num_image_tokens == 256
        assert manifest.target_size == (224, 224)

    def test_adapter_version_recorded(self, populated_store) -> None:
        run_vl_snapshot_export(populated_store, _vl_spec())
        manifest = load_vl_snapshot_manifest(populated_store.exports / VL_SNAPSHOT_SUBDIR)
        assert manifest.adapter_version == 1

    def test_adapter_artifacts_listed(self, populated_store) -> None:
        run_vl_snapshot_export(populated_store, _vl_spec())
        manifest = load_vl_snapshot_manifest(populated_store.exports / VL_SNAPSHOT_SUBDIR)
        paths = {entry.path for entry in manifest.artifacts}
        assert "adapter/adapter_config.json" in paths
        assert "adapter/adapter_model.safetensors" in paths


class TestVerifyArtifacts:
    def test_pristine_snapshot_verifies(self, populated_store) -> None:
        run_vl_snapshot_export(populated_store, _vl_spec())
        export_dir = populated_store.exports / VL_SNAPSHOT_SUBDIR
        manifest = load_vl_snapshot_manifest(export_dir)
        verify_artifacts(export_dir, manifest)  # no raise

    def test_tampered_artifact_detected(self, populated_store) -> None:
        run_vl_snapshot_export(populated_store, _vl_spec())
        export_dir = populated_store.exports / VL_SNAPSHOT_SUBDIR
        manifest = load_vl_snapshot_manifest(export_dir)
        # Corrupt one artifact on disk.
        target = export_dir / manifest.artifacts[0].path
        target.write_bytes(b"tampered")
        with pytest.raises(ExportManifestError, match="sha256 mismatch"):
            verify_artifacts(export_dir, manifest)

    def test_missing_artifact_detected(self, populated_store) -> None:
        run_vl_snapshot_export(populated_store, _vl_spec())
        export_dir = populated_store.exports / VL_SNAPSHOT_SUBDIR
        manifest = load_vl_snapshot_manifest(export_dir)
        (export_dir / manifest.artifacts[0].path).unlink()
        with pytest.raises(ExportManifestError, match="missing declared artifact"):
            verify_artifacts(export_dir, manifest)


class TestManifestLoadFailures:
    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ExportManifestError, match="missing"):
            load_vl_snapshot_manifest(tmp_path)

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        (tmp_path / SNAPSHOT_MANIFEST_FILENAME).write_text("not json", encoding="utf-8")
        with pytest.raises(ExportManifestError, match="cannot parse"):
            load_vl_snapshot_manifest(tmp_path)

    def test_invalid_shape_raises(self, tmp_path: Path) -> None:
        (tmp_path / SNAPSHOT_MANIFEST_FILENAME).write_text(
            json.dumps({"created_by": "dlm-test"}),
            encoding="utf-8",
        )
        with pytest.raises(ExportManifestError, match="invalid shape"):
            load_vl_snapshot_manifest(tmp_path)


class TestManifestModelDirect:
    def test_frozen(self) -> None:
        from datetime import UTC, datetime

        manifest = VlSnapshotManifest(
            created_at=datetime.now(UTC).replace(tzinfo=None),
            created_by="dlm-test",
            base_model_hf_id="x/y",
            base_model_revision="a" * 40,
            base_model_architecture="X",
            image_token="<image>",
            num_image_tokens=256,
            target_size=(224, 224),
            adapter_version=1,
        )
        with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
            manifest.adapter_version = 2  # type: ignore[misc]
