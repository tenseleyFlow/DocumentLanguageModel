"""End-to-end GGUF export — the single entry point `dlm export` calls.

Flow:

    preflight (adapter config / vocab / template)
    → base conversion  (HF → fp16 GGUF → quantized GGUF)         [cached by sha]
    → adapter conversion OR merged path                          [never both]
    → export_manifest.json                                        [atomic write]
    → manifest.exports.append                                    [atomic write]

`runner.run_export` is thin — each step lives in its own module so
unit tests can exercise the logic without standing up a real
llama.cpp. `subprocess_runner` is the injection seam the tests use to
substitute a no-op for the actual `subprocess.run` calls.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlm.export import adapter_gguf, base_gguf, merge, preflight
from dlm.export.manifest import (
    EXPORT_MANIFEST_FILENAME,
    ExportManifest,
    build_artifact,
    save_export_manifest,
    utc_now,
)
from dlm.export.plan import ExportPlan
from dlm.export.quantize import run_checked

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.store.paths import StorePath

_LOG = logging.getLogger(__name__)

# Type alias for the subprocess injection seam.
SubprocessRunner = Callable[[Sequence[str]], Any]


@dataclass(frozen=True)
class ExportResult:
    """Return value of `run_export` — what the CLI prints on success."""

    export_dir: Path
    manifest_path: Path
    artifacts: list[Path]
    plan: ExportPlan
    merged: bool
    cached: bool


def run_export(
    store: StorePath,
    spec: BaseModelSpec,
    plan: ExportPlan,
    *,
    cached_base_dir: Path,
    subprocess_runner: SubprocessRunner | None = None,
    vendor_override: Path | None = None,
) -> ExportResult:
    """Execute one GGUF export end-to-end.

    `cached_base_dir` is the HF cache path produced by
    `base_models.downloader.download_spec(spec).path`.

    `subprocess_runner` is a test seam — default wraps `run_checked`
    with a 10-min timeout + `SubprocessError` on failure.

    `vendor_override` is a test hook; production code leaves it `None`
    so `dlm.export.vendoring` resolves the real `vendor/llama.cpp/`
    submodule. Unit tests pass a populated tmp dir.
    """
    run = subprocess_runner if subprocess_runner is not None else run_checked

    adapter_path = store.resolve_current_adapter()
    if adapter_path is None or not adapter_path.exists():
        from dlm.export.errors import ExportError

        raise ExportError(
            f"no current adapter under {store.adapter_current_pointer}; "
            "run `dlm train` before exporting."
        )

    # 1. Preflight.
    preflight.check_adapter_config(adapter_path, spec)
    preflight.check_tokenizer_vocab(adapter_path)
    preflight.check_chat_template(adapter_path, required=plan.include_template)
    was_qlora = preflight.check_was_adapter_qlora(adapter_path)

    # 2. Merge-safety gate (pitfall #3).
    merge.check_merge_safety(plan, was_qlora=was_qlora)

    # 3. Prepare the output directory.
    export_dir = store.export_quant_dir(plan.quant)
    export_dir.mkdir(parents=True, exist_ok=True)

    # 4. Base conversion (cached).
    base_gguf_path = export_dir / f"base.{plan.quant}.gguf"
    cached = _cached_base_matches(export_dir, base_gguf_path, plan.quant)
    if not cached:
        _convert_and_quantize_base(
            cached_base_dir=cached_base_dir,
            export_dir=export_dir,
            base_gguf_path=base_gguf_path,
            plan=plan,
            run=run,
            vendor_override=vendor_override,
        )

    # 5. Adapter OR merged path.
    artifacts: list[Path] = [base_gguf_path]
    if plan.merged:
        # pragma: no cover — real merge requires HF + torch + model download.
        # The safety gate was already enforced; the full fp16-merge is
        # exercised by the slow integration test.
        _perform_merge_path(
            spec=spec,
            adapter_path=adapter_path,
            export_dir=export_dir,
            base_gguf_path=base_gguf_path,
            plan=plan,
            was_qlora=was_qlora,
            run=run,
        )
    else:
        adapter_gguf_path = export_dir / "adapter.gguf"
        cmd = adapter_gguf.build_convert_lora_args(
            adapter_path,
            out_gguf=adapter_gguf_path,
            script_override=vendor_override,
        )
        run(cmd)
        artifacts.append(adapter_gguf_path)

    # 6. Write export_manifest.json.
    from dlm import __version__ as dlm_version
    from dlm.export.vendoring import pinned_tag

    adapter_version = int(adapter_path.name.lstrip("v"))
    em = ExportManifest(
        quant=plan.quant,
        merged=plan.merged,
        dequantized=plan.dequantize_confirmed,
        ollama_name=plan.ollama_name,
        created_at=utc_now(),
        created_by=f"dlm-{dlm_version}",
        llama_cpp_tag=pinned_tag(vendor_override),
        base_model_hf_id=spec.hf_id,
        base_model_revision=spec.revision,
        adapter_version=adapter_version,
        artifacts=[build_artifact(export_dir, p) for p in artifacts],
    )
    manifest_path = save_export_manifest(export_dir, em)

    # 7. Append to store manifest.exports.
    _append_export_summary(
        store=store,
        plan=plan,
        llama_cpp_tag=em.llama_cpp_tag,
        artifacts=em.artifacts,
    )

    return ExportResult(
        export_dir=export_dir,
        manifest_path=manifest_path,
        artifacts=artifacts,
        plan=plan,
        merged=plan.merged,
        cached=cached,
    )


# --- internals ---------------------------------------------------------------


def _convert_and_quantize_base(
    *,
    cached_base_dir: Path,
    export_dir: Path,
    base_gguf_path: Path,
    plan: ExportPlan,
    run: SubprocessRunner,
    vendor_override: Path | None,
) -> None:
    """HF → fp16 GGUF → quantized GGUF. Removes the fp16 intermediate."""
    fp16_path = export_dir / "base.fp16.gguf"
    convert_cmd = base_gguf.build_convert_hf_args(
        cached_base_dir,
        out_fp16=fp16_path,
        script_override=vendor_override,
    )
    run(convert_cmd)

    quant_cmd = base_gguf.build_quantize_args(
        fp16_path,
        out_quant=base_gguf_path,
        quant=plan.quant,
        bin_override=vendor_override,
    )
    run(quant_cmd)

    # Don't keep the intermediate fp16 copy — users can regenerate if needed.
    if fp16_path.exists():
        fp16_path.unlink()


def _cached_base_matches(export_dir: Path, base_gguf_path: Path, quant: str) -> bool:
    """Return True iff `base.<quant>.gguf` already exists AND is referenced
    by a pre-existing `export_manifest.json` with matching sha256.
    """
    if not base_gguf_path.exists():
        return False
    manifest_path = export_dir / EXPORT_MANIFEST_FILENAME
    if not manifest_path.exists():
        return False
    try:
        from dlm.export.manifest import compute_sha256, load_export_manifest

        prior = load_export_manifest(export_dir)
    except Exception:
        return False
    if prior.quant != quant:
        return False
    recorded = next((a for a in prior.artifacts if a.path == base_gguf_path.name), None)
    if recorded is None:
        return False
    return compute_sha256(base_gguf_path) == recorded.sha256


def _perform_merge_path(  # pragma: no cover
    *,
    spec: BaseModelSpec,
    adapter_path: Path,
    export_dir: Path,
    base_gguf_path: Path,
    plan: ExportPlan,
    was_qlora: bool,
    run: SubprocessRunner,
) -> None:
    """Load base + adapter in fp16, merge, save as HF, then re-quantize.

    Replaces `base.<quant>.gguf` with the merged weights. Fully
    executed only under the slow integration test.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_hf = Path(tmp) / "merged_hf"
        merge.perform_merge(spec, adapter_path, tmp_hf, was_qlora=was_qlora)

        fp16_path = export_dir / "merged.fp16.gguf"
        run(base_gguf.build_convert_hf_args(tmp_hf, out_fp16=fp16_path))
        run(base_gguf.build_quantize_args(fp16_path, out_quant=base_gguf_path, quant=plan.quant))
        if fp16_path.exists():
            fp16_path.unlink()


def _append_export_summary(
    *,
    store: StorePath,
    plan: ExportPlan,
    llama_cpp_tag: str | None,
    artifacts: list[Any],
) -> None:
    """Update `manifest.exports` with a new `ExportSummary` row."""
    from dlm.store.manifest import ExportSummary, load_manifest, save_manifest

    base_sha = next((a.sha256 for a in artifacts if a.path.startswith("base.")), None)
    adapter_sha = next((a.sha256 for a in artifacts if a.path.startswith("adapter.")), None)

    summary = ExportSummary(
        exported_at=utc_now(),
        quant=plan.quant,
        merged=plan.merged,
        ollama_name=plan.ollama_name,
        llama_cpp_tag=llama_cpp_tag,
        base_gguf_sha256=base_sha,
        adapter_gguf_sha256=adapter_sha,
    )

    manifest = load_manifest(store.manifest)
    updated = manifest.model_copy(
        update={
            "exports": [*manifest.exports, summary],
            "updated_at": utc_now(),
        }
    )
    save_manifest(store.manifest, updated)
