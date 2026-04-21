"""Single-file VL GGUF emitter for SUPPORTED vision-language bases.

Entered only when `arch_probe.probe_gguf_arch` returns
`SupportLevel.SUPPORTED` (today: `Qwen2VLForConditionalGeneration` at
the pinned llama.cpp tag). PaliGemma and InternVL2 remain UNSUPPORTED
upstream and route to the HF-snapshot fallback via the dispatcher.

**Merged-only.** `convert_lora_to_gguf.py` doesn't recognize the
`model.language_model.*` layer-naming convention that VL adapters
carry, so we refuse non-merged plans and let the dispatcher fall back
to HF-snapshot. The renderer emits `FROM ./base.<quant>.gguf` with no
`ADAPTER` line as a consequence. See `render_vl_modelfile` docstring.

**Single-file vs mmproj.** Upstream at tag `b8816` writes a single
GGUF for Qwen2-VL — the ViT is dropped and Ollama's preprocessor runs
the vision path on its own. Manifest records `mmproj_path=None`; a
future tag that changes this (split emission with an `mmproj-*.gguf`
sidecar) would add a new field without breaking the single-file
contract.

**No imatrix.** The replay corpus is text-only; imatrix calibration
would mis-weight vision-adjacent layers on any future VL base that
bakes projector tensors into the GGUF. Quant proceeds without
`--imatrix`.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlm.export import base_gguf, merge, preflight
from dlm.export.arch_probe import ArchProbeResult, SupportLevel
from dlm.export.errors import ExportError, VlGgufUnsupportedError
from dlm.export.manifest import (
    ExportManifest,
    build_artifact,
    compute_sha256,
    save_export_manifest,
    utc_now,
)
from dlm.export.ollama.vl_modelfile import VlModelfileContext, render_vl_modelfile
from dlm.export.plan import ExportPlan
from dlm.export.quantize import run_checked
from dlm.io.atomic import write_text as atomic_write_text

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.store.paths import StorePath

_LOG = logging.getLogger(__name__)

# Injection seam matching `runner.py` — a callable that runs subprocess
# args and returns a `CompletedProcess`-shaped result. Unit tests
# substitute a recorder; production passes through to `run_checked`.
SubprocessRunner = Callable[[Sequence[str]], Any]


@dataclass(frozen=True)
class VlGgufResult:
    """Return value of `run_vl_gguf_export` — what the CLI prints."""

    export_dir: Path
    manifest_path: Path
    modelfile_path: Path
    gguf_path: Path
    mmproj_path: Path | None  # reserved for future split-emission archs
    quant: str
    llama_cpp_tag: str | None
    artifacts: list[Path]


def _assert_supported(verdict: ArchProbeResult, plan: ExportPlan) -> None:
    """Gate: refuse anything the emitter doesn't claim to handle.

    Three preconditions, each with a distinct error message so the
    dispatcher's fallback banner names the actual reason:

    1. `verdict.support is SupportLevel.SUPPORTED` — otherwise upstream
       `convert_hf_to_gguf.py` would either fail outright (UNSUPPORTED)
       or drop critical tensors (PARTIAL → would ship an
       under-converted GGUF silently).
    2. `plan.merged is True` — LoRA-to-GGUF for VL archs isn't
       plumbed upstream at our tag; merged-only is the safe shape.
    3. `plan.imatrix == "off"` — the replay corpus is text-only;
       importance-matrix calibration would mis-weight vision-adjacent
       layers once a future arch bakes them into the GGUF.
    """
    if verdict.support is not SupportLevel.SUPPORTED:
        raise VlGgufUnsupportedError(
            f"arch {verdict.arch_class!r} verdict={verdict.support.value!r} "
            f"at llama.cpp tag={verdict.llama_cpp_tag!r}; single-file VL GGUF "
            "emission requires SUPPORTED. Fallback: HF-snapshot."
        )
    if not plan.merged:
        raise VlGgufUnsupportedError(
            "VL GGUF emission is merged-only at this upstream tag (LoRA-to-GGUF "
            "for VL archs isn't supported by convert_lora_to_gguf.py). Pass "
            "--merged, or fall back to HF-snapshot."
        )
    if plan.imatrix != "off":
        raise VlGgufUnsupportedError(
            "VL GGUF emission refuses imatrix calibration: the replay corpus "
            "is text-only and would mis-weight vision-adjacent quant stats. "
            "Pass --no-imatrix (or omit --imatrix) to proceed."
        )


def _resolve_adapter(
    store: StorePath,
    *,
    adapter_name: str | None,
) -> tuple[Path, int]:
    """Return (adapter_path, version) with a typed refusal on empty stores.

    Mirrors the dispatch logic in `runner._resolve_adapter_for_export`
    but scoped to the VL path so the VL module stays self-contained.
    Callers receive a `VlGgufUnsupportedError` (not a generic
    `ExportError`) so the dispatcher's fallback banner fires correctly.
    """
    if adapter_name is None:
        resolved = store.resolve_current_adapter()
        pointer = store.adapter_current_pointer
    else:
        resolved = store.resolve_current_adapter_for(adapter_name)
        pointer = store.adapter_current_pointer_for(adapter_name)
    if resolved is None or not resolved.exists():
        raise VlGgufUnsupportedError(
            f"no current adapter under {pointer}; run `dlm train` before exporting."
        )
    version = _version_from_dir_name(resolved)
    return resolved, version


def _version_from_dir_name(path: Path) -> int:
    stem = path.name
    if not stem.startswith("v") or not stem[1:].isdigit():
        return 1
    return int(stem[1:])


def run_vl_gguf_export(
    store: StorePath,
    spec: BaseModelSpec,
    plan: ExportPlan,
    *,
    verdict: ArchProbeResult,
    cached_base_dir: Path,
    adapter_name: str | None = None,
    system_prompt: str | None = None,
    source_dlm_path: Path | None = None,
    dlm_version: str = "dev",
    training_sequence_len: int | None = None,
    subprocess_runner: SubprocessRunner | None = None,
    merge_runner: Callable[..., None] | None = None,
) -> VlGgufResult:
    """Orchestrate merge → convert → quantize → Modelfile for a VL base.

    `subprocess_runner` and `merge_runner` are injection seams: the
    production path wires in `run_checked` + `merge.perform_vl_merge`;
    unit tests substitute recorders. Every arg after `plan` is
    keyword-only — the production call is verbose but unambiguous.

    Returns a `VlGgufResult`; raises `VlGgufUnsupportedError` or
    `ExportError` on any precondition or subprocess failure. The
    dispatcher catches and falls back to HF-snapshot.
    """
    _assert_supported(verdict, plan)

    adapter_path, adapter_version = _resolve_adapter(store, adapter_name=adapter_name)

    preflight.check_adapter_config(adapter_path, spec)
    preflight.check_tokenizer_vocab(adapter_path)
    preflight.check_chat_template(adapter_path, required=False)
    preflight.check_vl_target_modules_lm_only(adapter_path)
    was_qlora = preflight.check_was_adapter_qlora(adapter_path)
    merge.check_merge_safety(plan, was_qlora=was_qlora)

    export_dir = store.exports / f"vl-gguf-{plan.quant}"
    export_dir.mkdir(parents=True, exist_ok=True)

    base_gguf_name = f"base.{plan.quant}.gguf"
    gguf_path = export_dir / base_gguf_name
    modelfile_path = export_dir / "Modelfile"

    run = subprocess_runner if subprocess_runner is not None else _default_runner
    do_merge = merge_runner if merge_runner is not None else merge.perform_vl_merge

    # Merge → fp16 HF dir → GGUF → quantized GGUF. Temp dir holds the
    # merged HF snapshot + fp16 GGUF — both are multi-GB artifacts we
    # don't need after the quantized GGUF lands in `export_dir`.
    with tempfile.TemporaryDirectory(prefix="dlm-vl-gguf-") as tmp_s:
        tmp = Path(tmp_s)
        merged_hf = tmp / "merged"
        fp16_gguf = tmp / f"base.{plan.quant}.f16.gguf"

        do_merge(adapter_path, merged_hf, cached_base_dir=cached_base_dir)

        run(base_gguf.build_convert_hf_args(merged_hf, out_fp16=fp16_gguf))
        run(base_gguf.build_quantize_args(fp16_gguf, out_quant=gguf_path, quant=plan.quant))

    if not gguf_path.exists():
        raise ExportError(
            f"VL GGUF emission: expected {gguf_path} after llama-quantize; "
            "subprocess succeeded but the file is missing (check disk space + "
            "vendored build)."
        )

    modelfile_body = render_vl_modelfile(
        VlModelfileContext(
            spec=spec,
            plan=plan,
            adapter_dir=adapter_path,
            base_gguf_name=base_gguf_name,
            adapter_gguf_name=None,  # merged-only path, no ADAPTER directive
            dlm_id=store.root.name,
            adapter_version=adapter_version,
            system_prompt=system_prompt,
            source_dlm_path=source_dlm_path,
            dlm_version=dlm_version,
            training_sequence_len=training_sequence_len,
        )
    )
    atomic_write_text(modelfile_path, modelfile_body)

    artifacts_files = [gguf_path, modelfile_path]
    manifest = ExportManifest(
        created_at=utc_now(),
        created_by=dlm_version,
        base_model_hf_id=spec.hf_id,
        base_model_revision=spec.revision,
        quant=plan.quant,
        merged=plan.merged,
        ollama_name=None,
        llama_cpp_tag=verdict.llama_cpp_tag,
        adapter_version=adapter_version,
        artifacts=[build_artifact(export_dir, p) for p in artifacts_files],
    )
    manifest_path = save_export_manifest(export_dir, manifest)

    # Attach a compact `vl_gguf.json` sidecar capturing the arch verdict
    # + the mmproj contract explicitly. The main manifest carries the
    # llama.cpp tag but not the SUPPORTED/PARTIAL verdict string.
    _write_vl_sidecar(export_dir, verdict=verdict, gguf_sha=compute_sha256(gguf_path))

    return VlGgufResult(
        export_dir=export_dir,
        manifest_path=manifest_path,
        modelfile_path=modelfile_path,
        gguf_path=gguf_path,
        mmproj_path=None,
        quant=plan.quant,
        llama_cpp_tag=verdict.llama_cpp_tag,
        artifacts=artifacts_files,
    )


def _default_runner(args: Sequence[str]) -> Any:
    """Production subprocess runner — `run_checked` from the quantize module.

    Pulled out so `run_vl_gguf_export`'s default is easy to override in
    tests via the `subprocess_runner` kwarg.
    """
    return run_checked(list(args), timeout=60 * 60)


def _write_vl_sidecar(
    export_dir: Path,
    *,
    verdict: ArchProbeResult,
    gguf_sha: str,
) -> None:
    """Persist the arch verdict + mmproj contract alongside the manifest."""
    sidecar = {
        "schema": 1,
        "arch_verdict": {
            "architecture": verdict.arch_class,
            "support": verdict.support.value,
            "llama_cpp_tag": verdict.llama_cpp_tag,
            "reason": verdict.reason,
        },
        "mmproj_path": None,
        "gguf_sha256": gguf_sha,
    }
    atomic_write_text(
        export_dir / "vl_gguf.json",
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
    )
