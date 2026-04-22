"""Modality-aware export dispatcher — pure logic, no CLI I/O.

Three public entry points (`dispatch_vl_export`, `dispatch_audio_export`,
`emit_vl_snapshot`) wrap the existing single-modality runners
(`run_vl_gguf_export`, `run_vl_snapshot_export`, `run_audio_snapshot_export`)
with the arch-probe + fallback decision tree that used to live inline
in `dlm.cli.commands`. They return :class:`DispatchResult`; the CLI
prints `banner_lines` and handles typed errors.

Why pull this out of the CLI:

- Business logic lived alongside `console.print` + `typer.Exit`, which
  made the dispatch decisions untestable without booting a full Typer
  runner. Returning data (lines + paths) means unit tests can assert
  which path was taken from the banner contents.
- `cli/commands.py` had direct heavy imports (`run_vl_gguf_export`,
  `run_audio_snapshot_export`) and modality `== "vision-language"`
  branches — both targeted by Sprint 38's B5 extraction.

All user-facing flow control is still here; the CLI is the thin
parse+print layer on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlm.export.errors import (
    ExportError,
    ProcessorLoadError,
    VendoringError,
    VlGgufUnsupportedError,
)

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.store.paths import StorePath


@dataclass(frozen=True)
class DispatchResult:
    """What the CLI needs to print after a successful dispatch.

    `banner_lines` are pre-formatted Rich markup accumulated during
    the dispatch (fallback announcements, final success). The CLI
    iterates them in order. `extras` carries path-specific extras
    (GGUF path, llama.cpp tag) under documented keys per modality.
    """

    export_dir: Path
    manifest_path: Path
    artifacts: list[Path]
    banner_lines: list[str]
    extras: dict[str, Any] = field(default_factory=dict)


def _load_processor_or_raise(spec: BaseModelSpec) -> Any:
    """Load HF processor, wrap any failure in :class:`ProcessorLoadError`.

    The HF snapshot exports need the processor to be loadable on the
    recipient side — shipping an incomplete tarball is worse than a
    crisp refusal. All flavors of import/network/license failure
    collapse into one typed error the CLI can print in one line.
    """
    from dlm.train.loader import load_processor  # pragma: no cover - heavy

    try:
        return load_processor(spec)  # pragma: no cover - heavy
    except Exception as exc:  # noqa: BLE001 - wrapping heterogeneous HF errors
        raise ProcessorLoadError(
            f"could not load processor for {spec.key!r} "
            f"({type(exc).__name__}: {exc}). "
            "The HF-snapshot export needs the processor to be loadable — "
            "verify license acceptance + network + cache, then re-run."
        ) from exc


def emit_vl_snapshot(
    *,
    store: StorePath,
    spec: BaseModelSpec,
    adapter_name: str | None,
    quant: str | None,
    merged: bool,
    adapter_mix_raw: str | None,
    skip_gguf_flag_warning: bool = False,
) -> DispatchResult:
    """Emit the HF-snapshot VL artifact + return its banner lines.

    Kept separate from the probe logic so the dispatcher can reach
    this both on non-SUPPORTED verdicts and on a GGUF emission
    fallback. `skip_gguf_flag_warning` is True on the fallback path —
    the user already saw a "GGUF emission refused" banner, and
    re-warning about --quant/--merged would be noisy.
    """
    from dlm.export.vl_snapshot import run_vl_snapshot_export

    banner: list[str] = []
    if not skip_gguf_flag_warning and (quant is not None or merged or adapter_mix_raw is not None):
        banner.append(
            "[yellow]export:[/yellow] ignoring GGUF-only flags "
            "(--quant / --merged / --adapter-mix) — they're not applicable "
            "to the HF-snapshot path."
        )

    processor = _load_processor_or_raise(spec)
    result = run_vl_snapshot_export(
        store,
        spec,
        adapter_name=adapter_name,
        processor=processor,
    )

    banner.append(
        f"[green]export:[/green] HF snapshot written to {result.export_dir}\n"
        f"  manifest: {result.manifest_path.name}\n"
        f"  adapter:  {result.adapter_dir}\n"
        f"  artifacts: {len(result.artifacts)} file(s)"
    )
    return DispatchResult(
        export_dir=result.export_dir,
        manifest_path=result.manifest_path,
        artifacts=list(result.artifacts),
        banner_lines=banner,
        extras={"adapter_dir": result.adapter_dir, "path": "hf-snapshot"},
    )


def dispatch_vl_export(
    *,
    store: StorePath,
    spec: BaseModelSpec,
    adapter_name: str | None,
    quant: str | None,
    merged: bool,
    adapter_mix_raw: str | None,
    gguf_emission_context: dict[str, Any] | None = None,
) -> DispatchResult:
    """Route a VL spec through the GGUF or HF-snapshot export path.

    Probes the vendored llama.cpp for arch coverage and picks a path:

    - **SUPPORTED** + `gguf_emission_context` present → try single-file
      GGUF emission via `run_vl_gguf_export`. On `VlGgufUnsupportedError`
      (plan refusal), `VendoringError` (missing/unbuilt vendor), or
      `ExportError` (subprocess failure), fall back to HF-snapshot
      with an explanatory banner.
    - **PARTIAL** → HF-snapshot with a banner explaining the split-arch
      caveat (vision tower would require an mmproj sidecar upstream
      doesn't emit at our pinned tag).
    - **UNSUPPORTED** (or probe failure) → HF-snapshot with a banner
      pointing the user at `scripts/bump-llama-cpp.sh`.

    `gguf_emission_context` carries everything the GGUF path needs
    (plan, cached base dir, source dlm path, sequence len, dlm
    version). `None` forces the snapshot path.
    """
    from dlm.export.arch_probe import SupportLevel, probe_gguf_arch
    from dlm.export.vl_gguf import run_vl_gguf_export

    probe_banner: list[str] = []
    try:
        verdict = probe_gguf_arch(spec.architecture)
    except VendoringError as exc:
        probe_banner.append(
            f"[yellow]export:[/yellow] llama.cpp probe unavailable ({exc}); "
            "falling back to HF-snapshot without a GGUF verdict."
        )
        verdict = None

    if verdict is None or verdict.support is SupportLevel.UNSUPPORTED:
        tag_note = f"at tag={verdict.llama_cpp_tag or 'unknown'} " if verdict is not None else ""
        probe_banner.append(
            f"[yellow]export:[/yellow] base {spec.key!r} "
            f"(arch={spec.architecture}) is not covered by the vendored "
            f"llama.cpp {tag_note}— emitting HF-snapshot. Run "
            "`scripts/bump-llama-cpp.sh` to pull a newer tag if upstream "
            "has added support, or ship this adapter as a snapshot."
        )
        result = emit_vl_snapshot(
            store=store,
            spec=spec,
            adapter_name=adapter_name,
            quant=quant,
            merged=merged,
            adapter_mix_raw=adapter_mix_raw,
        )
        return _prepend_banner(probe_banner, result)

    if verdict.support is SupportLevel.PARTIAL:
        probe_banner.append(
            f"[yellow]export:[/yellow] base {spec.key!r} has PARTIAL "
            "llama.cpp coverage (vision tower ships as mmproj sidecar). "
            "Emitting HF-snapshot — single-file GGUF emission for "
            "split VL archs is gated on upstream mmproj support."
        )
        result = emit_vl_snapshot(
            store=store,
            spec=spec,
            adapter_name=adapter_name,
            quant=quant,
            merged=merged,
            adapter_mix_raw=adapter_mix_raw,
        )
        return _prepend_banner(probe_banner, result)

    # SUPPORTED
    if gguf_emission_context is None:
        probe_banner.append(
            f"[yellow]export:[/yellow] base {spec.key!r} is SUPPORTED by "
            f"llama.cpp (tag={verdict.llama_cpp_tag or 'unknown'}), but "
            "this dispatcher was invoked without GGUF plan context — "
            "emitting HF-snapshot."
        )
        result = emit_vl_snapshot(
            store=store,
            spec=spec,
            adapter_name=adapter_name,
            quant=quant,
            merged=merged,
            adapter_mix_raw=adapter_mix_raw,
        )
        return _prepend_banner(probe_banner, result)

    probe_banner.append(
        f"[dim]export:[/dim] base {spec.key!r} is SUPPORTED by llama.cpp "
        f"(tag={verdict.llama_cpp_tag or 'unknown'}); attempting single-file "
        "VL GGUF emission."
    )
    try:
        gguf_result = run_vl_gguf_export(
            store,
            spec,
            gguf_emission_context["plan"],
            verdict=verdict,
            cached_base_dir=gguf_emission_context["cached_base_dir"],
            adapter_name=adapter_name,
            system_prompt=gguf_emission_context.get("system_prompt"),
            source_dlm_path=gguf_emission_context.get("source_dlm_path"),
            dlm_version=gguf_emission_context.get("dlm_version", "dev"),
            training_sequence_len=gguf_emission_context.get("training_sequence_len"),
        )
    except VlGgufUnsupportedError as exc:
        probe_banner.append(
            f"[yellow]export:[/yellow] VL GGUF emission refused ({exc}); "
            "falling back to HF-snapshot."
        )
        result = emit_vl_snapshot(
            store=store,
            spec=spec,
            adapter_name=adapter_name,
            quant=quant,
            merged=merged,
            adapter_mix_raw=adapter_mix_raw,
            skip_gguf_flag_warning=True,
        )
        return _prepend_banner(probe_banner, result)
    except (VendoringError, ExportError) as exc:
        probe_banner.append(
            f"[yellow]export:[/yellow] VL GGUF emission failed "
            f"({type(exc).__name__}: {exc}); falling back to HF-snapshot."
        )
        result = emit_vl_snapshot(
            store=store,
            spec=spec,
            adapter_name=adapter_name,
            quant=quant,
            merged=merged,
            adapter_mix_raw=adapter_mix_raw,
            skip_gguf_flag_warning=True,
        )
        return _prepend_banner(probe_banner, result)

    probe_banner.append(
        f"[green]export:[/green] VL GGUF written to {gguf_result.export_dir}\n"
        f"  manifest:  {gguf_result.manifest_path.name}\n"
        f"  gguf:      {gguf_result.gguf_path.name} ({gguf_result.quant})\n"
        f"  Modelfile: {gguf_result.modelfile_path.name}\n"
        f"  llama.cpp: {gguf_result.llama_cpp_tag or 'unknown'}\n"
        f"  artifacts: {len(gguf_result.artifacts)} file(s)"
    )
    return DispatchResult(
        export_dir=gguf_result.export_dir,
        manifest_path=gguf_result.manifest_path,
        artifacts=list(gguf_result.artifacts),
        banner_lines=probe_banner,
        extras={
            "path": "vl-gguf",
            "gguf_path": gguf_result.gguf_path,
            "modelfile_path": gguf_result.modelfile_path,
            "quant": gguf_result.quant,
            "llama_cpp_tag": gguf_result.llama_cpp_tag,
            "mmproj_path": gguf_result.mmproj_path,
        },
    )


def dispatch_audio_export(
    *,
    store: StorePath,
    spec: BaseModelSpec,
    adapter_name: str | None,
    quant: str | None,
    merged: bool,
    adapter_mix_raw: str | None,
) -> DispatchResult:
    """Route an audio-language spec through the HF-snapshot export path.

    Parallel to :func:`dispatch_vl_export` but simpler: llama.cpp has
    no audio-arch roadmap at our pinned tag, so there's nothing to
    probe — always emit an HF-snapshot.
    """
    from dlm.export.audio_snapshot import run_audio_snapshot_export

    banner: list[str] = [
        f"[yellow]export:[/yellow] base {spec.key!r} is audio-language; "
        "emitting HF-snapshot (GGUF not supported for audio archs)."
    ]
    if quant is not None or merged or adapter_mix_raw is not None:
        banner.append(
            "[yellow]export:[/yellow] ignoring GGUF-only flags "
            "(--quant / --merged / --adapter-mix) — they're not applicable "
            "to the HF-snapshot path."
        )

    processor = _load_processor_or_raise(spec)
    result = run_audio_snapshot_export(
        store,
        spec,
        adapter_name=adapter_name,
        processor=processor,
    )

    banner.append(
        f"[green]export:[/green] HF audio snapshot written to {result.export_dir}\n"
        f"  manifest: {result.manifest_path.name}\n"
        f"  adapter:  {result.adapter_dir}\n"
        f"  artifacts: {len(result.artifacts)} file(s)"
    )
    return DispatchResult(
        export_dir=result.export_dir,
        manifest_path=result.manifest_path,
        artifacts=list(result.artifacts),
        banner_lines=banner,
        extras={"path": "audio-snapshot", "adapter_dir": result.adapter_dir},
    )


def _prepend_banner(lines: list[str], result: DispatchResult) -> DispatchResult:
    """Return a new DispatchResult with `lines` prepended to banner_lines."""
    return DispatchResult(
        export_dir=result.export_dir,
        manifest_path=result.manifest_path,
        artifacts=result.artifacts,
        banner_lines=[*lines, *result.banner_lines],
        extras=result.extras,
    )
