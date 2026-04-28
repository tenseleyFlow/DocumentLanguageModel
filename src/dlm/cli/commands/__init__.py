"""Subcommand stubs for the v1.0 CLI surface.

Every stub raises `NotImplementedError` with the sprint number that will
implement it. This makes `dlm --help` self-documenting about project
progress. Arguments are accepted so `--help` renders the real eventual
surface; they're unused until each subcommand's owning sprint lands,
which is why `src/dlm/cli/commands.py` has a ruff per-file-ignore for
`ARG001` in `pyproject.toml`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer

from dlm.cli.commands._shared import _human_size as _human_size
from dlm.cli.commands._shared import _previously_accepted as _previously_accepted
from dlm.cli.commands.cache import _parse_duration as _parse_duration
from dlm.cli.commands.cache import cache_clear_cmd as cache_clear_cmd
from dlm.cli.commands.cache import cache_prune_cmd as cache_prune_cmd
from dlm.cli.commands.cache import cache_show_cmd as cache_show_cmd
from dlm.cli.commands.doctor import doctor_cmd as doctor_cmd
from dlm.cli.commands.harvest import harvest_cmd as harvest_cmd
from dlm.cli.commands.init import _prompt_accept_license as _prompt_accept_license
from dlm.cli.commands.init import init_cmd as init_cmd
from dlm.cli.commands.metrics import metrics_cmd as metrics_cmd
from dlm.cli.commands.metrics import metrics_watch_cmd as metrics_watch_cmd
from dlm.cli.commands.migrate import migrate_cmd as migrate_cmd
from dlm.cli.commands.pack import pack_cmd as pack_cmd
from dlm.cli.commands.preference import preference_apply_cmd as preference_apply_cmd
from dlm.cli.commands.preference import preference_list_cmd as preference_list_cmd
from dlm.cli.commands.preference import preference_mine_cmd as preference_mine_cmd
from dlm.cli.commands.preference import preference_revert_cmd as preference_revert_cmd
from dlm.cli.commands.prompt import _dispatch_audio_prompt as _dispatch_audio_prompt
from dlm.cli.commands.prompt import _dispatch_vl_prompt as _dispatch_vl_prompt
from dlm.cli.commands.prompt import prompt_cmd as prompt_cmd
from dlm.cli.commands.pull import pull_cmd as pull_cmd
from dlm.cli.commands.push import push_cmd as push_cmd
from dlm.cli.commands.repl import repl_cmd as repl_cmd
from dlm.cli.commands.serve import serve_cmd as serve_cmd
from dlm.cli.commands.show import show_cmd as show_cmd
from dlm.cli.commands.synth import synth_instructions_cmd as synth_instructions_cmd
from dlm.cli.commands.synth import synth_list_cmd as synth_list_cmd
from dlm.cli.commands.synth import synth_revert_cmd as synth_revert_cmd
from dlm.cli.commands.templates import templates_list_cmd as templates_list_cmd
from dlm.cli.commands.train import _maybe_dispatch_multi_gpu as _maybe_dispatch_multi_gpu
from dlm.cli.commands.train import _strip_gpus_from_argv as _strip_gpus_from_argv
from dlm.cli.commands.train import train_cmd as train_cmd
from dlm.cli.commands.unpack import unpack_cmd as unpack_cmd
from dlm.cli.commands.verify import verify_cmd as verify_cmd


def _stub(sprint: str, subject: str) -> None:
    """Raise a clear unimplemented error pointing to the owning sprint."""
    raise NotImplementedError(
        f"`{subject}` is not implemented yet (owned by Sprint {sprint}).",
    )





def export_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to export.")],
    target: Annotated[
        str,
        typer.Option(
            "--target",
            help="Export destination. Currently supported: ollama, llama-server, vllm, mlx-serve.",
        ),
    ] = "ollama",
    quant: Annotated[
        str | None,
        typer.Option("--quant", help="GGUF quant level (defaults to frontmatter)."),
    ] = None,
    merged: Annotated[
        bool,
        typer.Option("--merged", help="Merge the adapter into the base before export."),
    ] = False,
    dequantize: Annotated[
        bool,
        typer.Option(
            "--dequantize",
            help="Dequantize a QLoRA base to fp16 before merging.",
        ),
    ] = False,
    name: Annotated[str | None, typer.Option("--name", help="Ollama model name.")] = None,
    no_template: Annotated[
        bool,
        typer.Option("--no-template", help="Skip writing TEMPLATE into the Modelfile."),
    ] = False,
    no_smoke: Annotated[
        bool,
        typer.Option("--no-smoke", help="Register the export but skip the smoke prompt."),
    ] = False,
    no_imatrix: Annotated[
        bool,
        typer.Option(
            "--no-imatrix",
            help=(
                "Skip importance-matrix calibration. Default uses the "
                "replay corpus to calibrate k-quant quantization."
            ),
        ),
    ] = False,
    draft: Annotated[
        str | None,
        typer.Option(
            "--draft",
            help=(
                "Speculative-decoding draft model Ollama tag "
                "(e.g. qwen2.5:0.5b). Default uses the registered pair "
                "for this base; override here to pick a custom draft."
            ),
        ),
    ] = None,
    no_draft: Annotated[
        bool,
        typer.Option(
            "--no-draft",
            help="Suppress PARAMETER draft_model emission even when a pair is registered.",
        ),
    ] = False,
    skip_ollama: Annotated[
        bool,
        typer.Option(
            "--skip-ollama",
            help="Emit GGUFs + manifest only; do not touch the Ollama binary.",
        ),
    ] = False,
    adapter: Annotated[
        str | None,
        typer.Option(
            "--adapter",
            help=(
                "Named adapter to export. Required on multi-adapter "
                "documents; rejected on single-adapter documents."
            ),
        ),
    ] = None,
    adapter_mix: Annotated[
        str | None,
        typer.Option(
            "--adapter-mix",
            help=(
                "Weighted composition of named adapters, e.g. "
                "`knowledge:1.0,tone:0.5`. Mutually exclusive with --adapter. "
                "Multi-adapter docs only. LoRA-only; QLoRA requires "
                "--dequantize."
            ),
        ),
    ] = None,
    adapter_mix_method: Annotated[
        str,
        typer.Option(
            "--adapter-mix-method",
            help=(
                "PEFT combination strategy for --adapter-mix. `linear` "
                "(default) sums LoRA deltas; `svd` recomposes via SVD "
                "(higher fidelity, heavier compute). Only meaningful "
                "with --adapter-mix."
            ),
        ),
    ] = "linear",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Log each subprocess command as it launches."),
    ] = False,
    emit_sway_json: Annotated[
        bool,
        typer.Option(
            "--emit-sway-json",
            help=(
                "After the export, also write a ready-to-run sway.yaml "
                "(via dlm-sway autogen) into the export dir. Requires the "
                "[sway] extra: pip install 'dlm[sway]'."
            ),
        ),
    ] = False,
) -> None:
    """Export the adapter to a runtime target."""

    from rich.console import Console

    from dlm.base_models import GatedModelError, download_spec
    from dlm.base_models import resolve as resolve_base_model
    from dlm.doc.parser import parse_file
    from dlm.export import (
        ExportError,
        PreflightError,
        SubprocessError,
        UnknownExportTargetError,
        UnsafeMergeError,
        VendoringError,
        resolve_export_plan,
        run_export,
    )
    from dlm.export.ollama import (
        OllamaBinaryNotFoundError,
        OllamaCreateError,
        OllamaError,
        OllamaSmokeError,
        OllamaVersionError,
    )
    from dlm.export.quantize import run_checked
    from dlm.export.targets import (
        finalize_mlx_serve_export,
        finalize_vllm_export,
        prepare_llama_server_export,
        prepare_mlx_serve_export,
        prepare_vllm_export,
        resolve_target,
    )
    from dlm.store.paths import for_dlm

    console = Console(stderr=True)

    if draft is not None and no_draft:
        console.print("[red]error:[/red] --draft and --no-draft are mutually exclusive; pick one.")
        raise typer.Exit(code=2)
    if adapter is not None and adapter_mix is not None:
        console.print(
            "[red]export:[/red] --adapter and --adapter-mix are mutually exclusive; pick one."
        )
        raise typer.Exit(code=2)
    try:
        resolved_target = resolve_target(target)
    except UnknownExportTargetError as exc:
        console.print(f"[red]export:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    parsed = parse_file(path)
    adapters_declared = parsed.frontmatter.training.adapters
    if adapter is not None:
        if adapters_declared is None:
            console.print(
                "[red]export:[/red] --adapter is only valid on multi-adapter "
                "documents (this doc does not declare `training.adapters`)."
            )
            raise typer.Exit(code=2)
        if adapter not in adapters_declared:
            declared = sorted(adapters_declared)
            console.print(
                f"[red]export:[/red] --adapter {adapter!r} is not declared (declared: {declared})."
            )
            raise typer.Exit(code=2)

    mix_entries: list[tuple[str, float]] | None = None
    if adapter_mix is not None:
        from dlm.export.weighted_merge import (
            InvalidMixSpecError,
            parse_mix_spec,
            validate_mix_against_declared,
        )

        if adapters_declared is None:
            console.print(
                "[red]export:[/red] --adapter-mix is only valid on multi-adapter "
                "documents (this doc does not declare `training.adapters`)."
            )
            raise typer.Exit(code=2)
        if adapter_mix_method not in ("linear", "svd"):
            console.print(
                f"[red]export:[/red] --adapter-mix-method must be "
                f"`linear` or `svd`, got {adapter_mix_method!r}."
            )
            raise typer.Exit(code=2)
        try:
            entries = parse_mix_spec(adapter_mix)
            validate_mix_against_declared(entries, set(adapters_declared))
        except InvalidMixSpecError as exc:
            console.print(f"[red]export:[/red] {exc}")
            raise typer.Exit(code=2) from exc
        mix_entries = [(e.name, e.weight) for e in entries]

    store = for_dlm(parsed.frontmatter.dlm_id)

    # Gate-driven static mix: when the doc has an enabled gate and the
    # user didn't pass --adapter-mix / --adapter, freeze the learned
    # gate to per-adapter weights for the GGUF export path. Dynamic
    # routing only lives in the `dlm prompt` flow; the runtime can't
    # evaluate the torch gate, so we substitute the prior here. A CLI
    # --adapter-mix wins — users who know what they want get full
    # control.
    if mix_entries is None and adapter is None:
        from dlm.export.gate_fallback import resolve_and_announce

        resolution = resolve_and_announce(store, parsed)
        if resolution.entries is not None:
            mix_entries = resolution.entries
            for line in resolution.banner_lines:
                console.print(line)

    already_accepted = _previously_accepted(store.manifest)
    try:
        spec = resolve_base_model(parsed.frontmatter.base_model, accept_license=already_accepted)
    except GatedModelError as exc:
        console.print(f"[red]license:[/red] base model {parsed.frontmatter.base_model!r} is gated.")
        if exc.license_url:
            console.print(f"  review the license at: {exc.license_url}")
        console.print("  accept via `dlm train --i-accept-license` before exporting.")
        raise typer.Exit(code=1) from exc

    # Audio bases take HF-snapshot unconditionally — llama.cpp has no
    # audio-arch roadmap at our pinned tag — so branch early without
    # resolving a GGUF plan.
    from dlm.modality import modality_for

    export_dispatch = modality_for(spec)
    if resolved_target.name == "vllm" and export_dispatch.accepts_audio:
        console.print(
            "[red]export:[/red] --target vllm is not wired for audio-language "
            "documents yet; the current vllm export path only supports text bases."
        )
        raise typer.Exit(code=2)
    if resolved_target.name == "mlx-serve" and export_dispatch.accepts_audio:
        console.print(
            "[red]export:[/red] --target mlx-serve is not wired for audio-language "
            "documents yet; the current mlx-serve export path only supports text bases."
        )
        raise typer.Exit(code=2)
    if export_dispatch.accepts_audio:
        try:
            dispatch_result = export_dispatch.dispatch_export(
                store=store,
                spec=spec,
                adapter_name=adapter,
                quant=quant,
                merged=merged,
                adapter_mix_raw=adapter_mix,
            )
        except ExportError as exc:
            console.print(f"[red]export:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        assert dispatch_result is not None  # audio modality always returns a result
        for line in dispatch_result.banner_lines:
            console.print(line)
        return

    try:
        plan = resolve_export_plan(
            cli_quant=quant,
            cli_merged=merged,
            cli_dequantize=dequantize,
            cli_no_template=no_template,
            cli_ollama_name=name,
            cli_no_imatrix=no_imatrix,
            frontmatter_default_quant=parsed.frontmatter.export.default_quant,
        )
    except ValueError as exc:
        console.print(f"[red]export:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    store.ensure_layout()

    # VL bases: arch-probe + try single-file GGUF on SUPPORTED (with
    # fallback to HF-snapshot on refusal or subprocess failure). A
    # missing local base snapshot should not hard-fail the whole
    # export — the dispatcher can still emit the HF-snapshot path
    # without GGUF context.
    if resolved_target.name == "vllm" and export_dispatch.accepts_images:
        console.print(
            "[red]export:[/red] --target vllm is not wired for vision-language "
            "documents yet; the current vllm export path only supports text bases."
        )
        raise typer.Exit(code=2)
    if resolved_target.name == "mlx-serve" and export_dispatch.accepts_images:
        console.print(
            "[red]export:[/red] --target mlx-serve is not wired for vision-language "
            "documents yet; the current mlx-serve export path only supports text bases."
        )
        raise typer.Exit(code=2)
    if export_dispatch.accepts_images:
        gguf_emission_context = None
        try:
            cached_vl = download_spec(spec, local_files_only=True)
        except RuntimeError as exc:
            _ = exc
        else:
            gguf_emission_context = {
                "plan": plan,
                "cached_base_dir": cached_vl.path,
                "source_dlm_path": path.resolve(),
                "training_sequence_len": parsed.frontmatter.training.sequence_len,
                "dlm_version": f"v{parsed.frontmatter.dlm_version}",
            }
        try:
            dispatch_result = export_dispatch.dispatch_export(
                store=store,
                spec=spec,
                adapter_name=adapter,
                quant=quant,
                merged=merged,
                adapter_mix_raw=adapter_mix,
                gguf_emission_context=gguf_emission_context,
            )
        except ExportError as exc:
            console.print(f"[red]export:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        assert dispatch_result is not None  # VL modality always returns a result
        for line in dispatch_result.banner_lines:
            console.print(line)
        return

    try:
        cached = download_spec(spec, local_files_only=True)
    except RuntimeError as exc:
        console.print(
            f"[red]export:[/red] base model not in local cache — run `dlm train` first.\n  {exc}"
        )
        raise typer.Exit(code=1) from exc

    def _verbose_runner(cmd: Sequence[str]) -> object:
        console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
        return run_checked(cmd)

    adapter_path_override = None
    if mix_entries is not None:  # pragma: no cover - heavy path
        # Build the weighted-merged adapter into an ephemeral dir,
        # then feed the path to run_export as an override. The tmp
        # dir lives under the store's cache/ so it cleans up with
        # the rest of the store on `dlm pack`.
        from dlm.export.weighted_merge import MixEntry, build_and_stage

        entries_typed = [MixEntry(name=n, weight=w) for (n, w) in mix_entries]
        adapter_path_override = build_and_stage(
            store=store,
            spec=spec,
            cached_base_dir=cached.path,
            entries=entries_typed,
            combination_type=adapter_mix_method,  # type: ignore[arg-type]
        )

    if resolved_target.name == "vllm":
        ignored_flags: list[str] = []
        if quant is not None:
            ignored_flags.append("--quant")
        if merged:
            ignored_flags.append("--merged")
        if dequantize:
            ignored_flags.append("--dequantize")
        if no_template:
            ignored_flags.append("--no-template")
        if skip_ollama:
            ignored_flags.append("--skip-ollama")
        if no_imatrix:
            ignored_flags.append("--no-imatrix")
        if draft is not None:
            ignored_flags.append("--draft")
        if no_draft:
            ignored_flags.append("--no-draft")
        if ignored_flags:
            console.print(
                "[yellow]export:[/yellow] ignoring flags not applicable to "
                f"`--target vllm`: {', '.join(ignored_flags)}"
            )

        declared_adapter_names = tuple(adapters_declared.keys()) if adapters_declared else None
        try:
            vllm_result = prepare_vllm_export(
                store=store,
                spec=spec,
                served_model_name=name or f"dlm-{parsed.frontmatter.dlm_id.lower()}",
                training_sequence_len=parsed.frontmatter.training.sequence_len,
                adapter_name=adapter,
                adapter_path_override=adapter_path_override,
                declared_adapter_names=declared_adapter_names,
            )
        except ExportError as exc:
            console.print(f"[red]export:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        vllm_smoke = None if no_smoke else resolved_target.smoke_test(vllm_result)
        if vllm_smoke is not None and not vllm_smoke.ok:
            console.print(
                f"[red]smoke:[/red] {vllm_smoke.detail}\n"
                "  re-run with `--no-smoke` to skip the smoke test."
            )
            raise typer.Exit(code=1)

        manifest_path = finalize_vllm_export(
            store=store,
            spec=spec,
            prepared=vllm_result,
            smoke_output_first_line=None if vllm_smoke is None else vllm_smoke.detail,
            adapter_name=adapter,
            adapter_mix=mix_entries,
        )
        console.print(f"[green]exported:[/green] {vllm_result.export_dir}")
        console.print("target:  vllm")
        assert vllm_result.launch_script_path is not None
        assert vllm_result.config_path is not None
        console.print(f"launch:  {vllm_result.launch_script_path.name}")
        console.print(f"config:  {vllm_result.config_path.name}")
        console.print(f"manifest: {manifest_path.name}")
        if vllm_smoke is not None and vllm_smoke.detail:
            console.print(f"smoke:   {vllm_smoke.detail}")
        return

    if resolved_target.name == "mlx-serve":
        mlx_ignored_flags: list[str] = []
        if quant is not None:
            mlx_ignored_flags.append("--quant")
        if merged:
            mlx_ignored_flags.append("--merged")
        if dequantize:
            mlx_ignored_flags.append("--dequantize")
        if name is not None:
            mlx_ignored_flags.append("--name")
        if no_template:
            mlx_ignored_flags.append("--no-template")
        if skip_ollama:
            mlx_ignored_flags.append("--skip-ollama")
        if no_imatrix:
            mlx_ignored_flags.append("--no-imatrix")
        if draft is not None:
            mlx_ignored_flags.append("--draft")
        if no_draft:
            mlx_ignored_flags.append("--no-draft")
        if mlx_ignored_flags:
            console.print(
                "[yellow]export:[/yellow] ignoring flags not applicable to "
                f"`--target mlx-serve`: {', '.join(mlx_ignored_flags)}"
            )

        declared_adapter_names = tuple(adapters_declared.keys()) if adapters_declared else None
        try:
            mlx_serve_result = prepare_mlx_serve_export(
                store=store,
                spec=spec,
                adapter_name=adapter,
                adapter_path_override=adapter_path_override,
                declared_adapter_names=declared_adapter_names,
            )
        except ExportError as exc:
            console.print(f"[red]export:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        mlx_serve_smoke = None if no_smoke else resolved_target.smoke_test(mlx_serve_result)
        if mlx_serve_smoke is not None and not mlx_serve_smoke.ok:
            console.print(
                f"[red]smoke:[/red] {mlx_serve_smoke.detail}\n"
                "  re-run with `--no-smoke` to skip the smoke test."
            )
            raise typer.Exit(code=1)

        manifest_path = finalize_mlx_serve_export(
            store=store,
            spec=spec,
            prepared=mlx_serve_result,
            smoke_output_first_line=None if mlx_serve_smoke is None else mlx_serve_smoke.detail,
            adapter_name=adapter,
            adapter_mix=mix_entries,
        )
        console.print(f"[green]exported:[/green] {mlx_serve_result.export_dir}")
        console.print("target:  mlx-serve")
        assert mlx_serve_result.launch_script_path is not None
        console.print(f"launch:  {mlx_serve_result.launch_script_path.name}")
        console.print(f"manifest: {manifest_path.name}")
        if mlx_serve_smoke is not None and mlx_serve_smoke.detail:
            console.print(f"smoke:   {mlx_serve_smoke.detail}")
        return

    try:
        result = run_export(
            store,
            spec,
            plan,
            target=resolved_target.name,
            cached_base_dir=cached.path,
            subprocess_runner=_verbose_runner if verbose else None,
            skip_ollama=skip_ollama or resolved_target.name != "ollama",
            skip_smoke=no_smoke,
            source_dlm_path=path.resolve(),
            training_sequence_len=parsed.frontmatter.training.sequence_len,
            override_temperature=parsed.frontmatter.export.default_temperature,
            override_top_p=parsed.frontmatter.export.default_top_p,
            draft_override=draft,
            draft_disabled=no_draft,
            adapter_name=adapter,
            adapter_path_override=adapter_path_override,
            adapter_mix=mix_entries,
        )
    except UnsafeMergeError as exc:
        console.print(f"[red]merge:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except VendoringError as exc:
        console.print(
            f"[red]vendor:[/red] {exc}\n"
            "  run `scripts/bump-llama-cpp.sh build` or "
            "`git submodule update --init --recursive`."
        )
        raise typer.Exit(code=1) from exc
    except PreflightError as exc:
        console.print(f"[red]preflight[{exc.probe}]:[/red] {exc.detail}")
        raise typer.Exit(code=1) from exc
    except SubprocessError as exc:
        console.print(f"[red]subprocess:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except OllamaBinaryNotFoundError as exc:
        console.print(
            f"[red]ollama:[/red] {exc}\n"
            "  install from https://ollama.com/download "
            "or re-run with `--skip-ollama`."
        )
        raise typer.Exit(code=1) from exc
    except OllamaVersionError as exc:
        console.print(f"[red]ollama:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except OllamaCreateError as exc:
        console.print(f"[red]ollama create:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except OllamaSmokeError as exc:
        console.print(
            f"[red]smoke:[/red] {exc}\n  re-run with `--no-smoke` to skip the smoke test."
        )
        raise typer.Exit(code=1) from exc
    except OllamaError as exc:
        console.print(f"[red]ollama:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ExportError as exc:
        console.print(f"[red]export:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if resolved_target.name == "llama-server":
        adapter_dir = adapter_path_override
        if adapter_dir is None:
            if adapter is None:
                adapter_dir = store.resolve_current_adapter()
            else:
                adapter_dir = store.resolve_current_adapter_for(adapter)
        assert adapter_dir is not None
        try:
            llama_server_result = prepare_llama_server_export(
                export_dir=result.export_dir,
                manifest_path=result.manifest_path,
                artifacts=result.artifacts,
                adapter_dir=adapter_dir,
                spec=spec,
                training_sequence_len=parsed.frontmatter.training.sequence_len,
            )
        except VendoringError as exc:
            console.print(
                f"[red]vendor:[/red] {exc}\n"
                "  run `scripts/bump-llama-cpp.sh build --with-server` or "
                "`git submodule update --init --recursive`."
            )
            raise typer.Exit(code=1) from exc
        except ExportError as exc:
            console.print(f"[red]export:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        llama_server_smoke = None if no_smoke else resolved_target.smoke_test(llama_server_result)
        if llama_server_smoke is not None and not llama_server_smoke.ok:
            console.print(
                f"[red]smoke:[/red] {llama_server_smoke.detail}\n"
                "  re-run with `--no-smoke` to skip the smoke test."
            )
            raise typer.Exit(code=1)

    cached_tag = " [dim](cached base)[/dim]" if result.cached else ""
    console.print(f"[green]exported:[/green] {result.export_dir}{cached_tag}")
    for artifact in result.artifacts:
        console.print(f"  {artifact.name}")

    # S26 X1 — also emit a sway.yaml next to the GGUF when the user
    # asks for it. Done AFTER the regular export so a sway-side
    # failure can never roll back a working GGUF deployment.
    if emit_sway_json:
        from dlm.export.sway_json import SwayJsonExportError, write_sway_json

        try:
            sway_yaml_path = write_sway_json(path, result.export_dir)
        except SwayJsonExportError as exc:
            console.print(f"[red]sway-json:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        console.print(f"[green]sway.yaml:[/green] {sway_yaml_path}")
        console.print("  next: sway run " + str(sway_yaml_path))
    if resolved_target.name == "llama-server":
        assert llama_server_result.launch_script_path is not None
        assert llama_server_result.config_path is not None
        console.print(f"target:  {result.target}")
        console.print(f"launch:  {llama_server_result.launch_script_path.name}")
        console.print(f"template: {llama_server_result.config_path.name}")
        if llama_server_smoke is not None and llama_server_smoke.detail:
            console.print(f"smoke:   {llama_server_smoke.detail}")
        return
    if result.ollama_name:
        console.print(f"ollama:  {result.ollama_name} (v{result.ollama_version})")
    if result.smoke_output_first_line:
        console.print(f"smoke:   {result.smoke_output_first_line}")


