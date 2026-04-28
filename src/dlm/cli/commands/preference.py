"""`dlm preference` — mine / apply / revert / list preference sections."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from dlm.cli.commands._shared import _previously_accepted


def preference_mine_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to mine preferences from.")],
    samples: Annotated[
        int,
        typer.Option("--samples", help="Candidate responses to sample per prompt.", min=2),
    ] = 4,
    judge: Annotated[
        str,
        typer.Option(
            "--judge",
            help="Judge selector: sway, hf:<model>, or cli:<cmd>.",
        ),
    ] = "sway",
    threshold: Annotated[
        float | None,
        typer.Option(
            "--threshold",
            help="Minimum chosen-vs-rejected score margin. Defaults to the judge's native threshold.",
            min=0.0,
        ),
    ] = None,
    max_pairs: Annotated[
        int | None,
        typer.Option(
            "--max-pairs",
            help="Maximum mined preference pairs to keep from this run.",
            min=1,
        ),
    ] = None,
    temp: Annotated[
        float,
        typer.Option("--temp", help="Sampling temperature for candidate generation.", min=0.0),
    ] = 0.7,
    top_p: Annotated[
        float | None,
        typer.Option(
            "--top-p",
            help="Optional nucleus-sampling cutoff for candidate generation.",
            min=0.0,
            max=1.0,
        ),
    ] = None,
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help="Generation backend: auto, pytorch, or mlx.",
        ),
    ] = "auto",
    adapter: Annotated[
        str | None,
        typer.Option(
            "--adapter",
            help=(
                "Named adapter to mine from on multi-adapter documents. "
                "Required there; invalid on single-adapter documents."
            ),
        ),
    ] = None,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help=(
                "Write mined preference sections directly to the .dlm. "
                "Default stages them for `dlm preference apply`."
            ),
        ),
    ] = False,
) -> None:
    """Sample + stage auto-mined preference sections from the current adapter."""
    from rich.console import Console

    from dlm.base_models import GatedModelError
    from dlm.base_models import resolve as resolve_base_model
    from dlm.doc.errors import DlmParseError
    from dlm.doc.parser import parse_file
    from dlm.hardware import doctor
    from dlm.inference import AdapterNotFoundError
    from dlm.inference.backends import (
        UnsupportedBackendError,
        build_backend,
        select_backend,
    )
    from dlm.metrics import MetricsRecorder, PreferenceMineEvent
    from dlm.metrics.events import PreferenceMineWriteMode
    from dlm.modality import modality_for
    from dlm.preference import (
        InvalidJudgeSpecError,
        JudgeUnavailableError,
        build_apply_plan,
        build_judge,
        build_mine_plan,
        render_apply_plan,
        render_mine_plan,
    )
    from dlm.preference.apply import apply_plan as apply_preference_plan
    from dlm.preference.pending import clear_pending_plan, save_pending_plan
    from dlm.store.paths import for_dlm

    console = Console(stderr=True)
    out_console = Console()

    if backend not in ("auto", "pytorch", "mlx"):
        console.print(
            f"[red]preference:[/red] --backend must be `auto`, `pytorch`, or `mlx` (got {backend!r})."
        )
        raise typer.Exit(code=2)

    try:
        parsed = parse_file(path)
    except (DlmParseError, OSError) as exc:
        console.print(f"[red]preference:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    adapters_declared = parsed.frontmatter.training.adapters
    if adapter is not None:
        if adapters_declared is None:
            console.print(
                "[red]preference:[/red] --adapter is only valid on multi-adapter "
                "documents (this doc does not declare `training.adapters`)."
            )
            raise typer.Exit(code=2)
        if adapter not in adapters_declared:
            declared = sorted(adapters_declared)
            console.print(
                f"[red]preference:[/red] --adapter {adapter!r} is not declared "
                f"(declared: {declared})."
            )
            raise typer.Exit(code=2)
    elif adapters_declared is not None:
        console.print(
            "[red]preference:[/red] multi-adapter documents require --adapter "
            "so mining knows which adapter to sample."
        )
        raise typer.Exit(code=2)

    judge_kind = judge.split(":", 1)[0].strip()
    if adapter is not None and judge_kind == "sway":
        console.print(
            "[red]preference:[/red] --judge sway is not yet wired for named adapters; "
            "use `hf:<model>` or `cli:<cmd>` for multi-adapter mining."
        )
        raise typer.Exit(code=2)

    store = for_dlm(parsed.frontmatter.dlm_id)
    run_id = _latest_training_run_id(store)
    if run_id is None:
        console.print(
            "[red]preference:[/red] mining requires a prior training run (run `dlm train` first)."
        )
        raise typer.Exit(code=1)

    already_accepted = _previously_accepted(store.manifest)
    try:
        spec = resolve_base_model(parsed.frontmatter.base_model, accept_license=already_accepted)
    except GatedModelError as exc:
        console.print(
            f"[red]license:[/red] base {parsed.frontmatter.base_model!r} is gated and has "
            "no recorded acceptance in this store; run `dlm train --i-accept-license` first."
        )
        raise typer.Exit(code=1) from exc

    dispatch = modality_for(spec)
    if dispatch.accepts_images or dispatch.accepts_audio:
        console.print(
            f"[red]preference:[/red] preference mining currently supports text bases only; "
            f"base {spec.key!r} is modality='{spec.modality}'."
        )
        raise typer.Exit(code=2)

    caps = doctor().capabilities
    try:
        backend_name = select_backend(backend, caps)  # type: ignore[arg-type]
    except UnsupportedBackendError as exc:
        console.print(f"[red]preference:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    backend_obj = build_backend(backend_name, caps)

    try:
        backend_obj.load(spec, store, adapter_name=adapter)
    except AdapterNotFoundError as exc:
        console.print(f"[red]preference:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        judge_obj = build_judge(judge, dlm_path=path)
        plan = build_mine_plan(
            parsed,
            backend_obj,
            judge_obj,
            mined_run_id=run_id,
            samples=samples,
            max_pairs=max_pairs,
            threshold=threshold,
            temperature=temp,
            top_p=top_p,
        )
    except InvalidJudgeSpecError as exc:
        console.print(f"[red]preference:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    except JudgeUnavailableError as exc:
        console.print(f"[red]preference:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print(f"[red]preference:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    finally:
        backend_obj.unload()

    recorder = MetricsRecorder(store.root)

    def _record_preference_mine(write_mode: PreferenceMineWriteMode) -> None:
        recorder.record_preference_mine(
            PreferenceMineEvent(
                run_id=run_id,
                judge_name=judge_obj.name,
                sample_count=samples,
                mined_pairs=len(plan.additions),
                skipped_prompts=len(plan.skipped),
                write_mode=write_mode,
            )
        )

    out_console.print(render_mine_plan(plan))

    if not plan.additions:
        clear_pending_plan(store)
        _record_preference_mine("empty")
        out_console.print(
            "\n[yellow]no candidates to mine[/yellow] — either instruction prompts "
            "did not yield a confident pair, or the matching preference sections "
            "already exist in the document."
        )
        raise typer.Exit(code=2)

    sections = [addition.section for addition in plan.additions]

    if apply:
        apply_plan = build_apply_plan(parsed, sections)
        out_console.print("")
        out_console.print(render_apply_plan(apply_plan))
        summary = apply_preference_plan(parsed, apply_plan, target=path)
        clear_pending_plan(store)
        _record_preference_mine("applied")
        out_console.print(
            f"\n[green]preference:[/green] wrote {summary.added} section(s) to {path} "
            f"({summary.skipped} skipped)"
        )
        return

    pending = save_pending_plan(store, source_path=path.resolve(), sections=sections)
    _record_preference_mine("staged")
    out_console.print(
        f"\n[green]preference:[/green] staged {len(pending.sections)} mined preference "
        f"section(s). Run [bold]dlm preference apply {path}[/bold] to write them."
    )


def preference_apply_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to apply staged preferences into.")],
) -> None:
    """Write the staged preference-mine plan into the `.dlm`."""
    from rich.console import Console

    from dlm.doc.errors import DlmParseError
    from dlm.doc.parser import parse_file
    from dlm.preference import build_apply_plan, render_apply_plan
    from dlm.preference.apply import apply_plan as apply_preference_plan
    from dlm.preference.pending import (
        PendingPreferencePlanError,
        clear_pending_plan,
        load_pending_plan,
    )
    from dlm.store.paths import for_dlm

    console = Console(stderr=True)
    out_console = Console()

    try:
        parsed = parse_file(path)
    except (DlmParseError, OSError) as exc:
        console.print(f"[red]preference:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    store = for_dlm(parsed.frontmatter.dlm_id)
    try:
        pending = load_pending_plan(store)
    except PendingPreferencePlanError as exc:
        console.print(f"[red]preference:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if pending is None:
        console.print(
            "[red]preference:[/red] no staged mined preferences found; "
            "run `dlm preference mine` first."
        )
        raise typer.Exit(code=1)

    plan = build_apply_plan(parsed, list(pending.sections))
    out_console.print(render_apply_plan(plan))

    if not plan.additions:
        clear_pending_plan(store)
        out_console.print(
            "\n[yellow]no staged preferences to write[/yellow] — the pending plan was "
            "already present in the document."
        )
        raise typer.Exit(code=2)

    summary = apply_preference_plan(parsed, plan, target=path)
    clear_pending_plan(store)
    out_console.print(
        f"\n[green]preference:[/green] wrote {summary.added} section(s) to {path} "
        f"({summary.skipped} skipped)"
    )


def preference_revert_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to strip auto-mined preferences from.")],
) -> None:
    """Remove every `auto_mined: true` preference section from the `.dlm`."""
    from rich.console import Console

    from dlm.doc.errors import DlmParseError
    from dlm.doc.parser import parse_file
    from dlm.preference import revert_all_auto_mined

    console = Console(stderr=True)
    out_console = Console()

    try:
        parsed = parse_file(path)
    except (DlmParseError, OSError) as exc:
        console.print(f"[red]preference:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    summary = revert_all_auto_mined(parsed, target=path)
    out_console.print(
        f"[green]preference:[/green] stripped {len(summary.added_section_ids)} "
        f"auto-mined preference section(s) from {path}"
    )


def preference_list_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file whose auto-mined preferences we list.")],
) -> None:
    """List applied + staged auto-mined preference sections."""
    from rich.console import Console

    from dlm.doc.errors import DlmParseError
    from dlm.doc.parser import parse_file
    from dlm.doc.sections import SectionType
    from dlm.preference.pending import PendingPreferencePlanError, load_pending_plan
    from dlm.store.paths import for_dlm

    console = Console(stderr=True)
    out_console = Console()

    try:
        parsed = parse_file(path)
    except (DlmParseError, OSError) as exc:
        console.print(f"[red]preference:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    store = for_dlm(parsed.frontmatter.dlm_id)
    try:
        pending = load_pending_plan(store)
    except PendingPreferencePlanError as exc:
        console.print(f"[red]preference:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    applied = [
        section
        for section in parsed.sections
        if section.type is SectionType.PREFERENCE and section.auto_mined
    ]

    out_console.print(f"[bold]{path}[/bold]")
    out_console.print(f"  applied auto-mined: {len(applied)}")
    out_console.print(f"  staged pending:     {len(pending.sections) if pending else 0}")

    if not applied and pending is None:
        out_console.print("  [dim]no auto-mined preference sections yet[/dim]")
        return

    if applied:
        out_console.print("\n[bold]Applied[/bold]")
        for section in applied:
            prompt = _preference_prompt_summary(section.content, section_id=section.section_id)
            judge_name = section.judge_name or "unknown"
            run_id = section.mined_run_id if section.mined_run_id is not None else "?"
            out_console.print(
                f"  - {section.section_id}  judge={judge_name}  run={run_id}  prompt={prompt}"
            )

    if pending is not None:
        out_console.print("\n[bold]Pending[/bold]")
        for section in pending.sections:
            prompt = _preference_prompt_summary(section.content, section_id=section.section_id)
            judge_name = section.judge_name or "unknown"
            run_id = section.mined_run_id if section.mined_run_id is not None else "?"
            out_console.print(
                f"  - {section.section_id}  judge={judge_name}  run={run_id}  prompt={prompt}"
            )


def _latest_training_run_id(store: object) -> int | None:
    """Most recent run id from metrics DB or manifest."""
    from dlm.metrics.queries import latest_run_id
    from dlm.store.errors import ManifestCorruptError
    from dlm.store.manifest import load_manifest
    from dlm.store.paths import StorePath

    assert isinstance(store, StorePath)

    metrics_run_id = latest_run_id(store.root)
    if metrics_run_id is not None:
        return metrics_run_id
    if not store.manifest.exists():
        return None
    try:
        manifest = load_manifest(store.manifest)
    except (ManifestCorruptError, OSError):
        return None
    if not manifest.training_runs:
        return None
    return max(run.run_id for run in manifest.training_runs)


def _preference_prompt_summary(content: str, *, section_id: str) -> str:
    """Best-effort prompt summary for `preference list`."""
    from dlm.data.errors import PreferenceParseError
    from dlm.data.preference_parser import parse_preference_body

    try:
        triples = parse_preference_body(content, section_id=section_id)
    except PreferenceParseError:
        return "<unparseable>"
    if not triples:
        return "<empty>"
    prompt = triples[0].prompt.splitlines()[0].strip()
    return prompt or "<blank>"
