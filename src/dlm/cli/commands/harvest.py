"""`dlm harvest` — adversarial replay: harvest failing sway probes into a .dlm."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def harvest_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to harvest into.")],
    sway_json: Annotated[
        Path | None,
        typer.Option(
            "--sway-json",
            help="Path to a sway JSON report. Required unless --revert is set.",
        ),
    ] = None,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Write harvested sections to the .dlm. Default is dry-run (review only).",
        ),
    ] = False,
    tag: Annotated[
        str,
        typer.Option(
            "--tag",
            help="Prefix for the synthesized section's harvest_source metadata.",
        ),
    ] = "auto-harvest",
    min_confidence: Annotated[
        float,
        typer.Option(
            "--min-confidence",
            help="Drop candidates whose sway evidence.confidence is below this.",
            min=0.0,
            max=1.0,
        ),
    ] = 0.0,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict/--lax",
            help=(
                "Strict (default): refuse if any failing probe lacks a "
                "reference. Lax: log a warning and skip those probes."
            ),
        ),
    ] = True,
    revert: Annotated[
        bool,
        typer.Option(
            "--revert",
            help=(
                "Strip every auto-harvested section from the document. "
                "Mutually exclusive with --sway-json / --apply."
            ),
        ),
    ] = False,
) -> None:
    """Adversarial replay: harvest failing sway probes back into the .dlm.

    Default mode is `--dry-run`-style preview; pass `--apply` to write.
    """
    from rich.console import Console

    from dlm.doc.errors import DlmParseError
    from dlm.doc.parser import parse_file
    from dlm.harvest import (
        HarvestError,
        MalformedSwayReportError,
        NoReferenceError,
        apply_plan,
        build_plan,
        read_sway_report,
        render_plan,
        revert_all_auto_harvests,
    )

    console = Console(stderr=True)
    out_console = Console()

    if revert and (sway_json is not None or apply):
        console.print(
            "[red]harvest:[/red] --revert is mutually exclusive with --sway-json / --apply"
        )
        raise typer.Exit(code=1)
    if not revert and sway_json is None:
        console.print(
            "[red]harvest:[/red] --sway-json is required (or pass --revert "
            "to strip auto-harvested sections)"
        )
        raise typer.Exit(code=1)

    try:
        parsed = parse_file(path)
    except (DlmParseError, OSError) as exc:
        console.print(f"[red]harvest:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if revert:
        summary = revert_all_auto_harvests(parsed, target=path)
        out_console.print(
            f"[green]harvest:[/green] stripped {len(summary.added_section_ids)} "
            f"auto-harvested section(s) from {path} (all harvest runs, not just last)"
        )
        return

    assert sway_json is not None  # narrowed by the check above
    try:
        candidates = read_sway_report(
            sway_json,
            strict=strict,
            min_confidence=min_confidence,
        )
    except MalformedSwayReportError as exc:
        console.print(f"[red]harvest:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except NoReferenceError as exc:
        console.print(f"[red]harvest:[/red] {exc}")
        console.print("  Pass [bold]--lax[/bold] to skip probes without references instead.")
        raise typer.Exit(code=1) from exc
    except HarvestError as exc:
        console.print(f"[red]harvest:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    plan = build_plan(parsed, candidates, tag=tag)
    out_console.print(render_plan(plan))

    if not plan.additions:
        out_console.print(
            "\n[yellow]no candidates to harvest[/yellow] — either the sway "
            "report had no failing probes with references, or all matched "
            "sections already exist in the document."
        )
        raise typer.Exit(code=2)

    if not apply:
        out_console.print("\n[dim]dry-run — re-run with [bold]--apply[/bold] to write.[/dim]")
        return

    summary = apply_plan(parsed, plan, target=path)
    out_console.print(
        f"\n[green]harvest:[/green] wrote {summary.added} section(s) to {path} "
        f"({summary.skipped} skipped)"
    )
