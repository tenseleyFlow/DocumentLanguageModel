"""`dlm synth` — generate / list / revert auto-synth instruction sections."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, cast

import typer


def synth_instructions_cmd(
    path: Annotated[
        Path, typer.Argument(help=".dlm file to synthesize instruction sections from.")
    ],
    teacher: Annotated[
        str,
        typer.Option(
            "--teacher",
            help=(
                "Teacher selector: self, hf:<model>, openai:<model>, "
                "anthropic:<model>, or vllm-server:<url>."
            ),
        ),
    ] = "self",
    per_section: Annotated[
        int,
        typer.Option(
            "--per-section",
            help="Instruction pairs to generate per prose section.",
            min=1,
        ),
    ] = 3,
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            help="Synthesis strategy: extraction, expansion, or both.",
        ),
    ] = "extraction",
    filter_kind: Annotated[
        str,
        typer.Option(
            "--filter",
            help="Filter pipeline: sway, none, or dedup-only.",
        ),
    ] = "sway",
    threshold: Annotated[
        float | None,
        typer.Option(
            "--threshold",
            help="Optional minimum sway-judge margin when --filter=sway.",
            min=0.0,
        ),
    ] = None,
    max_pairs: Annotated[
        int | None,
        typer.Option(
            "--max-pairs",
            help="Maximum accepted synth pairs to keep from this run.",
            min=1,
        ),
    ] = None,
    max_new_tokens: Annotated[
        int,
        typer.Option(
            "--max-new-tokens",
            help="Maximum new tokens the teacher may emit per prompt.",
            min=1,
        ),
    ] = 512,
    temp: Annotated[
        float,
        typer.Option("--temp", help="Teacher sampling temperature.", min=0.0),
    ] = 0.0,
    top_p: Annotated[
        float | None,
        typer.Option(
            "--top-p",
            help="Optional top-p cutoff for teacher sampling.",
            min=0.0,
            max=1.0,
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Optional teacher sampling seed."),
    ] = None,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Write accepted auto-synth sections directly to the .dlm.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Preview the synth plan without staging or writing anything.",
        ),
    ] = False,
) -> None:
    """Generate, stage, or apply auto-synth instruction sections."""
    from rich.console import Console

    from dlm.doc.errors import DlmParseError
    from dlm.doc.parser import parse_file
    from dlm.preference import JudgeUnavailableError
    from dlm.store.paths import for_dlm
    from dlm.synth import (
        InvalidTeacherSpecError,
        TeacherInvocationError,
        TeacherUnavailableError,
        render_apply_plan,
        render_filter_report,
        render_synth_plan,
    )
    from dlm.synth.dispatch import (
        SynthInstructionsRequest,
        SynthOutcome,
        run_synth_instructions,
    )

    console = Console(stderr=True)
    out_console = Console()

    if strategy not in ("extraction", "expansion", "both"):
        console.print(
            "[red]synth:[/red] --strategy must be one of extraction|expansion|both "
            f"(got {strategy!r})."
        )
        raise typer.Exit(code=2)
    if filter_kind not in ("sway", "none", "dedup-only"):
        console.print(
            f"[red]synth:[/red] --filter must be one of sway|none|dedup-only (got {filter_kind!r})."
        )
        raise typer.Exit(code=2)
    if apply and dry_run:
        console.print("[red]synth:[/red] --apply and --dry-run are mutually exclusive.")
        raise typer.Exit(code=2)
    if threshold is not None and filter_kind != "sway":
        console.print("[red]synth:[/red] --threshold is only valid when --filter is `sway`.")
        raise typer.Exit(code=2)

    try:
        parsed = parse_file(path)
    except (DlmParseError, OSError) as exc:
        console.print(f"[red]synth:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    store = for_dlm(parsed.frontmatter.dlm_id)

    request = SynthInstructionsRequest(
        parsed=parsed,
        target_path=path,
        store=store,
        teacher=teacher,
        per_section=per_section,
        strategy=cast(Literal["extraction", "expansion", "both"], strategy),
        filter_kind=cast(Literal["sway", "none", "dedup-only"], filter_kind),
        threshold=threshold,
        max_pairs=max_pairs,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        top_p=top_p,
        seed=seed,
        apply=apply,
        dry_run=dry_run,
    )

    try:
        result = run_synth_instructions(request)
    except InvalidTeacherSpecError as exc:
        console.print(f"[red]synth:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    except TeacherUnavailableError as exc:
        console.print(f"[red]synth:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except TeacherInvocationError as exc:
        console.print(f"[red]synth:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except JudgeUnavailableError as exc:
        console.print(f"[red]synth:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print(f"[red]synth:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    out_console.print(render_synth_plan(result.plan))
    out_console.print("")
    out_console.print(render_filter_report(result.filtered_plan))

    if result.outcome is SynthOutcome.NO_ADDITIONS:
        out_console.print(
            "\n[yellow]no synth additions accepted[/yellow] — either generation "
            "yielded no valid pairs, dedup removed them, or the filter rejected them."
        )
        raise typer.Exit(code=2)

    if result.outcome is SynthOutcome.APPLIED:
        assert result.apply_plan is not None
        assert result.apply_summary is not None
        out_console.print("")
        out_console.print(render_apply_plan(result.apply_plan))
        out_console.print(
            f"\n[green]synth:[/green] wrote {result.apply_summary.added} section(s) to {path} "
            f"({result.apply_summary.skipped} skipped)"
        )
        return

    if result.outcome is SynthOutcome.DRY_RUN:
        out_console.print("\n[green]synth:[/green] dry-run only — nothing staged.")
        return

    # SynthOutcome.STAGED
    out_console.print(
        f"\n[green]synth:[/green] staged {result.pending_count} auto-synth instruction "
        f"section(s). Run [bold]dlm synth list {path}[/bold] to inspect them."
    )


def synth_revert_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to strip auto-synth instructions from.")],
) -> None:
    """Remove every `auto_synth: true` instruction section from the `.dlm`."""
    from rich.console import Console

    from dlm.doc.errors import DlmParseError
    from dlm.doc.parser import parse_file
    from dlm.synth import revert_all_auto_synth

    console = Console(stderr=True)
    out_console = Console()

    try:
        parsed = parse_file(path)
    except (DlmParseError, OSError) as exc:
        console.print(f"[red]synth:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    summary = revert_all_auto_synth(parsed, target=path)
    out_console.print(
        f"[green]synth:[/green] stripped {len(summary.added_section_ids)} "
        f"auto-synth instruction section(s) from {path}"
    )


def synth_list_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file whose auto-synth instructions we list.")],
) -> None:
    """List applied + staged auto-synth instruction sections."""
    from rich.console import Console

    from dlm.doc.errors import DlmParseError
    from dlm.doc.parser import parse_file
    from dlm.doc.sections import SectionType
    from dlm.store.paths import for_dlm
    from dlm.synth import PendingSynthPlanError, load_pending_plan

    console = Console(stderr=True)
    out_console = Console()

    try:
        parsed = parse_file(path)
    except (DlmParseError, OSError) as exc:
        console.print(f"[red]synth:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    store = for_dlm(parsed.frontmatter.dlm_id)
    try:
        pending = load_pending_plan(store)
    except PendingSynthPlanError as exc:
        console.print(f"[red]synth:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    applied = [
        section
        for section in parsed.sections
        if section.type is SectionType.INSTRUCTION and section.auto_synth
    ]

    out_console.print(f"[bold]{path}[/bold]")
    out_console.print(f"  applied auto-synth: {len(applied)}")
    out_console.print(f"  staged pending:     {len(pending.sections) if pending else 0}")

    if not applied and pending is None:
        out_console.print("  [dim]no auto-synth instruction sections yet[/dim]")
        return

    if applied:
        _render_synth_listing(out_console, "Applied", applied)
    if pending is not None:
        _render_synth_listing(out_console, "Pending", pending.sections)


def _render_synth_listing(
    out_console: object,
    heading: str,
    sections: Sequence[object],
) -> None:
    from collections import Counter

    from rich.console import Console

    from dlm.doc.sections import Section

    assert isinstance(out_console, Console)
    typed_sections = [section for section in sections if isinstance(section, Section)]

    out_console.print(f"\n[bold]{heading}[/bold]")

    teacher_counts = Counter(section.synth_teacher or "unknown" for section in typed_sections)
    strategy_counts = Counter(section.synth_strategy or "unknown" for section in typed_sections)
    source_counts = Counter(section.source_section_id or "unknown" for section in typed_sections)

    out_console.print("  by teacher:")
    for teacher_name in sorted(teacher_counts):
        out_console.print(f"    - {teacher_name}: {teacher_counts[teacher_name]}")

    out_console.print("  by strategy:")
    for strategy_name in sorted(strategy_counts):
        out_console.print(f"    - {strategy_name}: {strategy_counts[strategy_name]}")

    out_console.print("  by source section:")
    for source_id in sorted(source_counts):
        out_console.print(f"    - {source_id}: {source_counts[source_id]}")

    out_console.print("  sections:")
    for section in typed_sections:
        prompt = _synth_prompt_summary(section.content, section_id=section.section_id)
        out_console.print(
            "    - "
            f"{section.section_id}  teacher={section.synth_teacher or 'unknown'}  "
            f"strategy={section.synth_strategy or 'unknown'}  "
            f"source={section.source_section_id or 'unknown'}  "
            f"prompt={prompt}"
        )


def _synth_prompt_summary(content: str, *, section_id: str) -> str:
    """Best-effort prompt summary for `synth list`."""
    from dlm.data.errors import InstructionParseError
    from dlm.data.instruction_parser import parse_instruction_body

    try:
        pairs = parse_instruction_body(content, section_id=section_id)
    except InstructionParseError:
        return "<unparseable>"
    if not pairs:
        return "<empty>"
    prompt = pairs[0].question.splitlines()[0].strip()
    return prompt or "<blank>"
