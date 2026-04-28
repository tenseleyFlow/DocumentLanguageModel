"""`dlm metrics` — query / watch the per-store metrics database."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Annotated

import typer


def metrics_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file whose store we query.")],
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
    csv_out: Annotated[bool, typer.Option("--csv", help="Emit CSV.")] = False,
    run_id: Annotated[
        int | None,
        typer.Option("--run-id", help="Only show this run (drill-down)."),
    ] = None,
    phase: Annotated[
        str | None,
        typer.Option("--phase", help="Filter by phase: sft|dpo|orpo|cpt."),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option(
            "--since",
            help="Time window (e.g. `24h`, `7d`, `30m`). Filters `started_at`.",
        ),
    ] = None,
    limit: Annotated[int, typer.Option("--limit")] = 20,
) -> None:
    """Query the per-store metrics database."""
    import csv
    import json
    import sys

    from rich.console import Console

    from dlm.doc.parser import parse_file
    from dlm.metrics.queries import (
        evals_for_run,
        evals_to_dict,
        preference_mining_for_run,
        preference_mining_to_dict,
        recent_runs,
        runs_to_dict,
        steps_for_run,
        steps_to_dict,
    )
    from dlm.store.paths import for_dlm

    console = Console(stderr=True)

    if json_out and csv_out:
        console.print("[red]metrics:[/red] --json and --csv are mutually exclusive")
        raise typer.Exit(code=2)

    since_delta = _parse_since_arg(since, console) if since else None

    parsed = parse_file(path)
    store = for_dlm(parsed.frontmatter.dlm_id)

    runs = recent_runs(store.root, limit=limit, phase=phase, since=since_delta, run_id=run_id)

    if run_id is not None:
        # Drill-down: show this run's steps + evals.
        if not runs:
            console.print(f"[red]metrics:[/red] no run with run_id={run_id}")
            raise typer.Exit(code=1)
        run = runs[0]
        steps = steps_for_run(store.root, run_id)
        evals = evals_for_run(store.root, run_id)
        preference_rows = preference_mining_for_run(store.root, run_id)

        if json_out:
            payload = {
                "run": runs_to_dict([run])[0],
                "steps": steps_to_dict(steps),
                "evals": evals_to_dict(evals),
                "preference_mining": preference_mining_to_dict(preference_rows),
            }
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
            return
        if csv_out:
            writer = csv.writer(sys.stdout)
            writer.writerow(["step", "loss", "lr", "grad_norm", "val_loss"])
            eval_by_step = {e.step: e.val_loss for e in evals}
            for s in steps:
                writer.writerow([s.step, s.loss, s.lr, s.grad_norm, eval_by_step.get(s.step)])
            return
        console.print(
            f"[green]run_id={run.run_id}[/green]  phase={run.phase}  "
            f"seed={run.seed}  status={run.status}  steps={len(steps)}  "
            f"evals={len(evals)}"
        )
        if evals:
            last = evals[-1]
            console.print(
                f"  last eval: step={last.step}  val_loss={last.val_loss}  "
                f"perplexity={last.perplexity}"
            )
        if preference_rows:
            last_pref = preference_rows[-1]
            console.print(
                "  preference mining: "
                f"events={len(preference_rows)}  "
                f"mined_pairs={sum(row.mined_pairs for row in preference_rows)}  "
                f"skipped_prompts={sum(row.skipped_prompts for row in preference_rows)}  "
                f"last_mode={last_pref.write_mode}  "
                f"judge={last_pref.judge_name}"
            )
        return

    # Top-level: list runs.
    if json_out:
        sys.stdout.write(json.dumps({"runs": runs_to_dict(runs)}, indent=2) + "\n")
        return
    if csv_out:
        writer = csv.writer(sys.stdout)
        writer.writerow(["run_id", "phase", "seed", "status", "started_at", "ended_at"])
        for r in runs:
            writer.writerow([r.run_id, r.phase, r.seed, r.status, r.started_at, r.ended_at])
        return

    if not runs:
        console.print("[dim]metrics:[/dim] no runs found (hint: train first, or adjust filters)")
        return
    console.print(f"[bold]Runs: {len(runs)}[/bold]")
    for r in runs:
        console.print(
            f"  run_id={r.run_id}  phase={r.phase}  seed={r.seed}  "
            f"status={r.status}  started={r.started_at}"
        )


def metrics_watch_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file whose store we tail.")],
    poll_seconds: Annotated[
        float,
        typer.Option("--poll-seconds", help="How often to re-read the metrics DB."),
    ] = 1.0,
) -> None:
    """Tail the metrics DB: print new steps/evals as they land."""
    import time

    from rich.console import Console

    from dlm.doc.parser import parse_file
    from dlm.metrics.queries import evals_for_run, latest_run_id, steps_for_run
    from dlm.store.paths import for_dlm

    console = Console()

    parsed = parse_file(path)
    store = for_dlm(parsed.frontmatter.dlm_id)

    console.print(
        f"[dim]metrics watch:[/dim] polling {store.root} every {poll_seconds}s (Ctrl-C to exit)"
    )

    current_run: int | None = None
    last_step_seen = 0
    last_eval_step_seen = 0
    try:
        while True:
            run_id = latest_run_id(store.root)
            if run_id is None:
                time.sleep(poll_seconds)
                continue
            if run_id != current_run:
                current_run = run_id
                last_step_seen = 0
                last_eval_step_seen = 0
                console.print(f"[green]→ following run_id={run_id}[/green]")

            new_steps = steps_for_run(store.root, run_id, since_step=last_step_seen)
            for s in new_steps:
                console.print(
                    f"  step {s.step:>5}  loss={s.loss}  lr={s.lr}  grad_norm={s.grad_norm}"
                )
                last_step_seen = s.step

            new_evals = evals_for_run(store.root, run_id, since_step=last_eval_step_seen)
            for e in new_evals:
                console.print(
                    f"  [yellow]eval @ step {e.step}[/yellow]  "
                    f"val_loss={e.val_loss}  perplexity={e.perplexity}"
                )
                last_eval_step_seen = e.step

            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        console.print("[dim]metrics watch:[/dim] bye")


def _parse_since_arg(since: str, console: object) -> timedelta:
    """Parse `24h` / `7d` / `30m` / `10s` into a timedelta."""
    from rich.console import Console

    assert isinstance(console, Console)

    if not since:
        raise typer.Exit(code=2)
    unit = since[-1].lower()
    try:
        value = int(since[:-1])
    except ValueError:
        console.print(f"[red]metrics:[/red] --since {since!r} not an integer+unit")
        raise typer.Exit(code=2) from None
    if unit == "s":
        return timedelta(seconds=value)
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    console.print(f"[red]metrics:[/red] --since {since!r} unit must be s/m/h/d")
    raise typer.Exit(code=2)
