"""`dlm migrate` — migrate a .dlm frontmatter to the current schema version."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def migrate_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to migrate.")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    no_backup: Annotated[bool, typer.Option("--no-backup")] = False,
) -> None:
    """Migrate a .dlm frontmatter to the current schema version."""
    from rich.console import Console

    from dlm.doc.errors import DlmParseError
    from dlm.doc.migrate import migrate_file

    console = Console(stderr=True)

    try:
        result = migrate_file(path, dry_run=dry_run, no_backup=no_backup)
    except DlmParseError as exc:
        console.print(f"[red]migrate:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not result.applied:
        console.print(
            f"[green]migrate:[/green] {path} already at v{result.target_version} "
            "(no migrations needed)."
        )
        return

    applied_str = " → ".join(f"v{v}" for v in (*result.applied, result.target_version))
    if dry_run:
        console.print(
            f"[yellow]dry-run:[/yellow] {path} would migrate {applied_str} "
            "(re-run without --dry-run to apply)."
        )
        return

    if result.backup_path is not None:
        console.print(f"[dim]backup:[/dim]  {result.backup_path}")
    console.print(f"[green]migrated:[/green] {path} {applied_str}")
