"""`dlm pull` — download + verify + unpack a .dlm.pack from a remote source."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def pull_cmd(
    source: Annotated[
        str,
        typer.Argument(
            help=(
                "Source: `hf:<org>/<repo>`, `https://...`, "
                "`peer://host:port/<id>?token=...`, or a local path."
            )
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Directory for the restored .dlm (default: CWD)."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing store with the same dlm_id."),
    ] = False,
) -> None:
    """Download + verify + unpack a .dlm.pack from a remote source."""
    from rich.console import Console

    from dlm.pack.errors import PackError
    from dlm.share import ShareError, pull
    from dlm.share.signing import VerifyStatus

    console = Console(stderr=True)

    try:
        result = pull(source, out_dir=out, force=force)
    except ShareError as exc:
        console.print(f"[red]pull:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except PackError as exc:
        console.print(f"[red]pull:[/red] pack integrity: {exc}")
        raise typer.Exit(code=1) from exc

    size_mb = result.bytes_received / (1024 * 1024)
    console.print(f"[green]pulled:[/green] {result.source} → {result.dlm_path} ({size_mb:.2f} MB)")

    status = result.verification.status
    if status == VerifyStatus.VERIFIED:
        console.print(
            f"[green]verified:[/green] signature matches "
            f"[bold]{result.verification.key_path}[/bold]"
        )
    elif status == VerifyStatus.UNVERIFIED:
        console.print(
            f"[yellow]unverified:[/yellow] signature present but "
            f"not matched ({result.verification.detail}); sha256 still validated"
        )
    else:
        console.print("[dim]unsigned[/dim] (sha256 integrity still validated)")
