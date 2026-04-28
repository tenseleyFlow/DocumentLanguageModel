"""`dlm push` — upload a .dlm or .dlm.pack to an HF repo, URL, or local path."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def push_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm or .dlm.pack to push.")],
    to: Annotated[
        str,
        typer.Option(
            "--to",
            help=(
                "Destination. `hf:<org>/<repo>` for HuggingFace Hub, "
                "`https://...` for a generic HTTPS endpoint, or a local path."
            ),
        ),
    ],
    sign: Annotated[
        bool,
        typer.Option("--sign", help="Sign the pack with minisign before upload."),
    ] = False,
    include_exports: Annotated[bool, typer.Option("--include-exports")] = False,
    include_base: Annotated[bool, typer.Option("--include-base")] = False,
    include_logs: Annotated[bool, typer.Option("--include-logs")] = False,
    licensee: Annotated[
        str | None,
        typer.Option(
            "--i-am-the-licensee",
            help="URL ack for --include-base on non-redistributable bases.",
        ),
    ] = None,
) -> None:
    """Upload a .dlm or .dlm.pack to an HF repo, URL endpoint, or local path."""
    from rich.console import Console

    from dlm.share import ShareError, push
    from dlm.share.signing import MinisignNotAvailableError

    console = Console(stderr=True)

    try:
        result = push(
            path,
            to,
            sign=sign,
            include_exports=include_exports,
            include_base=include_base,
            include_logs=include_logs,
            licensee_acceptance_url=licensee,
        )
    except MinisignNotAvailableError as exc:
        console.print(f"[red]push:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ShareError as exc:
        console.print(f"[red]push:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    size_mb = result.bytes_sent / (1024 * 1024)
    console.print(f"[green]pushed:[/green] {result.destination} ({size_mb:.2f} MB)")
    if result.sink_kind.value == "hf":
        console.print(f"[dim]install:[/dim] dlm pull {result.destination}")
    if result.detail:
        console.print(f"[dim]{result.detail}[/dim]")
