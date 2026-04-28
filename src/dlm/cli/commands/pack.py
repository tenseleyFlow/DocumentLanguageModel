"""`dlm pack` — produce a portable .dlm.pack bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def pack_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to pack.")],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output .dlm.pack path."),
    ] = None,
    include_exports: Annotated[
        bool,
        typer.Option("--include-exports", help="Bundle all GGUF exports into the pack."),
    ] = False,
    include_base: Annotated[
        bool,
        typer.Option(
            "--include-base",
            help="Bundle the base model snapshot (license rules still apply).",
        ),
    ] = False,
    include_logs: Annotated[
        bool,
        typer.Option("--include-logs", help="Bundle per-run JSONL logs."),
    ] = False,
    licensee: Annotated[
        str | None,
        typer.Option(
            "--i-am-the-licensee",
            help="URL acknowledging separate acceptance of a non-redistributable base (required for --include-base on gated models).",
        ),
    ] = None,
) -> None:
    """Produce a portable .dlm.pack bundle."""
    from rich.console import Console

    from dlm.doc.errors import DlmParseError
    from dlm.pack.errors import BaseLicenseRefusedError, PackError
    from dlm.pack.packer import pack

    console = Console(stderr=True)

    try:
        result = pack(
            path,
            out=out,
            include_exports=include_exports,
            include_base=include_base,
            include_logs=include_logs,
            licensee_acceptance_url=licensee,
        )
    except BaseLicenseRefusedError as exc:
        console.print(f"[red]pack:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except PackError as exc:
        console.print(f"[red]pack:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except DlmParseError as exc:
        console.print(f"[red]parse:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    size_mb = result.bytes_written / (1024 * 1024)
    console.print(
        f"[green]packed:[/green] {result.path} "
        f"({size_mb:.2f} MB, content_type={result.content_type})"
    )
