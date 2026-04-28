"""`dlm serve` — serve a .dlm's pack over LAN for peers to pull."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def serve_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to serve.")],
    port: Annotated[int, typer.Option("--port")] = 7337,
    public: Annotated[
        bool,
        typer.Option(
            "--public",
            help="Bind 0.0.0.0 (requires --i-know-this-is-public); otherwise 127.0.0.1.",
        ),
    ] = False,
    i_know_public: Annotated[
        bool,
        typer.Option(
            "--i-know-this-is-public",
            help="Confirm binding 0.0.0.0 is safe on this network.",
        ),
    ] = False,
    max_concurrency: Annotated[
        int,
        typer.Option("--max-concurrency", help="Max concurrent connections per token."),
    ] = 4,
    rate_limit: Annotated[
        int,
        typer.Option("--rate-limit", help="Max requests per minute per token."),
    ] = 30,
    token_ttl_minutes: Annotated[
        int, typer.Option("--token-ttl-minutes", help="Token lifetime in minutes.")
    ] = 15,
) -> None:
    """Serve a .dlm's pack over LAN for peers to pull."""
    from rich.console import Console

    from dlm.doc.parser import parse_file
    from dlm.pack.packer import pack as pack_fn
    from dlm.share import ServeOptions, serve
    from dlm.store.paths import for_dlm

    console = Console(stderr=True)

    parsed = parse_file(path)
    dlm_id = parsed.frontmatter.dlm_id

    # pack() calls load_manifest(), which crashes with an unhelpful
    # "store manifest corrupt" error on a .dlm that's never been
    # trained. Surface the true cause instead.
    store = for_dlm(dlm_id)
    if not store.manifest.exists():
        console.print(
            f"[red]serve:[/red] no training state for {dlm_id} — run [bold]dlm train[/bold] first."
        )
        raise typer.Exit(code=1)

    # Pack into a temp file that lives as long as the server does.
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="dlm-serve-"))
    tmp_pack = tmp_dir / f"{path.stem}.dlm.pack"
    pack_fn(path, out=tmp_pack)
    console.print(f"[dim]packed:[/dim] {tmp_pack} ({tmp_pack.stat().st_size} bytes)")

    opts = ServeOptions(
        port=port,
        public=public,
        i_know_this_is_public=i_know_public,
        max_concurrency=max_concurrency,
        rate_limit_per_min=rate_limit,
        token_ttl_seconds=token_ttl_minutes * 60,
    )
    handle = serve(dlm_id, tmp_pack, opts)

    console.print(
        f"[green]serving:[/green] {path.name} (dlm_id {dlm_id}) on "
        f"[bold]http://{handle.bind_host}:{handle.port}/{dlm_id}[/bold]"
    )
    console.print(f"[bold]peer URL:[/bold] {handle.peer_url}")
    console.print(f"[dim]token valid for {token_ttl_minutes} min. Ctrl-C to stop.[/dim]")

    try:
        handle.wait_shutdown()
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
    console.print("[dim]stopped.[/dim]")
