"""`dlm templates list` — list the bundled template gallery."""

from __future__ import annotations

from typing import Annotated

import typer


def templates_list_cmd(
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit a JSON array of template metadata."),
    ] = False,
    refresh: Annotated[
        bool,
        typer.Option(
            "--refresh",
            help=(
                "Refresh from the upstream template gallery. Currently a no-op — "
                "upstream repo + signing key are deferred."
            ),
        ),
    ] = False,
    accept_unsigned: Annotated[
        bool,
        typer.Option(
            "--accept-unsigned",
            help=(
                "Bypass signed-tag verification on --refresh. Reserved; takes effect "
                "once the upstream gallery signs its releases."
            ),
        ),
    ] = False,
) -> None:
    """List the bundled (and, one day, remote) template gallery."""

    import json as _json

    from rich.console import Console

    from dlm.templates import list_bundled

    console_out = Console()
    console_err = Console(stderr=True)

    if refresh:
        from dlm.templates.fetcher import RemoteFetchUnavailable, cache_dir, fetch_all

        try:
            fetch_all(cache_dir(), remote="")
        except RemoteFetchUnavailable as exc:
            console_err.print(
                f"[yellow]templates:[/yellow] {exc} Falling back to the bundled gallery."
            )
        # --accept-unsigned is reserved for when the live fetcher lands;
        # touching it here silences ARG001 without ceremony.
        _ = accept_unsigned

    templates = list_bundled()

    if json_out:
        payload = [
            {
                "name": t.name,
                "title": t.meta.title,
                "domain_tags": list(t.meta.domain_tags),
                "recommended_base": t.meta.recommended_base,
                "expected_steps": t.meta.expected_steps,
                "expected_duration": dict(t.meta.expected_duration),
                "summary": t.meta.summary,
                "sample_prompts": list(t.meta.sample_prompts),
            }
            for t in templates
        ]
        console_out.print_json(_json.dumps(payload))
        return

    if not templates:
        console_err.print("[yellow]templates:[/yellow] no bundled templates found.")
        raise typer.Exit(code=1)

    name_width = max(len(t.name) for t in templates)
    for t in templates:
        console_out.print(
            f"[bold]{t.name:<{name_width}}[/bold]  {t.meta.title}  "
            f"[dim]({t.meta.recommended_base})[/dim]"
        )
