"""`dlm cache` — show / prune / clear the tokenized-section cache."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from dlm.cli.commands._shared import _human_size


def cache_show_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to inspect the cache for.")],
    json_out: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Show tokenized-section cache size, entry count, last-run hit rate."""
    import json as _json
    import sys as _sys

    from rich.console import Console

    from dlm.directives.cache import TokenizedCache
    from dlm.doc.errors import DlmParseError
    from dlm.doc.parser import parse_file
    from dlm.metrics import queries as _queries
    from dlm.store.paths import for_dlm

    console = Console(stderr=True)
    out_console = Console()

    try:
        parsed = parse_file(path)
    except (DlmParseError, OSError) as exc:
        console.print(f"[red]cache:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    store = for_dlm(parsed.frontmatter.dlm_id)
    cache = TokenizedCache.open(store.tokenized_cache_dir)
    last = _queries.latest_tokenization(store.root)

    payload: dict[str, object] = {
        "dlm_id": parsed.frontmatter.dlm_id,
        "cache_path": str(store.tokenized_cache_dir),
        "entry_count": cache.entry_count,
        "bytes": cache.total_bytes,
        "last_run_hit_rate": last.hit_rate if last else None,
        "last_run_id": last.run_id if last else None,
    }
    if json_out:
        _sys.stdout.write(_json.dumps(payload, indent=2) + "\n")
        return

    out_console.print(f"[bold]Cache for {parsed.frontmatter.dlm_id}[/bold]")
    out_console.print(f"  path:              {store.tokenized_cache_dir}")
    out_console.print(f"  entries:           {cache.entry_count}")
    out_console.print(f"  size:              {_human_size(cache.total_bytes)}")
    if last is not None:
        out_console.print(
            f"  last-run hit rate: {last.hit_rate:.1%} "
            f"({last.cache_hits}/{last.cache_hits + last.cache_misses})"
        )
    else:
        out_console.print("  last-run hit rate: [dim]no tokenization runs yet[/dim]")


def cache_prune_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to prune the cache for.")],
    older_than: Annotated[
        str | None,
        typer.Option(
            "--older-than",
            help=(
                "Drop entries not accessed in this duration. "
                "Format: `30d`, `12h`, `45m`. When omitted, defaults to "
                "the document's `training.cache.prune_older_than_days` "
                "(90d pre-v9 docs inherit)."
            ),
        ),
    ] = None,
) -> None:
    """Remove tokenized-cache entries not accessed within a cutoff."""
    from rich.console import Console

    from dlm.directives.cache import TokenizedCache
    from dlm.doc.errors import DlmParseError
    from dlm.doc.parser import parse_file
    from dlm.store.paths import for_dlm

    console = Console(stderr=True)

    # Parse the doc first — we need it either way (for dlm_id) AND
    # for the frontmatter default when --older-than is absent.
    try:
        parsed = parse_file(path)
    except (DlmParseError, OSError) as exc:
        console.print(f"[red]cache:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if older_than is not None:
        seconds = _parse_duration(older_than)
        if seconds is None:
            console.print(
                f"[red]cache:[/red] invalid --older-than {older_than!r} "
                "(expected e.g. 30d, 12h, 45m)"
            )
            raise typer.Exit(code=2)
        cutoff_label = older_than
    else:
        # Fall back to the frontmatter's per-doc default. Pre-v9 docs
        # get the CacheConfig default of 90 days via the Pydantic
        # factory on parse.
        days = parsed.frontmatter.training.cache.prune_older_than_days
        seconds = float(days) * 86400.0
        cutoff_label = f"{days}d"

    store = for_dlm(parsed.frontmatter.dlm_id)
    cache = TokenizedCache.open(store.tokenized_cache_dir)
    removed = cache.prune(older_than_seconds=seconds)
    cache.save_manifest()
    console.print(f"[green]cache:[/green] pruned {removed} entr(y/ies) older than {cutoff_label}")


def cache_clear_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to wipe the cache for.")],
    force: Annotated[
        bool,
        typer.Option("--force", help="Skip the confirmation prompt."),
    ] = False,
) -> None:
    """Wipe every entry in the tokenized-section cache for this store."""
    from rich.console import Console

    from dlm.directives.cache import TokenizedCache
    from dlm.doc.errors import DlmParseError
    from dlm.doc.parser import parse_file
    from dlm.store.paths import for_dlm

    console = Console(stderr=True)

    try:
        parsed = parse_file(path)
    except (DlmParseError, OSError) as exc:
        console.print(f"[red]cache:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    store = for_dlm(parsed.frontmatter.dlm_id)
    cache = TokenizedCache.open(store.tokenized_cache_dir)

    if not force and cache.entry_count > 0:
        confirmed = typer.confirm(
            f"wipe {cache.entry_count} entries ({_human_size(cache.total_bytes)})?"
        )
        if not confirmed:
            console.print("[yellow]cache:[/yellow] clear cancelled")
            raise typer.Exit(code=0)

    removed = cache.clear()
    cache.save_manifest()
    console.print(f"[green]cache:[/green] cleared {removed} entr(y/ies)")


def _parse_duration(spec: str) -> float | None:
    """Parse a duration like `30d`, `12h`, `45m` → seconds. None on
    malformed input."""
    if not spec or not spec[:-1].isdigit():
        return None
    n = int(spec[:-1])
    unit = spec[-1].lower()
    if unit == "s":
        return float(n)
    if unit == "m":
        return float(n) * 60
    if unit == "h":
        return float(n) * 3600
    if unit == "d":
        return float(n) * 86400
    return None
