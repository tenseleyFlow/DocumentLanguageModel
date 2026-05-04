"""`dlm repl` — interactive REPL against the trained adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from dlm.cli.commands._shared import _previously_accepted


def repl_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to start a REPL against.")],
    adapter: Annotated[
        str | None,
        typer.Option(
            "--adapter",
            help=(
                "Named adapter to load. Required on multi-adapter "
                "documents; rejected on single-adapter documents."
            ),
        ),
    ] = None,
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help="Inference backend: `auto`, `pytorch`, or `mlx`.",
        ),
    ] = "auto",
) -> None:
    """Interactive REPL against the trained adapter."""
    from rich.console import Console

    from dlm.base_models import GatedModelError
    from dlm.base_models import resolve as resolve_base_model
    from dlm.doc.parser import parse_file
    from dlm.hardware import doctor
    from dlm.inference import AdapterNotFoundError
    from dlm.inference.backends import (
        UnsupportedBackendError,
        build_backend,
        select_backend,
    )
    from dlm.repl.session import ReplSession
    from dlm.store.paths import for_dlm

    console = Console(stderr=True)

    if backend not in ("auto", "pytorch", "mlx"):
        console.print(
            f"[red]repl:[/red] --backend must be `auto`, `pytorch`, or `mlx` (got {backend!r})."
        )
        raise typer.Exit(code=2)

    parsed = parse_file(path)
    declared = parsed.frontmatter.training.adapters
    if adapter is not None:
        if declared is None:
            console.print(
                "[red]repl:[/red] --adapter is only valid on multi-adapter "
                "documents (this doc does not declare `training.adapters`)."
            )
            raise typer.Exit(code=2)
        if adapter not in declared:
            console.print(
                f"[red]repl:[/red] --adapter {adapter!r} is not declared "
                f"(declared: {sorted(declared)!r})."
            )
            raise typer.Exit(code=2)

    store = for_dlm(parsed.frontmatter.dlm_id)
    already_accepted = _previously_accepted(store.manifest)
    try:
        spec = resolve_base_model(parsed.frontmatter.base_model, accept_license=already_accepted)
    except GatedModelError as exc:
        console.print(
            f"[red]license:[/red] base {parsed.frontmatter.base_model!r} is gated and has "
            "no recorded acceptance in this store; run `dlm train --i-accept-license` first."
        )
        raise typer.Exit(code=1) from exc
    caps = doctor().capabilities

    try:
        backend_name = select_backend(backend, caps)  # type: ignore[arg-type]
    except UnsupportedBackendError as exc:
        console.print(f"[red]repl:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    backend_obj = build_backend(backend_name, caps)

    try:
        backend_obj.load(spec, store, adapter_name=adapter)
    except AdapterNotFoundError as exc:
        console.print(f"[red]repl:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    tokenizer = getattr(backend_obj, "_loaded", None)
    tokenizer = tokenizer.tokenizer if tokenizer is not None else None

    session = ReplSession(
        backend=backend_obj,
        tokenizer=tokenizer,
        active_adapter=adapter,
        declared_adapters=tuple(sorted(declared)) if declared else (),
    )

    from dlm.repl.app import run_repl

    raise typer.Exit(code=run_repl(session, console=console))
