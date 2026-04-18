"""Subcommand stubs for the v1.0 CLI surface.

Every stub raises `NotImplementedError` with the sprint number that will
implement it. This makes `dlm --help` self-documenting about project
progress.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def _stub(sprint: str, subject: str) -> None:
    """Raise a clear unimplemented error pointing to the owning sprint."""
    raise NotImplementedError(
        f"`{subject}` is not implemented yet (owned by Sprint {sprint}).",
    )


def init_cmd(
    path: Annotated[Path, typer.Argument(help="Target .dlm path to create.")],
    base: Annotated[
        str, typer.Option("--base", help="Base model key or hf:org/name.")
    ] = "qwen2.5-1.5b",
    template: Annotated[
        str | None, typer.Option("--template", help="Starter template name (Sprint 27).")
    ] = None,
    i_accept_license: Annotated[
        bool,
        typer.Option("--i-accept-license", help="Accept gated base-model license (Sprint 12b)."),
    ] = False,
) -> None:
    """Bootstrap a new .dlm file with sensible defaults."""
    _ = (path, base, template, i_accept_license)
    _stub("13", "dlm init")


def train_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to train.")],
    resume: Annotated[bool, typer.Option("--resume", help="Resume from last checkpoint.")] = False,
    fresh: Annotated[bool, typer.Option("--fresh", help="Discard prior adapter state.")] = False,
    seed: Annotated[int | None, typer.Option("--seed", help="Override training seed.")] = None,
    max_steps: Annotated[int | None, typer.Option("--max-steps", help="Cap step count.")] = None,
) -> None:
    """Train / retrain a .dlm against its base model."""
    _ = (path, resume, fresh, seed, max_steps)
    _stub("09", "dlm train")


def prompt_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to query.")],
    query: Annotated[str | None, typer.Argument(help="One-shot prompt (omit for stdin).")] = None,
    max_tokens: Annotated[int, typer.Option("--max-tokens")] = 256,
    temp: Annotated[float, typer.Option("--temp")] = 0.7,
) -> None:
    """Run inference against the trained adapter."""
    _ = (path, query, max_tokens, temp)
    _stub("10", "dlm prompt")


def export_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to export.")],
    quant: Annotated[str, typer.Option("--quant")] = "Q4_K_M",
    merged: Annotated[bool, typer.Option("--merged")] = False,
    dequantize: Annotated[bool, typer.Option("--dequantize")] = False,
    name: Annotated[str | None, typer.Option("--name", help="Ollama model name.")] = None,
    no_smoke: Annotated[bool, typer.Option("--no-smoke")] = False,
) -> None:
    """Export the adapter to an Ollama-registered model."""
    _ = (path, quant, merged, dequantize, name, no_smoke)
    _stub("11+12", "dlm export")


def pack_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to pack.")],
    out: Annotated[Path | None, typer.Option("--out")] = None,
    include_exports: Annotated[bool, typer.Option("--include-exports")] = False,
    include_base: Annotated[bool, typer.Option("--include-base")] = False,
) -> None:
    """Produce a portable .dlm.pack bundle."""
    _ = (path, out, include_exports, include_base)
    _stub("14", "dlm pack")


def unpack_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm.pack to install.")],
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Install a .dlm.pack into the local store."""
    _ = (path, force)
    _stub("14", "dlm unpack")


def doctor_cmd(
    json_out: Annotated[bool, typer.Option("--json", help="Emit machine-readable output.")] = False,
) -> None:
    """Inspect hardware and print the resolved training plan."""
    _ = json_out
    _stub("05", "dlm doctor")


def show_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to inspect.")],
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show training history, exports, and adapter state."""
    _ = (path, json_out)
    _stub("13", "dlm show")


def migrate_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to migrate.")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    no_backup: Annotated[bool, typer.Option("--no-backup")] = False,
) -> None:
    """Migrate a .dlm frontmatter to the current schema version."""
    _ = (path, dry_run, no_backup)
    _stub("12b", "dlm migrate")
