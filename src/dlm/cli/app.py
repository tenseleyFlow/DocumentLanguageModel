"""Top-level Typer application.

Telemetry-off defaults are applied before any downstream imports (audit F13).
Subcommand stubs live in `dlm.cli.commands` and raise `NotImplementedError`
citing the sprint that will implement them.
"""

from __future__ import annotations

import os


def _disable_third_party_telemetry() -> None:
    """Force opt-out env vars before third-party imports.

    This is load-bearing for the "no telemetry, ever" promise (audit F13).
    Must run before `transformers`, `huggingface_hub`, `wandb`, or similar
    are imported anywhere in the process.
    """
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


_disable_third_party_telemetry()


# Imports below are deferred so `_disable_third_party_telemetry` runs first.
from typing import Annotated  # noqa: E402

import typer  # noqa: E402

from dlm import __version__  # noqa: E402
from dlm.cli import commands  # noqa: E402

app = typer.Typer(
    name="dlm",
    help="DocumentLanguageModel — a text file becomes a local, trainable LLM.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"dlm {__version__}")
        raise typer.Exit(code=0)


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
    home: Annotated[
        str | None,
        typer.Option("--home", envvar="DLM_HOME", help="Override $DLM_HOME (default ~/.dlm)."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose logging.")] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress informational output.")
    ] = False,
) -> None:
    """Root callback. Sprints 13+ flesh out logging and $DLM_HOME handling."""
    # Arguments captured for future use; this sprint is scaffolding only.
    _ = (version, home, verbose, quiet)


# Register subcommand stubs.
app.command("init")(commands.init_cmd)
app.command("train")(commands.train_cmd)
app.command("prompt")(commands.prompt_cmd)
app.command("export")(commands.export_cmd)
app.command("pack")(commands.pack_cmd)
app.command("unpack")(commands.unpack_cmd)
app.command("doctor")(commands.doctor_cmd)
app.command("show")(commands.show_cmd)
app.command("migrate")(commands.migrate_cmd)


def main() -> None:
    """Installed entry point (`dlm` on PATH)."""
    app()
