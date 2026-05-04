"""Top-level Typer application.

Telemetry-off defaults are applied before any downstream imports.
Subcommand stubs live in `dlm.cli.commands`.
"""

from __future__ import annotations

import os


def _disable_third_party_telemetry() -> None:
    """Force opt-out env vars before third-party imports.

    This is load-bearing for the "no telemetry, ever" promise. Must
    run before `transformers`, `huggingface_hub`, `wandb`, or similar
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
    """Root callback — configure DLM_HOME + logger level for downstream cmds."""
    import logging

    _ = version  # consumed by is_eager callback before we arrive

    if home is not None:
        # Make the override visible to every `StorePath.for_dlm`
        # downstream of this callback.
        os.environ["DLM_HOME"] = home

    if verbose and quiet:
        raise typer.BadParameter("--verbose and --quiet are mutually exclusive")

    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING

    # Only attach a handler if the root logger hasn't already been
    # configured by the parent process (e.g. in tests). `force=True`
    # resets any prior CLI-run handlers so --verbose on a second call
    # actually takes effect.
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        force=True,
    )


# Register subcommand stubs.
app.command("init")(commands.init_cmd)
app.command("train")(commands.train_cmd)
app.command("prompt")(commands.prompt_cmd)
app.command("repl")(commands.repl_cmd)
app.command("export")(commands.export_cmd)
app.command("pack")(commands.pack_cmd)
app.command("unpack")(commands.unpack_cmd)
app.command("verify")(commands.verify_cmd)
app.command("push")(commands.push_cmd)
app.command("pull")(commands.pull_cmd)
app.command("serve")(commands.serve_cmd)
app.command("doctor")(commands.doctor_cmd)
app.command("show")(commands.show_cmd)
app.command("migrate")(commands.migrate_cmd)
app.command("harvest")(commands.harvest_cmd)

# `dlm synth instructions|preferences|revert|list` — synthetic data loop.
_synth_app = typer.Typer(
    help="Synthesize instruction or preference training data.",
    no_args_is_help=True,
)
_synth_app.command("instructions")(commands.synth_instructions_cmd)
_synth_app.command("preferences")(commands.preference_mine_cmd)
_synth_app.command("revert")(commands.synth_revert_cmd)
_synth_app.command("list")(commands.synth_list_cmd)
app.add_typer(_synth_app, name="synth")

# `dlm preference mine|apply|revert|list` — auto-mined preference loop.
_preference_app = typer.Typer(
    help="Mine, stage, apply, and inspect auto-mined preference sections.",
    no_args_is_help=True,
)
_preference_app.command("mine")(commands.preference_mine_cmd)
_preference_app.command("apply")(commands.preference_apply_cmd)
_preference_app.command("revert")(commands.preference_revert_cmd)
_preference_app.command("list")(commands.preference_list_cmd)
app.add_typer(_preference_app, name="preference")

# `dlm metrics show <path>` + `dlm metrics watch <path>` as a
# subcommand group. The previous shape — a callback that took the
# positional `path` plus a `watch` subcommand — broke with
# "Missing argument 'PATH'" when an option came after the positional
# (`dlm metrics PATH --run-id 1`), because click can't disambiguate
# a positional-then-option from a subcommand-then-args inside the
# same group. Audit 13 M13.3. The explicit `show` subcommand removes
# the ambiguity. Run-time impact: `dlm metrics PATH` now needs to be
# `dlm metrics show PATH` — flagged in CHANGELOG.
_metrics_app = typer.Typer(
    help="Query the per-store metrics database.",
    no_args_is_help=True,
)
_metrics_app.command("show")(commands.metrics_cmd)
_metrics_app.command("watch")(commands.metrics_watch_cmd)
app.add_typer(_metrics_app, name="metrics")

# `dlm templates list` lives under its own subcommand group.
_templates_app = typer.Typer(
    help="Browse the starter template gallery.",
    no_args_is_help=True,
)
_templates_app.command("list")(commands.templates_list_cmd)
app.add_typer(_templates_app, name="templates")

# `dlm cache show|prune|clear` — per-store tokenized-section cache
# maintenance.
_cache_app = typer.Typer(
    help="Inspect and manage the per-store tokenized-section cache.",
    no_args_is_help=True,
)
_cache_app.command("show")(commands.cache_show_cmd)
_cache_app.command("prune")(commands.cache_prune_cmd)
_cache_app.command("clear")(commands.cache_clear_cmd)
app.add_typer(_cache_app, name="cache")


def main() -> None:
    """Installed entry point (`dlm` on PATH).

    Routes uncaught exceptions through `dlm.cli.reporter` so users see
    a clean one-liner instead of a raw traceback; `--verbose` surfaces
    the traceback for debugging.
    """
    import sys

    from dlm.cli.reporter import run_with_reporter

    sys.exit(run_with_reporter(app))


if __name__ == "__main__":
    main()
