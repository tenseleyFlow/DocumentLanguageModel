"""Subcommand stubs for the v1.0 CLI surface.

Every stub raises `NotImplementedError` with the sprint number that will
implement it. This makes `dlm --help` self-documenting about project
progress. Arguments are accepted so `--help` renders the real eventual
surface; they're unused until each subcommand's owning sprint lands,
which is why `src/dlm/cli/commands.py` has a ruff per-file-ignore for
`ARG001` in `pyproject.toml`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

from dlm.cli.commands._shared import _human_size as _human_size
from dlm.cli.commands._shared import _previously_accepted as _previously_accepted
from dlm.cli.commands.cache import _parse_duration as _parse_duration
from dlm.cli.commands.cache import cache_clear_cmd as cache_clear_cmd
from dlm.cli.commands.cache import cache_prune_cmd as cache_prune_cmd
from dlm.cli.commands.cache import cache_show_cmd as cache_show_cmd
from dlm.cli.commands.doctor import doctor_cmd as doctor_cmd
from dlm.cli.commands.harvest import harvest_cmd as harvest_cmd
from dlm.cli.commands.metrics import metrics_cmd as metrics_cmd
from dlm.cli.commands.metrics import metrics_watch_cmd as metrics_watch_cmd
from dlm.cli.commands.migrate import migrate_cmd as migrate_cmd
from dlm.cli.commands.pack import pack_cmd as pack_cmd
from dlm.cli.commands.preference import preference_apply_cmd as preference_apply_cmd
from dlm.cli.commands.preference import preference_list_cmd as preference_list_cmd
from dlm.cli.commands.preference import preference_mine_cmd as preference_mine_cmd
from dlm.cli.commands.preference import preference_revert_cmd as preference_revert_cmd
from dlm.cli.commands.pull import pull_cmd as pull_cmd
from dlm.cli.commands.push import push_cmd as push_cmd
from dlm.cli.commands.repl import repl_cmd as repl_cmd
from dlm.cli.commands.serve import serve_cmd as serve_cmd
from dlm.cli.commands.show import show_cmd as show_cmd
from dlm.cli.commands.synth import synth_instructions_cmd as synth_instructions_cmd
from dlm.cli.commands.synth import synth_list_cmd as synth_list_cmd
from dlm.cli.commands.synth import synth_revert_cmd as synth_revert_cmd
from dlm.cli.commands.templates import templates_list_cmd as templates_list_cmd
from dlm.cli.commands.unpack import unpack_cmd as unpack_cmd
from dlm.cli.commands.verify import verify_cmd as verify_cmd


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
        str | None,
        typer.Option(
            "--template",
            help="Start from a named gallery template (see `dlm templates list`).",
        ),
    ] = None,
    i_accept_license: Annotated[
        bool,
        typer.Option("--i-accept-license", help="Accept gated base-model license."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing .dlm at path."),
    ] = False,
    skip_export_probes: Annotated[
        bool,
        typer.Option(
            "--skip-export-probes",
            help=(
                "Skip the llama.cpp / GGUF-conversion probes so brand-new "
                "architectures (not yet in our vendored llama.cpp) can still "
                "be used for training + HF inference. Forfeits `dlm export` "
                "to Ollama until the vendored copy catches up."
            ),
        ),
    ] = False,
    multimodal: Annotated[
        bool,
        typer.Option(
            "--multimodal",
            help=(
                "Scaffold a vision-language .dlm with an `::image::` section. "
                "Defaults --base to paligemma-3b-mix-224 and skips GGUF "
                "export probes because current GGUF export does not "
                "support vision-language bases."
            ),
        ),
    ] = False,
    audio: Annotated[
        bool,
        typer.Option(
            "--audio",
            help=(
                "Scaffold an audio-language .dlm with an `::audio::` section. "
                "Defaults --base to qwen2-audio-7b-instruct and skips GGUF "
                "export probes (audio archs are not on llama.cpp's roadmap)."
            ),
        ),
    ] = False,
) -> None:
    """Bootstrap a new .dlm file with sensible defaults."""

    from rich.console import Console

    from dlm.base_models import (
        GatedModelError,
        UnknownBaseModelError,
        is_gated,
        require_acceptance,
    )
    from dlm.base_models import resolve as resolve_base_model
    from dlm.io.ulid import mint_ulid

    console = Console(stderr=True)

    if path.exists() and not force:
        console.print(
            f"[red]init:[/red] {path} already exists. "
            "Re-run with [bold]--force[/bold] to overwrite."
        )
        raise typer.Exit(code=1)

    # --multimodal / --audio are mutually exclusive with each other and
    # with --template (templates pin their own base + body shape; v1
    # doesn't ship media templates yet).
    if multimodal and audio:
        console.print(
            "[red]init:[/red] --multimodal and --audio are mutually exclusive "
            "(each targets a different modality)."
        )
        raise typer.Exit(code=2)
    if multimodal and template is not None:
        console.print(
            "[red]init:[/red] --multimodal and --template are mutually exclusive; "
            "v1 doesn't ship a VL template (see `dlm templates list`)."
        )
        raise typer.Exit(code=2)
    if audio and template is not None:
        console.print(
            "[red]init:[/red] --audio and --template are mutually exclusive; "
            "v1 doesn't ship an audio template (see `dlm templates list`)."
        )
        raise typer.Exit(code=2)

    # --multimodal / --audio override the text-first --base default. A
    # user who wants a different media base passes --base explicitly;
    # we verify the pick is the right modality below.
    if multimodal and base == "qwen2.5-1.5b":
        base = "paligemma-3b-mix-224"
    if audio and base == "qwen2.5-1.5b":
        base = "qwen2-audio-7b-instruct"

    # --template resolves the base from the template's meta.yaml; the
    # --base default is kept for the no-template path only. Users who
    # pass both a template and an explicit --base get a warning but the
    # template still wins (the template body was authored against its
    # recommended base).
    if template is not None:
        from dlm.templates import load_template

        # Peek at the template's recommended base WITHOUT writing
        # anything yet, so we can handle the license prompt against the
        # right base (the template's, not `--base`) before committing.
        try:
            resolved_base = load_template(template).meta.recommended_base
        except Exception as exc:
            console.print(f"[red]init:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        if base != "qwen2.5-1.5b" and base != resolved_base:
            console.print(
                f"[yellow]init:[/yellow] --base {base} ignored; template "
                f"{template!r} uses {resolved_base}."
            )
    else:
        resolved_base = base

    # Media bases can't clear the GGUF-conversion probes. Force-skip
    # them so the probe suite doesn't false-fail the init.
    if multimodal or audio:
        skip_export_probes = True

    try:
        spec = resolve_base_model(
            resolved_base,
            accept_license=i_accept_license,
            skip_export_probes=skip_export_probes,
        )
    except UnknownBaseModelError as exc:
        console.print(f"[red]init:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except GatedModelError as exc:
        # Gated + user didn't pass --i-accept-license up-front. Prompt
        # interactively if we have a TTY; otherwise refuse with the flag
        # pointer (audit F22 non-interactive path).
        if not _prompt_accept_license(console, resolved_base, exc.license_url):
            console.print(
                "[red]license:[/red] refused. Re-run with "
                "[bold]--i-accept-license[/bold] to accept non-interactively."
            )
            raise typer.Exit(code=1) from exc
        spec = resolve_base_model(
            resolved_base,
            accept_license=True,
            skip_export_probes=skip_export_probes,
        )

    # NOW apply the template — license has already been accepted
    # (either by --i-accept-license or interactive prompt), so pass
    # the acceptance through. apply_template enforces the license
    # contract at its boundary.
    applied_result = None
    if template is not None:
        from dlm.templates import TemplateError, apply_template

        try:
            applied_result = apply_template(template, path, force=force, accept_license=True)
        except TemplateError as exc:
            console.print(f"[red]init:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    # Record the license acceptance (or None for non-gated specs). We
    # know `resolve_base_model` already validated the flag/prompt chain
    # — `accept_license=True` means either the user passed the flag or
    # answered the interactive prompt. Either path is a real
    # acceptance; persist the record now so subsequent `dlm train` /
    # `dlm export` don't re-prompt.
    acceptance_via: Literal["cli_flag", "interactive"] = (
        "cli_flag" if i_accept_license else "interactive"
    )
    acceptance = (
        require_acceptance(spec, accept_license=True, via=acceptance_via)
        if is_gated(spec)
        else None
    )

    # Media flags require a matching-modality base. Check after resolve
    # so users pointing at an unknown or wrong-modality hf:org/name get
    # a clear explanation rather than a schema error deep in parse time.
    if multimodal and spec.modality != "vision-language":
        console.print(
            f"[red]init:[/red] --multimodal requires a vision-language base; "
            f"{spec.key!r} is modality='{spec.modality}'. "
            "Pick --base paligemma-3b-mix-224 or drop --multimodal."
        )
        raise typer.Exit(code=2)
    if audio and spec.modality != "audio-language":
        console.print(
            f"[red]init:[/red] --audio requires an audio-language base; "
            f"{spec.key!r} is modality='{spec.modality}'. "
            "Pick --base qwen2-audio-7b-instruct or drop --audio."
        )
        raise typer.Exit(code=2)

    if applied_result is not None:
        dlm_id = applied_result.dlm_id
    else:
        dlm_id = mint_ulid()
        if multimodal:
            _write_init_scaffold_multimodal(path, spec.key, dlm_id)
        elif audio:
            _write_init_scaffold_audio(path, spec.key, dlm_id)
        else:
            _write_init_scaffold(path, spec.key, dlm_id)

    # Create the store + write the initial manifest so `dlm show` sees
    # the license record and `dlm train` has a prior manifest to diff
    # against.
    from dlm.store.manifest import Manifest, save_manifest
    from dlm.store.paths import for_dlm

    store = for_dlm(dlm_id)
    store.ensure_layout()
    save_manifest(
        store.manifest,
        Manifest(
            dlm_id=dlm_id,
            base_model=spec.key,
            base_model_revision=spec.revision,
            source_path=path.resolve(),
            license_acceptance=acceptance,
        ),
    )
    if applied_result is not None:
        meta = applied_result.template.meta
        console.print(
            f"[green]init:[/green] wrote {path} from template "
            f"[bold]{meta.name}[/bold] ({meta.title}) — base {spec.key}."
        )
    else:
        console.print(f"[green]init:[/green] wrote {path}")


def _prompt_accept_license(console: object, base: str, license_url: str | None) -> bool:
    """Interactive y/N prompt for gated base-model license acceptance.

    Non-interactive runs (no TTY) return False; the caller surfaces the
    `--i-accept-license` flag pointer in that case.
    """
    import sys

    from rich.console import Console

    assert isinstance(console, Console)

    if not sys.stdin.isatty():
        return False

    console.print(
        f"[yellow]This base model ({base}) requires accepting the upstream license.[/yellow]"
    )
    if license_url:
        console.print(f"  Review the license at: {license_url}")
    console.print("Accept and continue? [y/N]: ", end="")
    try:
        answer = input().strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def _write_init_scaffold(path: Path, base_model_key: str, dlm_id: str) -> None:
    """Write a minimal-but-valid .dlm file at `path`.

    Body has one PROSE paragraph + a commented instruction section so
    users see both section shapes on first open.
    """
    scaffold = f"""---
dlm_id: {dlm_id}
dlm_version: 1
base_model: {base_model_key}
---

# Your document title

Write prose here. It will train via continued pretraining (CPT) loss.

::instruction::

### Q
Your example question.

### A
Your example answer.
"""
    path.write_text(scaffold, encoding="utf-8")


def _write_init_scaffold_multimodal(path: Path, base_model_key: str, dlm_id: str) -> None:
    """Write a VL-shaped .dlm file at `path`.

    Body shows the `::image::` attribute fence + a caption so users
    see the v10 grammar on first open. The placeholder path
    `figures/your-image.png` is deliberately non-existent — first
    `dlm train` will refuse with a clear file-missing error, prompting
    the user to drop a real image in. This is friendlier than
    committing an inert sample that users might not notice isn't theirs.

    `dlm_version: 10` because IMAGE sections require schema v10.
    """
    scaffold = f"""---
dlm_id: {dlm_id}
dlm_version: 10
base_model: {base_model_key}
---

# Your document title

Write prose here. It will train via continued pretraining (CPT) loss.

::image path="figures/your-image.png" alt="short description"::
Caption text describing the image. Training rows bundle the image
with this caption as `<image>\\n<caption>`.

::instruction::

### Q
What is in this image?

### A
Describe what the image shows.
"""
    path.write_text(scaffold, encoding="utf-8")


def _write_init_scaffold_audio(path: Path, base_model_key: str, dlm_id: str) -> None:
    """Write an audio-shaped .dlm file at `path`.

    Body shows the `::audio::` attribute fence with the sibling-
    transcript-friendly `transcript="..."` form so users see the v11
    grammar on first open. The placeholder path `clips/your-clip.wav`
    is deliberately non-existent — first `dlm train` refuses with a
    clear file-missing error rather than silently training on an inert
    sample.

    `dlm_version: 11` because AUDIO sections require schema v11.
    """
    scaffold = f"""---
dlm_id: {dlm_id}
dlm_version: 11
base_model: {base_model_key}
---

# Your document title

Write prose here. It will train via continued pretraining (CPT) loss.

::audio path="clips/your-clip.wav" transcript="Transcript of the audio clip."::

::instruction::

### Q
What was said in this recording?

### A
Describe what you hear in the audio.
"""
    path.write_text(scaffold, encoding="utf-8")


def train_cmd(
    path: Annotated[
        Path,
        typer.Argument(
            help=(
                ".dlm file to train. Or a directory — when passed a directory, "
                "`dlm train` auto-scaffolds `<dir>/.dlm/corpus.dlm` on first run "
                "(with --base) and reuses it on subsequent runs."
            ),
        ),
    ],
    resume: Annotated[bool, typer.Option("--resume", help="Resume from last checkpoint.")] = False,
    fresh: Annotated[bool, typer.Option("--fresh", help="Discard prior adapter state.")] = False,
    seed: Annotated[int | None, typer.Option("--seed", help="Override training seed.")] = None,
    max_steps: Annotated[int | None, typer.Option("--max-steps", help="Cap step count.")] = None,
    phase: Annotated[
        str,
        typer.Option(
            "--phase",
            help=(
                "Which training phases to run: 'sft' (supervised only), "
                "'preference' (DPO/ORPO only — requires a prior SFT "
                "adapter), or 'all' (SFT then preference when enabled). "
                "The preference method (dpo / orpo) comes from "
                "training.preference.method in the frontmatter."
            ),
        ),
    ] = "all",
    i_accept_license: Annotated[
        bool,
        typer.Option(
            "--i-accept-license",
            help="Accept the base model's license (required for gated bases like llama-3.2).",
        ),
    ] = False,
    strict_lock: Annotated[
        bool,
        typer.Option(
            "--strict-lock",
            help="Fail on any dlm.lock drift, including version warns.",
        ),
    ] = False,
    update_lock: Annotated[
        bool,
        typer.Option(
            "--update-lock",
            help="Overwrite dlm.lock without validating prior entries.",
        ),
    ] = False,
    ignore_lock: Annotated[
        bool,
        typer.Option(
            "--ignore-lock",
            help="Skip dlm.lock validation and don't write a new lock.",
        ),
    ] = False,
    strict_metrics: Annotated[
        bool,
        typer.Option(
            "--strict-metrics",
            help="Promote metrics SQLite write failures to hard errors.",
        ),
    ] = False,
    no_mined: Annotated[
        bool,
        typer.Option(
            "--no-mined",
            help=(
                "Exclude auto-mined preference sections from the preference "
                "phase, including replay-sampled mined pairs. Hand-authored "
                "`::preference::` sections still train normally."
            ),
        ),
    ] = False,
    gpus: Annotated[
        str | None,
        typer.Option(
            "--gpus",
            help=(
                "Multi-GPU training. `all` uses every visible CUDA device; "
                "`N` uses the first N; `0,1` selects exact device ids. "
                "Dispatches to `accelerate launch` when >1 device is "
                "selected. Omit for single-process training."
            ),
        ),
    ] = None,
    watch: Annotated[
        bool,
        typer.Option(
            "--watch",
            help=(
                "Save-to-train mode. After an initial train, block on "
                "filesystem events and run incremental retrains "
                "(mode=resume, step-capped) on each settled save. Ctrl-C "
                "exits cleanly between cycles."
            ),
        ),
    ] = False,
    watch_max_steps: Annotated[
        int,
        typer.Option(
            "--watch-max-steps",
            help="Per-cycle step cap for --watch. Default 100 keeps cycles responsive.",
        ),
    ] = 100,
    watch_debounce_ms: Annotated[
        int,
        typer.Option(
            "--watch-debounce-ms",
            help="Quiet interval (ms) before a burst of saves triggers a retrain.",
        ),
    ] = 400,
    watch_repl: Annotated[
        bool,
        typer.Option(
            "--repl",
            help=(
                "With --watch: also open the REPL so prompts reflect the "
                "latest adapter. **Scaffolded** — threading integration "
                "is untestable without a two-process harness; emit a "
                "not-yet-implemented refusal and exit 2."
            ),
        ),
    ] = False,
    base: Annotated[
        str | None,
        typer.Option(
            "--base",
            help=(
                "Base model key for auto-scaffold. Required on first run when "
                "`path` is a directory without an existing .dlm/ config. "
                "Accepts registry keys (smollm2-135m, qwen2.5-coder-1.5b, ...) "
                "or `hf:<org>/<name>` for off-registry models."
            ),
        ),
    ] = None,
    include: Annotated[
        list[str] | None,
        typer.Option(
            "--include",
            help=(
                "Glob pattern for files to train on (auto-scaffold only). "
                "Repeatable. Default: '**/*' with --recursive, '*' without. "
                "Examples: '**/*.py', '**/*.f90', '**/*.{md,rst}'."
            ),
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude",
            help=(
                "Glob pattern for files to skip (auto-scaffold only). "
                "Repeatable. Defaults (secrets, VCS, lockfiles, binaries) "
                "apply on top via the descent protocol."
            ),
        ),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive/--no-recursive",
            "-r/-R",
            help=(
                "Auto-scaffold include patterns descend into subdirectories. "
                "Default True. --no-recursive limits the default include to "
                "top-level files only."
            ),
        ),
    ] = True,
    name: Annotated[
        str,
        typer.Option(
            "--name",
            help=(
                "Adapter name for auto-scaffold → `<dir>/.dlm/<name>.dlm`. "
                "Default 'corpus'. Lets a single tree host multiple adapters."
            ),
        ),
    ] = "corpus",
    policy: Annotated[
        str,
        typer.Option(
            "--policy",
            help=(
                "Auto-scaffold sources_policy: 'strict' (default; confines "
                "training to the target directory) or 'permissive' (allows "
                "absolute paths anywhere)."
            ),
        ),
    ] = "strict",
    rescaffold: Annotated[
        bool,
        typer.Option(
            "--rescaffold",
            help=(
                "Rewrite an existing scaffolded .dlm in place with the new "
                "--base/--include/--exclude/--policy flags. Keeps the same "
                "dlm_id (store stays intact). Without it, re-running with "
                "frontmatter-editing flags refuses to shadow-edit."
            ),
        ),
    ] = False,
    listen_rpc: Annotated[
        str | None,
        typer.Option(
            "--listen-rpc",
            help=(
                "Open a JSON-RPC endpoint at <host:port> (e.g. `127.0.0.1:7429`) "
                "that accepts `inject_probe` pushes from sway-style eval "
                "harnesses. Probes enter the queue and drain at the next "
                "training-cycle boundary. Requires --watch or --max-cycles. "
                "Bearer token from DLM_PROBE_TOKEN."
            ),
        ),
    ] = None,
    max_cycles: Annotated[
        int,
        typer.Option(
            "--max-cycles",
            help=(
                "Convergence stop for --listen-rpc without --watch: cap the "
                "probe-driven retrain loop at N cycles. Ignored without "
                "--listen-rpc."
            ),
        ),
    ] = 0,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help=(
                "Opt out of the tokenized-section cache for this run. By "
                "default, `dlm train` pre-tokenizes directive-sourced rows "
                "via ~/.dlm/store/<id>/tokenized-cache/ so subsequent runs "
                "on the same corpus skip re-tokenization. Use this to "
                "bypass the cache for debugging or to compare cached vs "
                "uncached training determinism."
            ),
        ),
    ] = False,
    skip_export_probes: Annotated[
        bool,
        typer.Option(
            "--skip-export-probes",
            help=(
                "Skip the llama.cpp / GGUF-conversion probes so brand-new "
                "architectures (not yet in our vendored llama.cpp) can still "
                "be used for training + HF inference. Forfeits `dlm export` "
                "to Ollama until the vendored copy catches up. Mirrors the "
                "flag of the same name on `dlm init`."
            ),
        ),
    ] = False,
) -> None:
    """Train / retrain a .dlm against its base model."""
    import sqlite3
    import sys

    from rich.console import Console

    from dlm.base_models import GatedModelError
    from dlm.base_models import resolve as resolve_base_model
    from dlm.doc.errors import DlmParseError
    from dlm.doc.parser import parse_file
    from dlm.hardware import doctor
    from dlm.lock import LockMode, LockValidationError
    from dlm.store.paths import for_dlm
    from dlm.train import (
        DiskSpaceError,
        OOMError,
        ResumeIntegrityError,
        TrainingError,
    )
    from dlm.train.preference import (
        DpoPhaseError,
        NoPreferenceContentError,
        PriorAdapterRequiredError,
    )
    from dlm.train.preference.phase_orchestrator import Phase, run_phases

    console = Console(stderr=True)

    if phase not in ("sft", "preference", "all"):
        console.print(f"[red]error:[/red] --phase must be one of sft|preference|all, got {phase!r}")
        raise typer.Exit(code=2)
    phase_literal: Phase = phase  # type: ignore[assignment]

    if resume and fresh:
        console.print("[red]error:[/red] --resume and --fresh are mutually exclusive")
        raise typer.Exit(code=2)
    mode: Literal["fresh", "resume"] = "resume" if resume else "fresh"

    # --gpus dispatches to accelerate launch when >1 device is
    # selected. The single-GPU path falls through to the existing
    # in-process trainer; a bare `--gpus 1` is a no-op (users can use
    # it to lock the visible device set via CUDA_VISIBLE_DEVICES
    # without spawning a subprocess).
    if gpus is not None:
        # Resolve mixed_precision from capabilities so bf16-incapable
        # CUDA GPUs (SM<8.0) don't trip the `accelerate launch`
        # default. `probe()` is cheap and runs in the launcher-side
        # process only; each rank re-probes via `doctor()` later.
        from dlm.hardware.capabilities import probe as _probe_caps

        _caps = _probe_caps()
        _mp = "bf16" if _caps.supports_bf16 else "fp16"
        exit_code = _maybe_dispatch_multi_gpu(gpus, sys.argv, console, mixed_precision=_mp)
        if exit_code is not None:
            raise typer.Exit(code=exit_code)

    # Mutual-exclusion gate for the three lock flags. Exactly one (or
    # zero) may be set — silently ignoring a conflicting pair would
    # mask operator intent.
    lock_flag_count = sum((strict_lock, update_lock, ignore_lock))
    if lock_flag_count > 1:
        console.print(
            "[red]error:[/red] --strict-lock / --update-lock / --ignore-lock "
            "are mutually exclusive",
        )
        raise typer.Exit(code=2)
    lock_mode: LockMode = "default"
    if strict_lock:
        lock_mode = "strict"
    elif update_lock:
        lock_mode = "update"
    elif ignore_lock:
        lock_mode = "ignore"

    # `--no-cache` bypasses the tokenized-section cache for this run.
    # Plumbed as an env var because the trainer's pre-tokenize helper
    # already reads one — the CLI flag is a discoverable surface over
    # the same switch. Rolling the flag into `TrainingPlan` is a
    # deferred refactor; the env var is sufficient for the user-facing
    # contract and survives `accelerate launch` re-invocations.
    if no_cache:
        from dlm.train.cache import set_disable_flag

        set_disable_flag("--no-cache")

    if policy not in ("permissive", "strict"):
        console.print(
            f"[red]error:[/red] --policy must be 'permissive' or 'strict', got {policy!r}"
        )
        raise typer.Exit(code=2)
    policy_literal: Literal["permissive", "strict"] = policy  # type: ignore[assignment]

    # --listen-rpc requires a loop to drain the queue — either --watch
    # (file-change cycles) or --max-cycles N (bounded retrain loop).
    # Without one, the server would accept probes that never train. We
    # also need the bearer token up front so the user sees the refusal
    # before we spend time downloading weights.
    rpc_config: tuple[str, int, str] | None = None
    if listen_rpc is not None:
        if not watch and max_cycles <= 0:
            console.print(
                "[red]error:[/red] --listen-rpc requires --watch or --max-cycles N "
                "(the probe queue needs a drain cadence)"
            )
            raise typer.Exit(code=2)
        token = os.environ.get("DLM_PROBE_TOKEN", "").strip()
        if not token:
            console.print(
                "[red]error:[/red] --listen-rpc needs a bearer token; "
                "export DLM_PROBE_TOKEN=<secret>"
            )
            raise typer.Exit(code=2)
        host, _, port_s = listen_rpc.rpartition(":")
        if not host or not port_s:
            console.print(f"[red]error:[/red] --listen-rpc expects host:port, got {listen_rpc!r}")
            raise typer.Exit(code=2)
        try:
            port = int(port_s)
        except ValueError:
            console.print(f"[red]error:[/red] --listen-rpc port must be an integer, got {port_s!r}")
            raise typer.Exit(code=2) from None
        rpc_config = (host, port, token)

    # Directory targets auto-scaffold `<dir>/.dlm/corpus.dlm` (or
    # reuse an existing one). After this block, `path` always points
    # at an actual `.dlm` file that the rest of the flow can parse.
    if path.is_dir():
        from dlm.cli.scaffold import ScaffoldError, scaffold_train_target

        try:
            scaffold_result = scaffold_train_target(
                path,
                base=base,
                include=tuple(include or ()),
                exclude=tuple(exclude or ()),
                recursive=recursive,
                name=name,
                policy=policy_literal,
                rescaffold=rescaffold,
            )
        except ScaffoldError as exc:
            console.print(f"[red]scaffold:[/red] {exc.message}")
            raise typer.Exit(code=1) from exc

        if scaffold_result.scaffolded:
            console.print(
                f"[cyan]scaffolded:[/cyan] {scaffold_result.dlm_path} "
                f"(dlm_id={scaffold_result.dlm_id})"
            )
        path = scaffold_result.dlm_path

    try:
        parsed = parse_file(path)
    except (DlmParseError, OSError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    try:
        spec = resolve_base_model(
            parsed.frontmatter.base_model,
            accept_license=i_accept_license,
            skip_export_probes=skip_export_probes,
        )
    except GatedModelError as exc:
        console.print(f"[red]license:[/red] base model {parsed.frontmatter.base_model!r} is gated.")
        if exc.license_url:
            console.print(f"  review the license at: {exc.license_url}")
        console.print(
            "  re-run with [bold]--i-accept-license[/bold] once you have accepted. "
            "Acceptance will be persisted in the store manifest."
        )
        raise typer.Exit(code=1) from exc
    # Detect the DDP world_size set by `accelerate launch`
    # (WORLD_SIZE env var) and thread it into the doctor so the plan's
    # effective_batch_size reflects the rank count. Single-process
    # runs read 1 and the plan math is unchanged.
    from dlm.train.distributed import detect_world_size

    ws = detect_world_size()
    doctor_result = doctor(
        training_config=parsed.frontmatter.training,
        base_params=spec.params,
        seq_len=min(parsed.frontmatter.training.sequence_len, spec.effective_context_length),
        world_size=ws,
    )
    plan = doctor_result.plan
    if plan is None:
        console.print(
            "[red]doctor:[/red] no viable training plan for this host. "
            "Run `dlm doctor` for details."
        )
        raise typer.Exit(code=1)

    store = for_dlm(parsed.frontmatter.dlm_id)
    store.ensure_layout()

    # `dlm init` writes a manifest as part of store provisioning. Mirror
    # that manifest write here when the store layout exists but has no
    # manifest yet — covers two flows:
    #   - auto-scaffold via `dlm train <dir>` on a fresh directory
    #   - hand-authored .dlm with a fresh ULID that never went through
    #     `dlm init` (e.g. authored via the LSP / VSCode extension)
    # License acceptance has already been validated upstream by this
    # point, so we just record it.
    if not store.manifest.exists():
        from dlm.base_models import is_gated
        from dlm.base_models.license import require_acceptance
        from dlm.store.manifest import Manifest, save_manifest

        acceptance = (
            require_acceptance(spec, accept_license=True, via="cli_flag")
            if is_gated(spec)
            else None
        )
        save_manifest(
            store.manifest,
            Manifest(
                dlm_id=parsed.frontmatter.dlm_id,
                base_model=spec.key,
                base_model_revision=spec.revision,
                source_path=path.resolve(),
                license_acceptance=acceptance,
            ),
        )

    from dlm.modality import ModalityError

    try:
        phase_results = run_phases(
            store,
            parsed,
            spec,
            plan,
            phase=phase_literal,
            mode=mode,
            seed=seed,
            max_steps=max_steps,
            lock_mode=lock_mode,
            capabilities=doctor_result.capabilities,
            world_size=ws,
            strict_metrics=strict_metrics,
            include_auto_mined=not no_mined,
        )
    except sqlite3.Error as exc:
        console.print(f"[red]metrics:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except LockValidationError as exc:
        console.print(f"[red]lock:[/red] {exc}")
        console.print(
            "  Re-run with [bold]--update-lock[/bold] to accept the drift or "
            "[bold]--ignore-lock[/bold] to continue without persisting a new lock."
        )
        raise typer.Exit(code=1) from exc
    except DiskSpaceError as exc:
        console.print(f"[red]disk:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except OOMError as exc:
        from dlm.train import format_oom_message

        console.print(
            format_oom_message(
                step=exc.step,
                peak_bytes=exc.peak_bytes,
                free_at_start_bytes=exc.free_at_start_bytes,
                current_grad_accum=exc.current_grad_accum,
                recommended_grad_accum=exc.recommended_grad_accum,
            )
        )
        raise typer.Exit(code=1) from exc
    except ResumeIntegrityError as exc:
        console.print(f"[red]resume:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except (NoPreferenceContentError, PriorAdapterRequiredError) as exc:
        console.print(f"[red]dpo:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except DpoPhaseError as exc:
        console.print(f"[red]dpo:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except TrainingError as exc:
        console.print(f"[red]training:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ModalityError as exc:
        console.print(f"[red]training:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not phase_results:
        console.print(
            "[yellow]no-op:[/yellow] nothing to train for the requested phase. "
            "Check that the document has the section types the phase consumes "
            "(prose/instruction for SFT, preference for DPO)."
        )
        raise typer.Exit(code=0)

    for pr in phase_results:
        result = pr.result
        console.print(
            f"[green]{pr.phase}:[/green] v{result.adapter_version:04d} "
            f"({result.steps} steps, seed={result.seed}, "
            f"determinism={result.determinism.class_})"
        )
        console.print(f"adapter: {result.adapter_path}")
        console.print(f"log:     {result.log_path}")
    # Final-train-loss stdout line mirrors the last phase so existing
    # downstream scripts keep working.
    result = phase_results[-1].result
    if result.final_train_loss is not None:
        sys.stdout.write(f"{result.final_train_loss}\n")

    # --watch keeps the training context alive and re-runs incremental
    # cycles on file change. Entered AFTER the initial train so the
    # loop resumes from a real committed adapter.
    if watch:
        if watch_repl:
            console.print(
                "[red]train:[/red] --watch --repl is scaffolded but not yet "
                "implemented. The threaded REPL bridge needs a test "
                "harness we don't have in CI today."
            )
            raise typer.Exit(code=2)

        from dlm.watch.loop import run_watch
        from dlm.watch.status import WatchStatus, render_status

        status = WatchStatus(doc_path=str(path), sections=len(parsed.sections))

        # Start the probe-RPC server if --listen-rpc was requested. The
        # queue is exposed; end-to-end flow into `build_dataset` at the
        # next cycle boundary is the follow-up consumer task — for now
        # the server accepts and buffers probes so sway sinks can be
        # wired + tested against a live endpoint.
        rpc_server = None
        probe_queue = None
        if rpc_config is not None:
            from dlm.train.inject import InjectedProbeQueue
            from dlm.train.rpc import ProbeRpcServer

            rpc_host, rpc_port, rpc_token = rpc_config
            probe_queue = InjectedProbeQueue()
            rpc_server = ProbeRpcServer(
                host=rpc_host, port=rpc_port, token=rpc_token, queue=probe_queue
            )
            rpc_server.start()
            bound_host, bound_port = rpc_server.address
            console.print(
                f"[dim]rpc:[/dim] listening on {bound_host}:{bound_port} "
                f"(queue capacity {probe_queue.capacity})"
            )

        console.print(
            f"[dim]watch:[/dim] {render_status(status)}; "
            f"max_steps={watch_max_steps}, debounce_ms={watch_debounce_ms}"
        )

        def _log_cycle(result_: object) -> None:
            from dlm.watch.loop import CycleResult

            assert isinstance(result_, CycleResult)
            if result_.ran and result_.run_result is not None:
                status.mark_cycle_done(
                    train_loss=result_.run_result.final_train_loss,
                    val_loss=result_.run_result.final_val_loss,
                    steps=result_.run_result.steps,
                    coalesced=1,
                )
                console.print(f"[dim]watch:[/dim] {render_status(status)}")
            else:
                console.print("[dim]watch:[/dim] no new content, skipping retrain")

        try:
            exit_code = run_watch(
                doc_path=path,
                store=store,
                spec=spec,
                plan=plan,
                max_steps=watch_max_steps,
                debounce_ms=watch_debounce_ms,
                on_cycle=_log_cycle,
                drain_probes=probe_queue.drain if probe_queue is not None else None,
            )
        except KeyboardInterrupt:
            if rpc_server is not None:
                rpc_server.stop()
            console.print("[dim]watch:[/dim] Ctrl-C received, exiting")
            raise typer.Exit(code=0)  # noqa: B904
        finally:
            if rpc_server is not None:
                rpc_server.stop()
        raise typer.Exit(code=exit_code)

    # --max-cycles without --watch: the bounded-loop cycle driver is
    # the next consumer-side integration step. Accept the flags, refuse
    # execution until the loop lands.
    if rpc_config is not None and not watch:
        console.print(
            "[red]train:[/red] --listen-rpc --max-cycles (without --watch) is "
            "scaffolded; the bounded cycle loop is the follow-up. Use "
            "--watch for now."
        )
        raise typer.Exit(code=2)


def _maybe_dispatch_multi_gpu(
    gpus_flag: str,
    argv: list[str],
    console: object,
    *,
    mixed_precision: str = "bf16",
) -> int | None:
    """Resolve `--gpus`; if multi-GPU, spawn accelerate launch and return its exit code.

    Returns None when the resolved world_size is 1 — caller falls
    through to the in-process trainer. Returns an int exit code when
    the launcher ran, so the caller can `raise typer.Exit(code=...)`.
    """
    from rich.console import Console

    from dlm.train.distributed import UnsupportedGpuSpecError, launch_multi_gpu, parse_gpus

    assert isinstance(console, Console)

    try:
        spec = parse_gpus(gpus_flag)
    except UnsupportedGpuSpecError as exc:
        console.print(f"[red]train:[/red] {exc}")
        return 2

    try:
        import torch

        device_count = int(torch.cuda.device_count())
    except Exception:  # pragma: no cover - torch probing has many failure modes
        device_count = 0

    try:
        device_ids = spec.resolve(device_count)
    except UnsupportedGpuSpecError as exc:
        console.print(f"[red]train:[/red] {exc}")
        return 2

    if len(device_ids) < 2:
        # Single-GPU (or --gpus 1) — no subprocess needed. Caller
        # continues with the in-process path.
        return None

    # Forward the original argv minus `--gpus` / `--gpus=...`; the
    # worker entry strips it defensively too, but we drop it here so
    # the launched accelerate cmd carries exactly the intended args.
    cli_args = _strip_gpus_from_argv(argv)
    console.print(
        f"[dim]train:[/dim] dispatching to accelerate launch on devices {list(device_ids)} "
        f"(mixed_precision={mixed_precision})"
    )
    return launch_multi_gpu(device_ids, cli_args, mixed_precision=mixed_precision)


def _strip_gpus_from_argv(argv: list[str]) -> list[str]:
    """Drop `--gpus <v>` / `--gpus=<v>` from raw sys.argv (launcher side).

    Skips argv[0] (script path) — `accelerate launch -m <entry>`
    provides the rank entrypoint separately, so the launcher forwards
    argv[1:] minus the multi-GPU flag. Delegates to the shared
    `strip_gpus_flag` helper.
    """
    from dlm.train.distributed.gpus import strip_gpus_flag

    return strip_gpus_flag(argv, skip_argv0=True)


def prompt_cmd(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help=".dlm file to query.")],
    query: Annotated[str | None, typer.Argument(help="One-shot prompt (omit for stdin).")] = None,
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", help="Max new tokens to generate."),
    ] = 256,
    temp: Annotated[
        float,
        typer.Option("--temp", help="Sampling temperature. `0.0` = greedy decoding."),
    ] = 0.7,
    top_p: Annotated[
        float | None,
        typer.Option(
            "--top-p",
            help="Top-p sampling cutoff. Omit to disable nucleus sampling.",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Log resolved InferencePlan.")] = False,
    adapter: Annotated[
        str | None,
        typer.Option(
            "--adapter",
            help=(
                "Named adapter to prompt against. Required on multi-adapter "
                "documents; rejected on single-adapter documents."
            ),
        ),
    ] = None,
    gate: Annotated[
        str,
        typer.Option(
            "--gate",
            help=(
                "Learned adapter gate. `auto` (default) uses the "
                "gate when one exists in the store; `off` forces uniform "
                "weights across declared adapters. Ignored when --adapter "
                "explicitly pins a single adapter."
            ),
        ),
    ] = "auto",
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help=(
                "Inference backend: `auto` (default) picks MLX on Apple "
                "Silicon, else PyTorch. Force with `pytorch` or `mlx`. "
                "MLX requires `uv sync --extra mlx` on darwin-arm64."
            ),
        ),
    ] = "auto",
    image: Annotated[
        list[Path] | None,
        typer.Option(
            "--image",
            help=(
                "Attach an image file to the prompt. Repeat for multiple "
                "images; each expands to the base's image-token placeholder. "
                "Requires a vision-language base."
            ),
        ),
    ] = None,
    audio: Annotated[
        list[Path] | None,
        typer.Option(
            "--audio",
            help=(
                "Attach an audio file (.wav/.flac/.ogg) to the prompt. "
                "Repeat for multiple clips; each expands to the base's "
                "audio-token placeholder. Requires an audio-language base "
                "(for example Qwen2-Audio-7B-Instruct)."
            ),
        ),
    ] = None,
) -> None:
    """Run inference against the trained adapter."""
    import sys

    from rich.console import Console

    from dlm.base_models import resolve as resolve_base_model
    from dlm.doc.parser import parse_file
    from dlm.hardware import doctor
    from dlm.inference import AdapterNotFoundError
    from dlm.inference.backends import (
        UnsupportedBackendError,
        build_backend,
        select_backend,
    )
    from dlm.store.paths import for_dlm

    console = Console(stderr=True)

    if backend not in ("auto", "pytorch", "mlx"):
        console.print(
            f"[red]prompt:[/red] --backend must be `auto`, `pytorch`, or `mlx` (got {backend!r})."
        )
        raise typer.Exit(code=2)

    # Typer passes None when the option was never given; normalize early so
    # downstream branching can just check truthiness + len().
    image_paths: list[Path] = list(image or [])
    audio_paths: list[Path] = list(audio or [])
    if image_paths and audio_paths:
        console.print(
            "[red]prompt:[/red] --image and --audio cannot be combined "
            "(each targets a different modality)."
        )
        raise typer.Exit(code=2)

    from dlm.base_models import GatedModelError

    parsed = parse_file(path)
    adapters_declared = parsed.frontmatter.training.adapters
    if adapter is not None:
        if adapters_declared is None:
            console.print(
                "[red]prompt:[/red] --adapter is only valid on multi-adapter "
                "documents (this doc does not declare `training.adapters`)."
            )
            raise typer.Exit(code=2)
        if adapter not in adapters_declared:
            declared = sorted(adapters_declared)
            console.print(
                f"[red]prompt:[/red] --adapter {adapter!r} is not declared (declared: {declared})."
            )
            raise typer.Exit(code=2)

    if gate not in ("auto", "off"):
        console.print(f"[red]prompt:[/red] --gate must be `auto` or `off`, got {gate!r}.")
        raise typer.Exit(code=2)
    # --adapter explicitly pins a single adapter — gate routing is moot.
    # We silently ignore --gate in that case (the flag has a non-default
    # value only when the user cares, and pairing it with --adapter is
    # not an error, just a no-op).

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

    # --- VL path -------------------------------------------------------
    # The VL branch has its own model / processor / adapter loader and
    # its own generate function. `--image` and vision-language bases
    # must appear together; each alone is a usage error.
    from dlm.modality import modality_for

    dispatch = modality_for(spec)
    from click.core import ParameterSource

    if ctx.get_parameter_source("temp") == ParameterSource.DEFAULT:
        temp = spec.suggested_prompt_temperature
    if image_paths and not dispatch.accepts_images:
        console.print(
            f"[red]prompt:[/red] --image is only valid with vision-language bases; "
            f"base {spec.key!r} is modality='{spec.modality}'."
        )
        raise typer.Exit(code=2)
    if dispatch.accepts_images and not image_paths:
        console.print(
            f"[red]prompt:[/red] base {spec.key!r} is vision-language; "
            "pass at least one --image PATH to prompt it."
        )
        raise typer.Exit(code=2)
    if dispatch.accepts_images:
        _dispatch_vl_prompt(
            console=console,
            spec=spec,
            store=store,
            caps=caps,
            adapter_name=adapter,
            image_paths=image_paths,
            query=query,
            max_tokens=max_tokens,
            temp=temp,
            top_p=top_p,
            verbose=verbose,
        )
        return

    # --- Audio path ----------------------------------------------------
    if audio_paths and not dispatch.accepts_audio:
        console.print(
            f"[red]prompt:[/red] --audio is only valid with audio-language bases; "
            f"base {spec.key!r} is modality='{spec.modality}'."
        )
        raise typer.Exit(code=2)
    if dispatch.accepts_audio and not audio_paths:
        console.print(
            f"[red]prompt:[/red] base {spec.key!r} is audio-language; "
            "pass at least one --audio PATH to prompt it."
        )
        raise typer.Exit(code=2)
    if dispatch.accepts_audio:
        _dispatch_audio_prompt(
            console=console,
            spec=spec,
            store=store,
            caps=caps,
            adapter_name=adapter,
            audio_paths=audio_paths,
            query=query,
            max_tokens=max_tokens,
            temp=temp,
            top_p=top_p,
            verbose=verbose,
            auto_resample=parsed.frontmatter.training.audio.auto_resample,
        )
        return

    try:
        backend_name = select_backend(backend, caps)  # type: ignore[arg-type]
    except UnsupportedBackendError as exc:
        console.print(f"[red]prompt:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    backend_obj = build_backend(backend_name, caps)

    if verbose:
        console.print(f"[dim]backend:[/dim] {backend_name}")

    try:
        backend_obj.load(spec, store, adapter_name=adapter)
    except AdapterNotFoundError as exc:
        console.print(f"[red]prompt:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if query is None:
        query = sys.stdin.read().strip()
    if not query:
        console.print("[red]prompt:[/red] empty query (pass a string or pipe on stdin)")
        raise typer.Exit(code=2)

    response = backend_obj.generate(
        query,
        max_new_tokens=max_tokens,
        temperature=temp,
        top_p=top_p,
    )
    sys.stdout.write(response + "\n")


def _dispatch_vl_prompt(  # pragma: no cover
    *,
    console: Any,
    spec: Any,
    store: Any,
    caps: Any,
    adapter_name: str | None,
    image_paths: list[Path],
    query: str | None,
    max_tokens: int,
    temp: float,
    top_p: float | None,
    verbose: bool,
) -> None:
    """Run the VL generate path. Keeps `prompt_cmd` readable.

    Pragma'd from unit coverage because it calls the VL HF stack.
    Covered by the slow-marked vision-language integration test (T12).
    """
    import sys

    import typer

    from dlm.inference import (
        AdapterNotFoundError,
        generate_vl,
        load_for_vl_inference,
        load_images,
    )
    from dlm.modality import ProcessorContractError

    if verbose:
        console.print("[dim]vl-backend:[/dim] pytorch (AutoModelForImageTextToText)")

    try:
        loaded = load_for_vl_inference(store, spec, caps, adapter_name=adapter_name)
    except AdapterNotFoundError as exc:
        console.print(f"[red]prompt:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ProcessorContractError as exc:
        console.print(f"[red]prompt:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        images = load_images(image_paths)
    except FileNotFoundError as exc:
        console.print(f"[red]prompt:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    if query is None:
        query = sys.stdin.read().strip()
    if not query:
        console.print("[red]prompt:[/red] empty query (pass a string or pipe on stdin)")
        raise typer.Exit(code=2)

    # Every VL spec in the registry must declare a preprocessor plan
    # (schema validator); the fallback is defensive for the hf: escape
    # hatch, which could in principle skip one.
    image_token = "<image>"
    if spec.vl_preprocessor_plan is not None:
        image_token = spec.vl_preprocessor_plan.image_token

    response = generate_vl(
        loaded.model,
        loaded.processor,
        query,
        images,
        image_token=image_token,
        max_new_tokens=max_tokens,
        temperature=temp,
        top_p=top_p,
    )
    sys.stdout.write(response + "\n")


def _dispatch_audio_prompt(  # pragma: no cover
    *,
    console: Any,
    spec: Any,
    store: Any,
    caps: Any,
    adapter_name: str | None,
    audio_paths: list[Path],
    query: str | None,
    max_tokens: int,
    temp: float,
    top_p: float | None,
    verbose: bool,
    auto_resample: bool = False,
) -> None:
    """Run the audio-LM generate path. Keeps `prompt_cmd` readable.

    Pragma'd from unit coverage because it calls the audio HF stack.
    Covered by the slow-marked audio integration test (T12).
    """
    import sys

    import typer

    from dlm.inference import (
        AdapterNotFoundError,
        generate_audio,
        load_audios,
        load_for_audio_inference,
    )

    if verbose:
        console.print(f"[dim]audio-backend:[/dim] pytorch ({spec.architecture})")

    try:
        loaded = load_for_audio_inference(store, spec, caps, adapter_name=adapter_name)
    except AdapterNotFoundError as exc:
        console.print(f"[red]prompt:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if spec.audio_preprocessor_plan is None:
        # Defensive — every registry audio spec carries the plan, but
        # the hf: escape hatch could skip it.
        console.print(
            f"[red]prompt:[/red] base {spec.key!r} is audio-language "
            "but has no audio_preprocessor_plan; cannot resolve sample rate."
        )
        raise typer.Exit(code=2)

    target_sr = spec.audio_preprocessor_plan.sample_rate
    try:
        waveforms = load_audios(
            audio_paths,
            target_sample_rate=target_sr,
            auto_resample=auto_resample,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]prompt:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    except ValueError as exc:
        # Sample-rate mismatch — surface the actionable ffmpeg hint.
        console.print(f"[red]prompt:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    if query is None:
        query = sys.stdin.read().strip()
    if not query:
        console.print("[red]prompt:[/red] empty query (pass a string or pipe on stdin)")
        raise typer.Exit(code=2)

    audio_token = spec.audio_preprocessor_plan.audio_token

    response = generate_audio(
        loaded.model,
        loaded.processor,
        query,
        waveforms,
        audio_token=audio_token,
        sample_rate=target_sr,
        max_new_tokens=max_tokens,
        temperature=temp,
        top_p=top_p,
    )
    sys.stdout.write(response + "\n")


def export_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to export.")],
    target: Annotated[
        str,
        typer.Option(
            "--target",
            help="Export destination. Currently supported: ollama, llama-server, vllm, mlx-serve.",
        ),
    ] = "ollama",
    quant: Annotated[
        str | None,
        typer.Option("--quant", help="GGUF quant level (defaults to frontmatter)."),
    ] = None,
    merged: Annotated[
        bool,
        typer.Option("--merged", help="Merge the adapter into the base before export."),
    ] = False,
    dequantize: Annotated[
        bool,
        typer.Option(
            "--dequantize",
            help="Dequantize a QLoRA base to fp16 before merging.",
        ),
    ] = False,
    name: Annotated[str | None, typer.Option("--name", help="Ollama model name.")] = None,
    no_template: Annotated[
        bool,
        typer.Option("--no-template", help="Skip writing TEMPLATE into the Modelfile."),
    ] = False,
    no_smoke: Annotated[
        bool,
        typer.Option("--no-smoke", help="Register the export but skip the smoke prompt."),
    ] = False,
    no_imatrix: Annotated[
        bool,
        typer.Option(
            "--no-imatrix",
            help=(
                "Skip importance-matrix calibration. Default uses the "
                "replay corpus to calibrate k-quant quantization."
            ),
        ),
    ] = False,
    draft: Annotated[
        str | None,
        typer.Option(
            "--draft",
            help=(
                "Speculative-decoding draft model Ollama tag "
                "(e.g. qwen2.5:0.5b). Default uses the registered pair "
                "for this base; override here to pick a custom draft."
            ),
        ),
    ] = None,
    no_draft: Annotated[
        bool,
        typer.Option(
            "--no-draft",
            help="Suppress PARAMETER draft_model emission even when a pair is registered.",
        ),
    ] = False,
    skip_ollama: Annotated[
        bool,
        typer.Option(
            "--skip-ollama",
            help="Emit GGUFs + manifest only; do not touch the Ollama binary.",
        ),
    ] = False,
    adapter: Annotated[
        str | None,
        typer.Option(
            "--adapter",
            help=(
                "Named adapter to export. Required on multi-adapter "
                "documents; rejected on single-adapter documents."
            ),
        ),
    ] = None,
    adapter_mix: Annotated[
        str | None,
        typer.Option(
            "--adapter-mix",
            help=(
                "Weighted composition of named adapters, e.g. "
                "`knowledge:1.0,tone:0.5`. Mutually exclusive with --adapter. "
                "Multi-adapter docs only. LoRA-only; QLoRA requires "
                "--dequantize."
            ),
        ),
    ] = None,
    adapter_mix_method: Annotated[
        str,
        typer.Option(
            "--adapter-mix-method",
            help=(
                "PEFT combination strategy for --adapter-mix. `linear` "
                "(default) sums LoRA deltas; `svd` recomposes via SVD "
                "(higher fidelity, heavier compute). Only meaningful "
                "with --adapter-mix."
            ),
        ),
    ] = "linear",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Log each subprocess command as it launches."),
    ] = False,
    emit_sway_json: Annotated[
        bool,
        typer.Option(
            "--emit-sway-json",
            help=(
                "After the export, also write a ready-to-run sway.yaml "
                "(via dlm-sway autogen) into the export dir. Requires the "
                "[sway] extra: pip install 'dlm[sway]'."
            ),
        ),
    ] = False,
) -> None:
    """Export the adapter to a runtime target."""

    from rich.console import Console

    from dlm.base_models import GatedModelError, download_spec
    from dlm.base_models import resolve as resolve_base_model
    from dlm.doc.parser import parse_file
    from dlm.export import (
        ExportError,
        PreflightError,
        SubprocessError,
        UnknownExportTargetError,
        UnsafeMergeError,
        VendoringError,
        resolve_export_plan,
        run_export,
    )
    from dlm.export.ollama import (
        OllamaBinaryNotFoundError,
        OllamaCreateError,
        OllamaError,
        OllamaSmokeError,
        OllamaVersionError,
    )
    from dlm.export.quantize import run_checked
    from dlm.export.targets import (
        finalize_mlx_serve_export,
        finalize_vllm_export,
        prepare_llama_server_export,
        prepare_mlx_serve_export,
        prepare_vllm_export,
        resolve_target,
    )
    from dlm.store.paths import for_dlm

    console = Console(stderr=True)

    if draft is not None and no_draft:
        console.print("[red]error:[/red] --draft and --no-draft are mutually exclusive; pick one.")
        raise typer.Exit(code=2)
    if adapter is not None and adapter_mix is not None:
        console.print(
            "[red]export:[/red] --adapter and --adapter-mix are mutually exclusive; pick one."
        )
        raise typer.Exit(code=2)
    try:
        resolved_target = resolve_target(target)
    except UnknownExportTargetError as exc:
        console.print(f"[red]export:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    parsed = parse_file(path)
    adapters_declared = parsed.frontmatter.training.adapters
    if adapter is not None:
        if adapters_declared is None:
            console.print(
                "[red]export:[/red] --adapter is only valid on multi-adapter "
                "documents (this doc does not declare `training.adapters`)."
            )
            raise typer.Exit(code=2)
        if adapter not in adapters_declared:
            declared = sorted(adapters_declared)
            console.print(
                f"[red]export:[/red] --adapter {adapter!r} is not declared (declared: {declared})."
            )
            raise typer.Exit(code=2)

    mix_entries: list[tuple[str, float]] | None = None
    if adapter_mix is not None:
        from dlm.export.weighted_merge import (
            InvalidMixSpecError,
            parse_mix_spec,
            validate_mix_against_declared,
        )

        if adapters_declared is None:
            console.print(
                "[red]export:[/red] --adapter-mix is only valid on multi-adapter "
                "documents (this doc does not declare `training.adapters`)."
            )
            raise typer.Exit(code=2)
        if adapter_mix_method not in ("linear", "svd"):
            console.print(
                f"[red]export:[/red] --adapter-mix-method must be "
                f"`linear` or `svd`, got {adapter_mix_method!r}."
            )
            raise typer.Exit(code=2)
        try:
            entries = parse_mix_spec(adapter_mix)
            validate_mix_against_declared(entries, set(adapters_declared))
        except InvalidMixSpecError as exc:
            console.print(f"[red]export:[/red] {exc}")
            raise typer.Exit(code=2) from exc
        mix_entries = [(e.name, e.weight) for e in entries]

    store = for_dlm(parsed.frontmatter.dlm_id)

    # Gate-driven static mix: when the doc has an enabled gate and the
    # user didn't pass --adapter-mix / --adapter, freeze the learned
    # gate to per-adapter weights for the GGUF export path. Dynamic
    # routing only lives in the `dlm prompt` flow; the runtime can't
    # evaluate the torch gate, so we substitute the prior here. A CLI
    # --adapter-mix wins — users who know what they want get full
    # control.
    if mix_entries is None and adapter is None:
        from dlm.export.gate_fallback import resolve_and_announce

        resolution = resolve_and_announce(store, parsed)
        if resolution.entries is not None:
            mix_entries = resolution.entries
            for line in resolution.banner_lines:
                console.print(line)

    already_accepted = _previously_accepted(store.manifest)
    try:
        spec = resolve_base_model(parsed.frontmatter.base_model, accept_license=already_accepted)
    except GatedModelError as exc:
        console.print(f"[red]license:[/red] base model {parsed.frontmatter.base_model!r} is gated.")
        if exc.license_url:
            console.print(f"  review the license at: {exc.license_url}")
        console.print("  accept via `dlm train --i-accept-license` before exporting.")
        raise typer.Exit(code=1) from exc

    # Audio bases take HF-snapshot unconditionally — llama.cpp has no
    # audio-arch roadmap at our pinned tag — so branch early without
    # resolving a GGUF plan.
    from dlm.modality import modality_for

    export_dispatch = modality_for(spec)
    if resolved_target.name == "vllm" and export_dispatch.accepts_audio:
        console.print(
            "[red]export:[/red] --target vllm is not wired for audio-language "
            "documents yet; the current vllm export path only supports text bases."
        )
        raise typer.Exit(code=2)
    if resolved_target.name == "mlx-serve" and export_dispatch.accepts_audio:
        console.print(
            "[red]export:[/red] --target mlx-serve is not wired for audio-language "
            "documents yet; the current mlx-serve export path only supports text bases."
        )
        raise typer.Exit(code=2)
    if export_dispatch.accepts_audio:
        try:
            dispatch_result = export_dispatch.dispatch_export(
                store=store,
                spec=spec,
                adapter_name=adapter,
                quant=quant,
                merged=merged,
                adapter_mix_raw=adapter_mix,
            )
        except ExportError as exc:
            console.print(f"[red]export:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        assert dispatch_result is not None  # audio modality always returns a result
        for line in dispatch_result.banner_lines:
            console.print(line)
        return

    try:
        plan = resolve_export_plan(
            cli_quant=quant,
            cli_merged=merged,
            cli_dequantize=dequantize,
            cli_no_template=no_template,
            cli_ollama_name=name,
            cli_no_imatrix=no_imatrix,
            frontmatter_default_quant=parsed.frontmatter.export.default_quant,
        )
    except ValueError as exc:
        console.print(f"[red]export:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    store.ensure_layout()

    # VL bases: arch-probe + try single-file GGUF on SUPPORTED (with
    # fallback to HF-snapshot on refusal or subprocess failure). A
    # missing local base snapshot should not hard-fail the whole
    # export — the dispatcher can still emit the HF-snapshot path
    # without GGUF context.
    if resolved_target.name == "vllm" and export_dispatch.accepts_images:
        console.print(
            "[red]export:[/red] --target vllm is not wired for vision-language "
            "documents yet; the current vllm export path only supports text bases."
        )
        raise typer.Exit(code=2)
    if resolved_target.name == "mlx-serve" and export_dispatch.accepts_images:
        console.print(
            "[red]export:[/red] --target mlx-serve is not wired for vision-language "
            "documents yet; the current mlx-serve export path only supports text bases."
        )
        raise typer.Exit(code=2)
    if export_dispatch.accepts_images:
        gguf_emission_context = None
        try:
            cached_vl = download_spec(spec, local_files_only=True)
        except RuntimeError as exc:
            _ = exc
        else:
            gguf_emission_context = {
                "plan": plan,
                "cached_base_dir": cached_vl.path,
                "source_dlm_path": path.resolve(),
                "training_sequence_len": parsed.frontmatter.training.sequence_len,
                "dlm_version": f"v{parsed.frontmatter.dlm_version}",
            }
        try:
            dispatch_result = export_dispatch.dispatch_export(
                store=store,
                spec=spec,
                adapter_name=adapter,
                quant=quant,
                merged=merged,
                adapter_mix_raw=adapter_mix,
                gguf_emission_context=gguf_emission_context,
            )
        except ExportError as exc:
            console.print(f"[red]export:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        assert dispatch_result is not None  # VL modality always returns a result
        for line in dispatch_result.banner_lines:
            console.print(line)
        return

    try:
        cached = download_spec(spec, local_files_only=True)
    except RuntimeError as exc:
        console.print(
            f"[red]export:[/red] base model not in local cache — run `dlm train` first.\n  {exc}"
        )
        raise typer.Exit(code=1) from exc

    def _verbose_runner(cmd: Sequence[str]) -> object:
        console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
        return run_checked(cmd)

    adapter_path_override = None
    if mix_entries is not None:  # pragma: no cover - heavy path
        # Build the weighted-merged adapter into an ephemeral dir,
        # then feed the path to run_export as an override. The tmp
        # dir lives under the store's cache/ so it cleans up with
        # the rest of the store on `dlm pack`.
        from dlm.export.weighted_merge import MixEntry, build_and_stage

        entries_typed = [MixEntry(name=n, weight=w) for (n, w) in mix_entries]
        adapter_path_override = build_and_stage(
            store=store,
            spec=spec,
            cached_base_dir=cached.path,
            entries=entries_typed,
            combination_type=adapter_mix_method,  # type: ignore[arg-type]
        )

    if resolved_target.name == "vllm":
        ignored_flags: list[str] = []
        if quant is not None:
            ignored_flags.append("--quant")
        if merged:
            ignored_flags.append("--merged")
        if dequantize:
            ignored_flags.append("--dequantize")
        if no_template:
            ignored_flags.append("--no-template")
        if skip_ollama:
            ignored_flags.append("--skip-ollama")
        if no_imatrix:
            ignored_flags.append("--no-imatrix")
        if draft is not None:
            ignored_flags.append("--draft")
        if no_draft:
            ignored_flags.append("--no-draft")
        if ignored_flags:
            console.print(
                "[yellow]export:[/yellow] ignoring flags not applicable to "
                f"`--target vllm`: {', '.join(ignored_flags)}"
            )

        declared_adapter_names = tuple(adapters_declared.keys()) if adapters_declared else None
        try:
            vllm_result = prepare_vllm_export(
                store=store,
                spec=spec,
                served_model_name=name or f"dlm-{parsed.frontmatter.dlm_id.lower()}",
                training_sequence_len=parsed.frontmatter.training.sequence_len,
                adapter_name=adapter,
                adapter_path_override=adapter_path_override,
                declared_adapter_names=declared_adapter_names,
            )
        except ExportError as exc:
            console.print(f"[red]export:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        vllm_smoke = None if no_smoke else resolved_target.smoke_test(vllm_result)
        if vllm_smoke is not None and not vllm_smoke.ok:
            console.print(
                f"[red]smoke:[/red] {vllm_smoke.detail}\n"
                "  re-run with `--no-smoke` to skip the smoke test."
            )
            raise typer.Exit(code=1)

        manifest_path = finalize_vllm_export(
            store=store,
            spec=spec,
            prepared=vllm_result,
            smoke_output_first_line=None if vllm_smoke is None else vllm_smoke.detail,
            adapter_name=adapter,
            adapter_mix=mix_entries,
        )
        console.print(f"[green]exported:[/green] {vllm_result.export_dir}")
        console.print("target:  vllm")
        assert vllm_result.launch_script_path is not None
        assert vllm_result.config_path is not None
        console.print(f"launch:  {vllm_result.launch_script_path.name}")
        console.print(f"config:  {vllm_result.config_path.name}")
        console.print(f"manifest: {manifest_path.name}")
        if vllm_smoke is not None and vllm_smoke.detail:
            console.print(f"smoke:   {vllm_smoke.detail}")
        return

    if resolved_target.name == "mlx-serve":
        mlx_ignored_flags: list[str] = []
        if quant is not None:
            mlx_ignored_flags.append("--quant")
        if merged:
            mlx_ignored_flags.append("--merged")
        if dequantize:
            mlx_ignored_flags.append("--dequantize")
        if name is not None:
            mlx_ignored_flags.append("--name")
        if no_template:
            mlx_ignored_flags.append("--no-template")
        if skip_ollama:
            mlx_ignored_flags.append("--skip-ollama")
        if no_imatrix:
            mlx_ignored_flags.append("--no-imatrix")
        if draft is not None:
            mlx_ignored_flags.append("--draft")
        if no_draft:
            mlx_ignored_flags.append("--no-draft")
        if mlx_ignored_flags:
            console.print(
                "[yellow]export:[/yellow] ignoring flags not applicable to "
                f"`--target mlx-serve`: {', '.join(mlx_ignored_flags)}"
            )

        declared_adapter_names = tuple(adapters_declared.keys()) if adapters_declared else None
        try:
            mlx_serve_result = prepare_mlx_serve_export(
                store=store,
                spec=spec,
                adapter_name=adapter,
                adapter_path_override=adapter_path_override,
                declared_adapter_names=declared_adapter_names,
            )
        except ExportError as exc:
            console.print(f"[red]export:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        mlx_serve_smoke = None if no_smoke else resolved_target.smoke_test(mlx_serve_result)
        if mlx_serve_smoke is not None and not mlx_serve_smoke.ok:
            console.print(
                f"[red]smoke:[/red] {mlx_serve_smoke.detail}\n"
                "  re-run with `--no-smoke` to skip the smoke test."
            )
            raise typer.Exit(code=1)

        manifest_path = finalize_mlx_serve_export(
            store=store,
            spec=spec,
            prepared=mlx_serve_result,
            smoke_output_first_line=None if mlx_serve_smoke is None else mlx_serve_smoke.detail,
            adapter_name=adapter,
            adapter_mix=mix_entries,
        )
        console.print(f"[green]exported:[/green] {mlx_serve_result.export_dir}")
        console.print("target:  mlx-serve")
        assert mlx_serve_result.launch_script_path is not None
        console.print(f"launch:  {mlx_serve_result.launch_script_path.name}")
        console.print(f"manifest: {manifest_path.name}")
        if mlx_serve_smoke is not None and mlx_serve_smoke.detail:
            console.print(f"smoke:   {mlx_serve_smoke.detail}")
        return

    try:
        result = run_export(
            store,
            spec,
            plan,
            target=resolved_target.name,
            cached_base_dir=cached.path,
            subprocess_runner=_verbose_runner if verbose else None,
            skip_ollama=skip_ollama or resolved_target.name != "ollama",
            skip_smoke=no_smoke,
            source_dlm_path=path.resolve(),
            training_sequence_len=parsed.frontmatter.training.sequence_len,
            override_temperature=parsed.frontmatter.export.default_temperature,
            override_top_p=parsed.frontmatter.export.default_top_p,
            draft_override=draft,
            draft_disabled=no_draft,
            adapter_name=adapter,
            adapter_path_override=adapter_path_override,
            adapter_mix=mix_entries,
        )
    except UnsafeMergeError as exc:
        console.print(f"[red]merge:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except VendoringError as exc:
        console.print(
            f"[red]vendor:[/red] {exc}\n"
            "  run `scripts/bump-llama-cpp.sh build` or "
            "`git submodule update --init --recursive`."
        )
        raise typer.Exit(code=1) from exc
    except PreflightError as exc:
        console.print(f"[red]preflight[{exc.probe}]:[/red] {exc.detail}")
        raise typer.Exit(code=1) from exc
    except SubprocessError as exc:
        console.print(f"[red]subprocess:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except OllamaBinaryNotFoundError as exc:
        console.print(
            f"[red]ollama:[/red] {exc}\n"
            "  install from https://ollama.com/download "
            "or re-run with `--skip-ollama`."
        )
        raise typer.Exit(code=1) from exc
    except OllamaVersionError as exc:
        console.print(f"[red]ollama:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except OllamaCreateError as exc:
        console.print(f"[red]ollama create:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except OllamaSmokeError as exc:
        console.print(
            f"[red]smoke:[/red] {exc}\n  re-run with `--no-smoke` to skip the smoke test."
        )
        raise typer.Exit(code=1) from exc
    except OllamaError as exc:
        console.print(f"[red]ollama:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ExportError as exc:
        console.print(f"[red]export:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if resolved_target.name == "llama-server":
        adapter_dir = adapter_path_override
        if adapter_dir is None:
            if adapter is None:
                adapter_dir = store.resolve_current_adapter()
            else:
                adapter_dir = store.resolve_current_adapter_for(adapter)
        assert adapter_dir is not None
        try:
            llama_server_result = prepare_llama_server_export(
                export_dir=result.export_dir,
                manifest_path=result.manifest_path,
                artifacts=result.artifacts,
                adapter_dir=adapter_dir,
                spec=spec,
                training_sequence_len=parsed.frontmatter.training.sequence_len,
            )
        except VendoringError as exc:
            console.print(
                f"[red]vendor:[/red] {exc}\n"
                "  run `scripts/bump-llama-cpp.sh build --with-server` or "
                "`git submodule update --init --recursive`."
            )
            raise typer.Exit(code=1) from exc
        except ExportError as exc:
            console.print(f"[red]export:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        llama_server_smoke = None if no_smoke else resolved_target.smoke_test(llama_server_result)
        if llama_server_smoke is not None and not llama_server_smoke.ok:
            console.print(
                f"[red]smoke:[/red] {llama_server_smoke.detail}\n"
                "  re-run with `--no-smoke` to skip the smoke test."
            )
            raise typer.Exit(code=1)

    cached_tag = " [dim](cached base)[/dim]" if result.cached else ""
    console.print(f"[green]exported:[/green] {result.export_dir}{cached_tag}")
    for artifact in result.artifacts:
        console.print(f"  {artifact.name}")

    # S26 X1 — also emit a sway.yaml next to the GGUF when the user
    # asks for it. Done AFTER the regular export so a sway-side
    # failure can never roll back a working GGUF deployment.
    if emit_sway_json:
        from dlm.export.sway_json import SwayJsonExportError, write_sway_json

        try:
            sway_yaml_path = write_sway_json(path, result.export_dir)
        except SwayJsonExportError as exc:
            console.print(f"[red]sway-json:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        console.print(f"[green]sway.yaml:[/green] {sway_yaml_path}")
        console.print("  next: sway run " + str(sway_yaml_path))
    if resolved_target.name == "llama-server":
        assert llama_server_result.launch_script_path is not None
        assert llama_server_result.config_path is not None
        console.print(f"target:  {result.target}")
        console.print(f"launch:  {llama_server_result.launch_script_path.name}")
        console.print(f"template: {llama_server_result.config_path.name}")
        if llama_server_smoke is not None and llama_server_smoke.detail:
            console.print(f"smoke:   {llama_server_smoke.detail}")
        return
    if result.ollama_name:
        console.print(f"ollama:  {result.ollama_name} (v{result.ollama_version})")
    if result.smoke_output_first_line:
        console.print(f"smoke:   {result.smoke_output_first_line}")


