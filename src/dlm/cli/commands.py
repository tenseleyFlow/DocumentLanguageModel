"""Subcommand stubs for the v1.0 CLI surface.

Every stub raises `NotImplementedError` with the sprint number that will
implement it. This makes `dlm --help` self-documenting about project
progress. Arguments are accepted so `--help` renders the real eventual
surface; they're unused until each subcommand's owning sprint lands,
which is why `src/dlm/cli/commands.py` has a ruff per-file-ignore for
`ARG001` in `pyproject.toml`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer

if TYPE_CHECKING:
    from datetime import timedelta


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
            help="(reserved) Pick a starter template once the template library ships.",
            hidden=True,
        ),
    ] = None,
    i_accept_license: Annotated[
        bool,
        typer.Option("--i-accept-license", help="Accept gated base-model license (Sprint 12b)."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing .dlm at path."),
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

    if template is not None:
        console.print(
            "[yellow]note:[/yellow] --template is reserved; no starter templates "
            "ship yet — the scaffold below is the default."
        )

    if path.exists() and not force:
        console.print(
            f"[red]init:[/red] {path} already exists. "
            "Re-run with [bold]--force[/bold] to overwrite."
        )
        raise typer.Exit(code=1)

    try:
        spec = resolve_base_model(base, accept_license=i_accept_license)
    except UnknownBaseModelError as exc:
        console.print(f"[red]init:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except GatedModelError as exc:
        # Gated + user didn't pass --i-accept-license up-front. Prompt
        # interactively if we have a TTY; otherwise refuse with the flag
        # pointer (audit F22 non-interactive path).
        if not _prompt_accept_license(console, base, exc.license_url):
            console.print(
                "[red]license:[/red] refused. Re-run with "
                "[bold]--i-accept-license[/bold] to accept non-interactively."
            )
            raise typer.Exit(code=1) from exc
        spec = resolve_base_model(base, accept_license=True)

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

    dlm_id = mint_ulid()
    _write_init_scaffold(path, spec.key, dlm_id)

    # Create the store + write the initial manifest so `dlm show` sees
    # the license record and `dlm train` has a prior manifest to diff
    # against (audit-05 B2).
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
    console.print(f"[green]init:[/green] wrote {path}")


def _previously_accepted(store_manifest_path: Path) -> bool:
    """Return True iff the store manifest already holds a LicenseAcceptance.

    `dlm prompt` and `dlm export` operate on an already-trained adapter;
    the gated-base license was accepted at `dlm train --i-accept-license`
    time and persisted into `manifest.license_acceptance` (Sprint 12b).
    Replaying that acceptance here is correct; silently hardcoding
    `accept_license=True` is not — it would let a never-accepted
    gated base slip through.
    """
    if not store_manifest_path.exists():
        return False
    from dlm.store.errors import ManifestCorruptError
    from dlm.store.manifest import load_manifest

    try:
        manifest = load_manifest(store_manifest_path)
    except (ManifestCorruptError, OSError):
        # Audit-05 N2: narrow from bare `Exception` so programmer bugs
        # propagate instead of being silently treated as "no acceptance."
        return False
    return manifest.license_acceptance is not None


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


def train_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to train.")],
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
    gpus: Annotated[
        str | None,
        typer.Option(
            "--gpus",
            help=(
                "Multi-GPU training (Sprint 23). `all` uses every visible "
                "CUDA device; `N` uses the first N; `0,1` selects exact "
                "device ids. Dispatches to `accelerate launch` when >1 "
                "device is selected. Omit for single-process training."
            ),
        ),
    ] = None,
    watch: Annotated[
        bool,
        typer.Option(
            "--watch",
            help=(
                "Save-to-train mode (Sprint 25). After an initial train, "
                "block on filesystem events and run incremental retrains "
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
) -> None:
    """Train / retrain a .dlm against its base model."""
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

    # Sprint 23: --gpus dispatches to accelerate launch when >1 device
    # is selected. The single-GPU path falls through to the existing
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

    try:
        parsed = parse_file(path)
    except (DlmParseError, OSError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    try:
        spec = resolve_base_model(
            parsed.frontmatter.base_model,
            accept_license=i_accept_license,
        )
    except GatedModelError as exc:
        console.print(f"[red]license:[/red] base model {parsed.frontmatter.base_model!r} is gated.")
        if exc.license_url:
            console.print(f"  review the license at: {exc.license_url}")
        console.print(
            "  re-run with [bold]--i-accept-license[/bold] once you have accepted. "
            "(Sprint 12b will persist acceptance in the frontmatter.)"
        )
        raise typer.Exit(code=1) from exc
    # Audit-08 M1: detect the DDP world_size set by `accelerate launch`
    # (WORLD_SIZE env var) and thread it into the doctor so the plan's
    # effective_batch_size reflects the rank count. Single-process
    # runs read 1 and the plan math is unchanged.
    from dlm.train.distributed import detect_world_size

    ws = detect_world_size()
    plan = doctor(training_config=parsed.frontmatter.training, world_size=ws).plan
    if plan is None:
        console.print(
            "[red]doctor:[/red] no viable training plan for this host. "
            "Run `dlm doctor` for details."
        )
        raise typer.Exit(code=1)

    store = for_dlm(parsed.frontmatter.dlm_id)
    store.ensure_layout()

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
            capabilities=doctor().capabilities,
            world_size=ws,
        )
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

    # Sprint 25: --watch keeps the training context alive and re-runs
    # incremental cycles on file change. Entered AFTER the initial
    # train so the loop resumes from a real committed adapter.
    if watch:
        if watch_repl:
            console.print(
                "[red]train:[/red] --watch --repl is scaffolded but not yet "
                "implemented (Sprint 25 [~] per DoD). The threaded REPL "
                "bridge needs a test harness we don't have in CI today."
            )
            raise typer.Exit(code=2)

        from dlm.watch.loop import run_watch
        from dlm.watch.status import WatchStatus, render_status

        status = WatchStatus(doc_path=str(path), sections=len(parsed.sections))
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
            )
        except KeyboardInterrupt:
            console.print("[dim]watch:[/dim] Ctrl-C received, exiting")
            raise typer.Exit(code=0)  # noqa: B904
        raise typer.Exit(code=exit_code)


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
    `strip_gpus_flag` helper (audit-08 N1).
    """
    from dlm.train.distributed.gpus import strip_gpus_flag

    return strip_gpus_flag(argv, skip_argv0=True)


def prompt_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to query.")],
    query: Annotated[str | None, typer.Argument(help="One-shot prompt (omit for stdin).")] = None,
    max_tokens: Annotated[int, typer.Option("--max-tokens")] = 256,
    temp: Annotated[float, typer.Option("--temp")] = 0.7,
    top_p: Annotated[float | None, typer.Option("--top-p")] = None,
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


def export_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to export.")],
    quant: Annotated[
        str | None,
        typer.Option("--quant", help="GGUF quant level (defaults to frontmatter)."),
    ] = None,
    merged: Annotated[bool, typer.Option("--merged")] = False,
    dequantize: Annotated[bool, typer.Option("--dequantize")] = False,
    name: Annotated[str | None, typer.Option("--name", help="Ollama model name.")] = None,
    no_template: Annotated[
        bool,
        typer.Option("--no-template", help="Skip writing TEMPLATE into the Modelfile."),
    ] = False,
    no_smoke: Annotated[bool, typer.Option("--no-smoke")] = False,
    no_imatrix: Annotated[
        bool,
        typer.Option(
            "--no-imatrix",
            help=(
                "Skip importance-matrix calibration. Default uses the "
                "replay corpus to calibrate k-quant quantization (Sprint 11.6)."
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
) -> None:
    """Export the adapter to an Ollama-registered model."""
    from collections.abc import Sequence

    from rich.console import Console

    from dlm.base_models import GatedModelError, download_spec
    from dlm.base_models import resolve as resolve_base_model
    from dlm.doc.parser import parse_file
    from dlm.export import (
        ExportError,
        PreflightError,
        SubprocessError,
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
    already_accepted = _previously_accepted(store.manifest)
    try:
        spec = resolve_base_model(parsed.frontmatter.base_model, accept_license=already_accepted)
    except GatedModelError as exc:
        console.print(f"[red]license:[/red] base model {parsed.frontmatter.base_model!r} is gated.")
        if exc.license_url:
            console.print(f"  review the license at: {exc.license_url}")
        console.print("  accept via `dlm train --i-accept-license` before exporting.")
        raise typer.Exit(code=1) from exc

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
        from transformers import AutoModelForCausalLM

        from dlm.export.weighted_merge import (
            MixEntry,
            build_weighted_merged,
            resolve_first_source_path,
            save_merged_to_tmp,
        )

        store.ensure_layout()
        entries_typed = [MixEntry(name=n, weight=w) for (n, w) in mix_entries]
        base_model = AutoModelForCausalLM.from_pretrained(str(cached.path), revision=spec.revision)
        merged = build_weighted_merged(
            base_model,
            store,
            spec,
            entries_typed,
            combination_type=adapter_mix_method,  # type: ignore[arg-type]
        )
        merge_dir = store.cache_dir_for("_export_merged_" + "_".join(n for n, _ in mix_entries))
        # Copy tokenizer + training_run.json from a source adapter so
        # the downstream preflight (tokenizer_vocab) + merge-safety
        # (was_qlora) gates both work on the composite (audit-07 B2).
        first_source = resolve_first_source_path(store, entries_typed)
        adapter_path_override = save_merged_to_tmp(
            merged,
            merge_dir,
            tokenizer_source=first_source,
            training_run_source=first_source,
        )

    try:
        result = run_export(
            store,
            spec,
            plan,
            cached_base_dir=cached.path,
            subprocess_runner=_verbose_runner if verbose else None,
            skip_ollama=skip_ollama,
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

    cached_tag = " [dim](cached base)[/dim]" if result.cached else ""
    console.print(f"[green]exported:[/green] {result.export_dir}{cached_tag}")
    for artifact in result.artifacts:
        console.print(f"  {artifact.name}")
    if result.ollama_name:
        console.print(f"ollama:  {result.ollama_name} (v{result.ollama_version})")
    if result.smoke_output_first_line:
        console.print(f"smoke:   {result.smoke_output_first_line}")


def pack_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to pack.")],
    out: Annotated[Path | None, typer.Option("--out")] = None,
    include_exports: Annotated[bool, typer.Option("--include-exports")] = False,
    include_base: Annotated[bool, typer.Option("--include-base")] = False,
    include_logs: Annotated[bool, typer.Option("--include-logs")] = False,
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
    from dlm.pack.errors import BaseLicenseRefusedError
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
    except DlmParseError as exc:
        console.print(f"[red]parse:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    size_mb = result.bytes_written / (1024 * 1024)
    console.print(
        f"[green]packed:[/green] {result.path} "
        f"({size_mb:.2f} MB, content_type={result.content_type})"
    )


def unpack_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm.pack to install.")],
    force: Annotated[bool, typer.Option("--force")] = False,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out", help="Directory to place the restored .dlm (default: alongside the pack)."
        ),
    ] = None,
) -> None:
    """Install a .dlm.pack into the local store."""
    from rich.console import Console

    from dlm.pack.errors import (
        PackFormatVersionError,
        PackIntegrityError,
        PackLayoutError,
    )
    from dlm.pack.unpacker import unpack

    console = Console(stderr=True)

    try:
        result = unpack(path, force=force, out_dir=out)
    except PackFormatVersionError as exc:
        console.print(f"[red]unpack:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except PackIntegrityError as exc:
        console.print(f"[red]unpack:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except PackLayoutError as exc:
        console.print(f"[red]unpack:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]unpacked:[/green] {result.dlm_path}")
    console.print(f"  store:  {result.store_path}")
    console.print(f"  dlm_id: {result.dlm_id}")
    if result.applied_migrations:
        steps = " → ".join(
            f"v{v}" for v in (*result.applied_migrations, result.header.pack_format_version + 1)
        )
        console.print(f"  migrated: {steps}")


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
    """Interactive REPL against the trained adapter (Sprint 24)."""
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


def metrics_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file whose store we query.")],
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
    csv_out: Annotated[bool, typer.Option("--csv", help="Emit CSV.")] = False,
    run_id: Annotated[
        int | None,
        typer.Option("--run-id", help="Only show this run (drill-down)."),
    ] = None,
    phase: Annotated[
        str | None,
        typer.Option("--phase", help="Filter by phase: sft|dpo|orpo|cpt."),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option(
            "--since",
            help="Time window (e.g. `24h`, `7d`, `30m`). Filters `started_at`.",
        ),
    ] = None,
    limit: Annotated[int, typer.Option("--limit")] = 20,
) -> None:
    """Query the per-store metrics database (Sprint 26)."""
    import csv
    import json
    import sys

    from rich.console import Console

    from dlm.doc.parser import parse_file
    from dlm.metrics.queries import (
        evals_for_run,
        evals_to_dict,
        recent_runs,
        runs_to_dict,
        steps_for_run,
        steps_to_dict,
    )
    from dlm.store.paths import for_dlm

    console = Console(stderr=True)

    if json_out and csv_out:
        console.print("[red]metrics:[/red] --json and --csv are mutually exclusive")
        raise typer.Exit(code=2)

    since_delta = _parse_since_arg(since, console) if since else None

    parsed = parse_file(path)
    store = for_dlm(parsed.frontmatter.dlm_id)

    runs = recent_runs(
        store.root, limit=limit, phase=phase, since=since_delta, run_id=run_id
    )

    if run_id is not None:
        # Drill-down: show this run's steps + evals.
        if not runs:
            console.print(f"[red]metrics:[/red] no run with run_id={run_id}")
            raise typer.Exit(code=1)
        run = runs[0]
        steps = steps_for_run(store.root, run_id)
        evals = evals_for_run(store.root, run_id)

        if json_out:
            payload = {
                "run": runs_to_dict([run])[0],
                "steps": steps_to_dict(steps),
                "evals": evals_to_dict(evals),
            }
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
            return
        if csv_out:
            writer = csv.writer(sys.stdout)
            writer.writerow(["step", "loss", "lr", "grad_norm", "val_loss"])
            eval_by_step = {e.step: e.val_loss for e in evals}
            for s in steps:
                writer.writerow(
                    [s.step, s.loss, s.lr, s.grad_norm, eval_by_step.get(s.step)]
                )
            return
        console.print(
            f"[green]run_id={run.run_id}[/green]  phase={run.phase}  "
            f"seed={run.seed}  status={run.status}  steps={len(steps)}  "
            f"evals={len(evals)}"
        )
        if evals:
            last = evals[-1]
            console.print(
                f"  last eval: step={last.step}  val_loss={last.val_loss}  "
                f"perplexity={last.perplexity}"
            )
        return

    # Top-level: list runs.
    if json_out:
        sys.stdout.write(json.dumps({"runs": runs_to_dict(runs)}, indent=2) + "\n")
        return
    if csv_out:
        writer = csv.writer(sys.stdout)
        writer.writerow(["run_id", "phase", "seed", "status", "started_at", "ended_at"])
        for r in runs:
            writer.writerow(
                [r.run_id, r.phase, r.seed, r.status, r.started_at, r.ended_at]
            )
        return

    if not runs:
        console.print("[dim]metrics:[/dim] no runs found (hint: train first, or adjust filters)")
        return
    console.print(f"[bold]Runs: {len(runs)}[/bold]")
    for r in runs:
        console.print(
            f"  run_id={r.run_id}  phase={r.phase}  seed={r.seed}  "
            f"status={r.status}  started={r.started_at}"
        )


def metrics_watch_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file whose store we tail.")],
    poll_seconds: Annotated[
        float,
        typer.Option("--poll-seconds", help="How often to re-read the metrics DB."),
    ] = 1.0,
) -> None:
    """Tail the metrics DB: print new steps/evals as they land (Sprint 26)."""
    import time

    from rich.console import Console

    from dlm.doc.parser import parse_file
    from dlm.metrics.queries import evals_for_run, latest_run_id, steps_for_run
    from dlm.store.paths import for_dlm

    console = Console()

    parsed = parse_file(path)
    store = for_dlm(parsed.frontmatter.dlm_id)

    console.print(
        f"[dim]metrics watch:[/dim] polling {store.root} every {poll_seconds}s (Ctrl-C to exit)"
    )

    current_run: int | None = None
    last_step_seen = 0
    last_eval_step_seen = 0
    try:
        while True:
            run_id = latest_run_id(store.root)
            if run_id is None:
                time.sleep(poll_seconds)
                continue
            if run_id != current_run:
                current_run = run_id
                last_step_seen = 0
                last_eval_step_seen = 0
                console.print(f"[green]→ following run_id={run_id}[/green]")

            new_steps = steps_for_run(store.root, run_id, since_step=last_step_seen)
            for s in new_steps:
                console.print(
                    f"  step {s.step:>5}  loss={s.loss}  lr={s.lr}  grad_norm={s.grad_norm}"
                )
                last_step_seen = s.step

            new_evals = evals_for_run(store.root, run_id, since_step=last_eval_step_seen)
            for e in new_evals:
                console.print(
                    f"  [yellow]eval @ step {e.step}[/yellow]  "
                    f"val_loss={e.val_loss}  perplexity={e.perplexity}"
                )
                last_eval_step_seen = e.step

            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        console.print("[dim]metrics watch:[/dim] bye")


def _parse_since_arg(since: str, console: object) -> timedelta:
    """Parse `24h` / `7d` / `30m` / `10s` into a timedelta."""
    from datetime import timedelta

    from rich.console import Console

    assert isinstance(console, Console)

    if not since:
        raise typer.Exit(code=2)
    unit = since[-1].lower()
    try:
        value = int(since[:-1])
    except ValueError:
        console.print(f"[red]metrics:[/red] --since {since!r} not an integer+unit")
        raise typer.Exit(code=2) from None
    if unit == "s":
        return timedelta(seconds=value)
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    console.print(f"[red]metrics:[/red] --since {since!r} unit must be s/m/h/d")
    raise typer.Exit(code=2)


def doctor_cmd(
    json_out: Annotated[bool, typer.Option("--json", help="Emit machine-readable output.")] = False,
) -> None:
    """Inspect hardware and print the resolved training plan."""
    import json

    from dlm.hardware import doctor, render_text

    result = doctor()
    if json_out:
        typer.echo(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        typer.echo(render_text(result))


def show_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to inspect.")],
    json_out: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Show training history, exports, and adapter state."""
    import json as _json
    import sys

    from rich.console import Console

    from dlm.doc.errors import DlmParseError
    from dlm.doc.parser import parse_file
    from dlm.store.errors import ManifestCorruptError
    from dlm.store.inspect import inspect_store
    from dlm.store.paths import for_dlm

    console = Console(stderr=True)
    out_console = Console()

    try:
        parsed = parse_file(path)
    except (DlmParseError, OSError) as exc:
        console.print(f"[red]show:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    store = for_dlm(parsed.frontmatter.dlm_id)
    # Store may not exist yet (no `dlm train` run). Treat that as an
    # informational state rather than an error — useful after `dlm init`.
    if not store.manifest.exists():
        if json_out:
            sys.stdout.write(
                _json.dumps(
                    {
                        "dlm_id": parsed.frontmatter.dlm_id,
                        "base_model": parsed.frontmatter.base_model,
                        "store_initialized": False,
                        "source_path": str(path.resolve()),
                    },
                    indent=2,
                )
                + "\n"
            )
        else:
            out_console.print(f"[bold]{path}[/bold]")
            out_console.print(f"  dlm_id:       {parsed.frontmatter.dlm_id}")
            out_console.print(f"  base_model:   {parsed.frontmatter.base_model}")
            out_console.print("  store:        [dim]not yet initialized (run `dlm train`)[/dim]")
        return

    try:
        inspection = inspect_store(store, source_path=path.resolve())
    except ManifestCorruptError as exc:
        console.print(f"[red]show:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_out:
        # Write JSON to raw stdout — Rich's Console wraps lines at the
        # terminal width and would corrupt the JSON.
        sys.stdout.write(_json.dumps(_inspection_to_dict(inspection), indent=2, default=str) + "\n")
        return

    _render_inspection_text(out_console, path, inspection)


def _inspection_to_dict(inspection: object) -> dict[str, object]:
    """Flatten a StoreInspection into a JSON-safe dict.

    Schema is the v1 contract for `dlm show --json`; any reshape is a
    version bump (recorded in tests/golden/cli-json/).
    """
    from dlm.store.inspect import StoreInspection

    assert isinstance(inspection, StoreInspection)
    return {
        "dlm_id": inspection.dlm_id,
        "path": str(inspection.path),
        "base_model": inspection.base_model,
        "base_model_revision": inspection.base_model_revision,
        "adapter_version": inspection.adapter_version,
        "training_runs": inspection.training_runs,
        "last_trained_at": inspection.last_trained_at,
        "has_adapter_current": inspection.has_adapter_current,
        "replay_size_bytes": inspection.replay_size_bytes,
        "total_size_bytes": inspection.total_size_bytes,
        "source_path": str(inspection.source_path) if inspection.source_path else None,
        "orphaned": inspection.orphaned,
        "exports": [e.model_dump(mode="json") for e in inspection.exports],
        "content_hashes": dict(inspection.content_hashes),
        "pinned_versions": dict(inspection.pinned_versions),
        "named_adapters": [
            {
                "name": a.name,
                "has_current": a.has_current,
                "latest_version": a.latest_version,
            }
            for a in inspection.named_adapters
        ],
    }


def _render_inspection_text(console: object, path: Path, inspection: object) -> None:
    """Human-readable `dlm show` output."""
    from rich.console import Console

    from dlm.store.inspect import StoreInspection

    assert isinstance(console, Console)
    assert isinstance(inspection, StoreInspection)

    console.print(f"[bold]{path}[/bold]")
    console.print(f"  dlm_id:         {inspection.dlm_id}")
    rev = inspection.base_model_revision
    rev_str = f" (revision {rev[:7]})" if rev else ""
    console.print(f"  base_model:     {inspection.base_model}{rev_str}")
    console.print(
        f"  store:          {inspection.path}  ({_human_size(inspection.total_size_bytes)})"
    )
    if inspection.named_adapters:
        # Multi-adapter store: render the per-adapter pointers rather
        # than the flat field (which stays 0 on multi-adapter docs).
        console.print("  adapters:")
        for adapter in inspection.named_adapters:
            if adapter.has_current:
                console.print(f"    {adapter.name:16}v{adapter.latest_version:04d}")
            else:
                console.print(f"    {adapter.name:16}[dim]no current pointer[/dim]")
    elif inspection.has_adapter_current:
        console.print(f"  adapter:        v{inspection.adapter_version:04d}")
    else:
        console.print("  adapter:        [dim]none (no `dlm train` yet)[/dim]")
    last = inspection.last_trained_at
    last_str = f" — last {last.isoformat(timespec='seconds')}" if last else ""
    console.print(f"  training runs:  {inspection.training_runs}{last_str}")
    console.print(f"  exports:        {len(inspection.exports)}")
    for exp in inspection.exports:
        tag = f" — {exp.ollama_name}" if exp.ollama_name else ""
        console.print(f"                  {exp.quant}{tag}")
    if inspection.orphaned:
        console.print("  [yellow]orphaned:[/yellow]     source .dlm is missing or mismatched")


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n //= 1024
    return f"{n} PB"


def migrate_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm file to migrate.")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    no_backup: Annotated[bool, typer.Option("--no-backup")] = False,
) -> None:
    """Migrate a .dlm frontmatter to the current schema version."""
    from rich.console import Console

    from dlm.doc.errors import DlmParseError
    from dlm.doc.migrate import migrate_file

    console = Console(stderr=True)

    try:
        result = migrate_file(path, dry_run=dry_run, no_backup=no_backup)
    except DlmParseError as exc:
        console.print(f"[red]migrate:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not result.applied:
        console.print(
            f"[green]migrate:[/green] {path} already at v{result.target_version} "
            "(no migrations needed)."
        )
        return

    applied_str = " → ".join(f"v{v}" for v in (*result.applied, result.target_version))
    if dry_run:
        console.print(
            f"[yellow]dry-run:[/yellow] {path} would migrate {applied_str} "
            "(re-run without --dry-run to apply)."
        )
        return

    if result.backup_path is not None:
        console.print(f"[dim]backup:[/dim]  {result.backup_path}")
    console.print(f"[green]migrated:[/green] {path} {applied_str}")
