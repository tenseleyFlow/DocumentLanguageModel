"""`dlm train` — train / retrain a .dlm against its base model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

import typer


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
