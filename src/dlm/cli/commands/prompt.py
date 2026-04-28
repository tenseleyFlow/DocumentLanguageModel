"""`dlm prompt` — run inference against the trained adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from dlm.cli.commands._shared import _previously_accepted


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
    Covered by the slow-marked vision-language integration test.
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
    Covered by the slow-marked audio integration test.
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
