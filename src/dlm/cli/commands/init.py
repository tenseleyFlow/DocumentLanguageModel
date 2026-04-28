"""`dlm init` — bootstrap a new .dlm file with sensible defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer


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
        # pointer.
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
