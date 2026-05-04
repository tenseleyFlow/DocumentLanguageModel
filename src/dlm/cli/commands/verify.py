"""`dlm verify` — verify a .dlm.pack's provenance chain."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def verify_cmd(
    path: Annotated[Path, typer.Argument(help=".dlm.pack to verify.")],
    trust_on_first_use: Annotated[
        bool,
        typer.Option(
            "--trust-on-first-use",
            help=(
                "Record the signer's public key under ~/.dlm/trusted-keys/ "
                "on first verify. Without this flag an unknown signer is "
                "rejected with exit code 2."
            ),
        ),
    ] = False,
    trusted_keys_dir: Annotated[
        Path | None,
        typer.Option(
            "--trusted-keys-dir",
            help="Override ~/.dlm/trusted-keys/ (useful for scripted verify).",
            hidden=True,
        ),
    ] = None,
) -> None:
    """Verify a .dlm.pack's provenance chain.

    Exit codes: 0 verified, 1 broken chain (or missing provenance),
    2 untrusted signer, 3 signature rejected.
    """
    from rich.console import Console

    from dlm.pack.errors import PackLayoutError
    from dlm.pack.layout import PROVENANCE_FILENAME
    from dlm.pack.unpacker import read_pack_member_bytes
    from dlm.share.errors import ShareError
    from dlm.share.provenance import (
        ProvenanceChainBroken,
        ProvenanceSchemaError,
        UnknownSignerError,
        load_provenance_json,
        verify_provenance,
    )

    console = Console(stderr=True)
    keys_dir = trusted_keys_dir or (Path.home() / ".dlm" / "trusted-keys")

    try:
        payload = read_pack_member_bytes(path, PROVENANCE_FILENAME)
    except PackLayoutError as exc:
        console.print(f"[red]verify:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        console.print(f"[red]verify:[/red] cannot read {path}: {exc}")
        raise typer.Exit(code=1) from exc

    if payload is None:
        console.print(f"[red]verify:[/red] {path} is unsigned — no {PROVENANCE_FILENAME} inside.")
        raise typer.Exit(code=1)

    # Write the in-pack JSON to a temp file so `load_provenance_json`
    # can use its normal filesystem path. Keeps the parser single-
    # sourced and the error messages consistent with the filesystem
    # call-site.
    import tempfile

    with tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False) as fh:
        fh.write(payload)
        tmp_path = Path(fh.name)
    try:
        provenance = load_provenance_json(tmp_path)
    except ProvenanceSchemaError as exc:
        console.print(f"[red]verify:[/red] malformed provenance.json: {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    try:
        result = verify_provenance(
            provenance,
            trusted_keys_dir=keys_dir,
            tofu=trust_on_first_use,
        )
    except UnknownSignerError as exc:
        console.print(f"[red]verify:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    except ProvenanceChainBroken as exc:
        console.print(f"[red]verify:[/red] chain broken: {exc}")
        raise typer.Exit(code=1) from exc
    except ShareError as exc:
        console.print(f"[red]verify:[/red] signature rejected: {exc}")
        raise typer.Exit(code=3) from exc

    out = Console()
    out.print(f"[green]verified:[/green] {path.name}")
    out.print(f"  signer:          {result.signer_fingerprint}")
    out.print(f"  trusted-key:     {result.trusted_key_path}")
    out.print(f"  adapter_sha256:  {provenance.adapter_sha256[:12]}...")
    out.print(f"  base_revision:   {provenance.base_revision}")
    out.print(f"  corpus_root:     {provenance.corpus_root_sha256[:12]}...")
    out.print(f"  signed_at:       {provenance.signed_at}")
    if result.tofu_recorded:
        out.print(
            f"[yellow]note:[/yellow] recorded new trust entry "
            f"at {result.trusted_key_path}; subsequent verifies use strict mode."
        )
