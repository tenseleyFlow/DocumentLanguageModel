"""`dlm doctor` — inspect hardware and print the resolved training plan."""

from __future__ import annotations

from typing import Annotated

import typer


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
