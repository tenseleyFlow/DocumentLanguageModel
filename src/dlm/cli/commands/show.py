"""`dlm show` — show training history, exports, and adapter state."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from dlm.cli.commands._shared import _human_size


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
    from dlm.store.paths import for_dlm
    from dlm.store.show import StoreViewRequest, gather_store_view

    console = Console(stderr=True)
    out_console = Console()

    try:
        parsed = parse_file(path)
    except (DlmParseError, OSError) as exc:
        console.print(f"[red]show:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    store = for_dlm(parsed.frontmatter.dlm_id)

    try:
        view = gather_store_view(StoreViewRequest(parsed=parsed, target_path=path, store=store))
    except ManifestCorruptError as exc:
        console.print(f"[red]show:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if view.inspection is None:
        if json_out:
            payload: dict[str, object] = {
                "dlm_id": view.parsed_dlm_id,
                "base_model": view.parsed_base_model,
                "store_initialized": False,
                "source_path": str(view.source_path),
                "training_cache_config": view.training_cache_config,
            }
            if view.training_sources is not None:
                payload["training_sources"] = view.training_sources
            if view.discovered_configs:
                payload["discovered_training_configs"] = view.discovered_configs
            sys.stdout.write(_json.dumps(payload, indent=2) + "\n")
        else:
            out_console.print(f"[bold]{path}[/bold]")
            out_console.print(f"  dlm_id:       {view.parsed_dlm_id}")
            out_console.print(f"  base_model:   {view.parsed_base_model}")
            out_console.print("  store:        [dim]not yet initialized (run `dlm train`)[/dim]")
            if view.training_sources:
                _render_training_sources_text(out_console, view.training_sources)
        return

    if json_out:
        payload_full = _inspection_to_dict(view.inspection)
        if view.training_sources is not None:
            payload_full["training_sources"] = view.training_sources
        if view.discovered_configs:
            payload_full["discovered_training_configs"] = view.discovered_configs
        if view.training_cache is not None:
            payload_full["training_cache"] = view.training_cache
        payload_full["training_cache_config"] = view.training_cache_config
        if view.gate is not None:
            payload_full["gate"] = view.gate
        if view.preference_mining is not None:
            payload_full["preference_mining"] = view.preference_mining
            payload_full["preference_mining_runs"] = view.preference_mining["run_count"]
            payload_full["total_auto_mined_pairs"] = view.preference_mining["total_mined_pairs"]
        if view.base_security is not None:
            payload_full["base_security"] = view.base_security
        # Write JSON to raw stdout — Rich's Console wraps lines at the
        # terminal width and would corrupt the JSON.
        sys.stdout.write(_json.dumps(payload_full, indent=2, default=str) + "\n")
        return

    _render_inspection_text(out_console, path, view.inspection)
    if view.training_sources:
        _render_training_sources_text(out_console, view.training_sources)
    if view.training_cache is not None and view.training_cache.get("entry_count", 0):
        _render_training_cache_text(out_console, view.training_cache)
    if view.gate is not None:
        _render_gate_text(out_console, view.gate)
    if view.base_security is not None and view.base_security.get("trust_remote_code"):
        _render_base_security_text(out_console, view.base_security)


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


def _render_base_security_text(console: object, snap: dict[str, object]) -> None:
    from rich.console import Console

    assert isinstance(console, Console)
    arch = snap.get("architecture", "?")
    console.print(
        f"  [yellow]security:[/yellow] base uses [red]trust_remote_code=True[/red] "
        f"(arch={arch}) — HF loader will execute Python from the model repo"
    )


def _render_gate_text(console: object, snap: dict[str, object]) -> None:
    from rich.console import Console

    assert isinstance(console, Console)
    mode = snap.get("mode", "?")
    if mode == "diverged":
        console.print("  adapter gate ([red]diverged[/red]):")
        console.print(
            "    [yellow]gate training produced a non-finite loss; "
            "store fell back to gate-less routing[/yellow]"
        )
    else:
        console.print(f"  adapter gate ({mode}):")
    per_adapter = snap.get("per_adapter", [])
    if isinstance(per_adapter, list):
        for entry in per_adapter:
            if not isinstance(entry, dict):
                continue
            name = entry.get("adapter_name", "?")
            weight = entry.get("mean_weight")
            count = entry.get("sample_count")
            if weight is None:
                console.print(f"    {name}  [dim](no recorded events)[/dim]")
            else:
                w = float(weight) if isinstance(weight, (int, float)) else 0.0
                c = count if isinstance(count, int) else 0
                console.print(f"    {name:<16}  weight={w:.3f}  samples={c}")


def _render_training_cache_text(console: object, snap: dict[str, object]) -> None:
    from rich.console import Console

    assert isinstance(console, Console)
    ec_raw = snap.get("entry_count", 0)
    by_raw = snap.get("bytes", 0)
    entry_count = ec_raw if isinstance(ec_raw, int) else 0
    byte_count = by_raw if isinstance(by_raw, int) else 0
    console.print("  tokenized cache:")
    console.print(f"    entries:        {entry_count}")
    console.print(f"    size:           {_human_size(byte_count)}")
    rate = snap.get("last_run_hit_rate")
    if isinstance(rate, (int, float)):
        console.print(f"    last hit rate:  {float(rate):.1%}")


def _render_training_sources_text(console: object, records: list[dict[str, object]]) -> None:
    from rich.console import Console

    assert isinstance(console, Console)
    console.print("  training sources:")
    for rec in records:
        path = rec["path"]
        fc = rec.get("file_count")
        tb = rec.get("total_bytes")
        if fc is None:
            console.print(f"    {path}  [dim](not expanded)[/dim]")
        else:
            size = int(tb) if isinstance(tb, int) else 0
            console.print(f"    {path}  {fc} file(s), {_human_size(size)}")
