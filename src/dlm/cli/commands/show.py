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
    training_sources, discovered_configs = _summarize_training_sources_and_discovered(
        parsed, path.resolve().parent
    )
    # The per-document cache config comes from frontmatter, not on-disk
    # state — report it on both the pre-train and initialized-store paths
    # so authors can sanity-check the knobs before `dlm train` runs.
    cache_cfg = parsed.frontmatter.training.cache
    training_cache_config: dict[str, object] = {
        "enabled": cache_cfg.enabled,
        "max_bytes": cache_cfg.max_bytes,
        "prune_older_than_days": cache_cfg.prune_older_than_days,
    }

    # Store may not exist yet (no `dlm train` run). Treat that as an
    # informational state rather than an error — useful after `dlm init`.
    if not store.manifest.exists():
        if json_out:
            payload: dict[str, object] = {
                "dlm_id": parsed.frontmatter.dlm_id,
                "base_model": parsed.frontmatter.base_model,
                "store_initialized": False,
                "source_path": str(path.resolve()),
                "training_cache_config": training_cache_config,
            }
            if training_sources is not None:
                payload["training_sources"] = training_sources
            if discovered_configs:
                payload["discovered_training_configs"] = discovered_configs
            sys.stdout.write(_json.dumps(payload, indent=2) + "\n")
        else:
            out_console.print(f"[bold]{path}[/bold]")
            out_console.print(f"  dlm_id:       {parsed.frontmatter.dlm_id}")
            out_console.print(f"  base_model:   {parsed.frontmatter.base_model}")
            out_console.print("  store:        [dim]not yet initialized (run `dlm train`)[/dim]")
            if training_sources:
                _render_training_sources_text(out_console, training_sources)
        return

    try:
        inspection = inspect_store(store, source_path=path.resolve())
    except ManifestCorruptError as exc:
        console.print(f"[red]show:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    training_cache = _summarize_training_cache(store.tokenized_cache_dir, store.root)
    gate = _summarize_gate(store)
    preference_mining = _summarize_preference_mining(store.root)
    base_security = _summarize_base_security(parsed.frontmatter.base_model)

    if json_out:
        payload_full = _inspection_to_dict(inspection)
        if training_sources is not None:
            payload_full["training_sources"] = training_sources
        if discovered_configs:
            payload_full["discovered_training_configs"] = discovered_configs
        if training_cache is not None:
            payload_full["training_cache"] = training_cache
        payload_full["training_cache_config"] = training_cache_config
        if gate is not None:
            payload_full["gate"] = gate
        if preference_mining is not None:
            payload_full["preference_mining"] = preference_mining
            payload_full["preference_mining_runs"] = preference_mining["run_count"]
            payload_full["total_auto_mined_pairs"] = preference_mining["total_mined_pairs"]
        if base_security is not None:
            payload_full["base_security"] = base_security
        # Write JSON to raw stdout — Rich's Console wraps lines at the
        # terminal width and would corrupt the JSON.
        sys.stdout.write(_json.dumps(payload_full, indent=2, default=str) + "\n")
        return

    _render_inspection_text(out_console, path, inspection)
    if training_sources:
        _render_training_sources_text(out_console, training_sources)
    if training_cache is not None and training_cache.get("entry_count", 0):
        _render_training_cache_text(out_console, training_cache)
    if gate is not None:
        _render_gate_text(out_console, gate)
    if base_security is not None and base_security.get("trust_remote_code"):
        _render_base_security_text(out_console, base_security)


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


def _summarize_training_sources(parsed: object, base_path: Path) -> list[dict[str, object]] | None:
    """Best-effort resolution of `training.sources` for `dlm show`.

    Returns None when the frontmatter declares no directives; returns
    a list of per-source dicts otherwise. Failures to expand (missing
    paths, policy escapes) fall back to declared-only records so the
    show output stays useful for debugging a misconfigured directive.
    """
    records, _ = _summarize_training_sources_and_discovered(parsed, base_path)
    return records


def _summarize_training_sources_and_discovered(
    parsed: object, base_path: Path
) -> tuple[list[dict[str, object]] | None, list[dict[str, object]]]:
    """Like `_summarize_training_sources` but also returns the per-anchor
    `.dlm/training.yaml` + `.dlm/ignore` discovery records.

    Returns `(training_sources, discovered_configs)`. `discovered_configs`
    is always a list (empty when nothing was found or the expansion
    failed); `training_sources` matches the single-value helper's
    contract.
    """
    from dlm.directives import DirectiveError, expand_sources
    from dlm.doc.parser import ParsedDlm

    assert isinstance(parsed, ParsedDlm)
    directives = parsed.frontmatter.training.sources
    if not directives:
        return None, []

    declared: list[dict[str, object]] = [
        {
            "path": d.path,
            "include": list(d.include),
            "exclude": list(d.exclude),
            "max_files": d.max_files,
            "max_bytes_per_file": d.max_bytes_per_file,
        }
        for d in directives
    ]

    try:
        result = expand_sources(parsed, base_path=base_path)
    except (DirectiveError, OSError):
        return declared, []

    records: list[dict[str, object]] = []
    for decl, prov in zip(declared, result.provenance, strict=False):
        records.append(
            {
                **decl,
                "file_count": prov.file_count,
                "total_bytes": prov.total_bytes,
                "skipped_binary": prov.skipped_binary,
                "skipped_encoding": prov.skipped_encoding,
                "skipped_over_size": prov.skipped_over_size,
            }
        )
    # If the expander returned fewer entries than declared (shouldn't
    # happen on success but defensive), pad with declared-only.
    if len(records) < len(declared):
        records.extend(declared[len(records) :])

    discovered_records: list[dict[str, object]] = []
    for dc in result.discovered:
        discovered_records.append(
            {
                "anchor": str(dc.anchor),
                "has_training_yaml": dc.config is not None,
                "has_ignore": bool(dc.ignore_rules),
                "include": list(dc.config.include) if dc.config else [],
                "exclude": list(dc.config.exclude) if dc.config else [],
                "exclude_defaults": (dc.config.exclude_defaults if dc.config else True),
                "metadata": dict(dc.config.metadata) if dc.config else {},
                "ignore_rules": len(dc.ignore_rules),
            }
        )
    return records, discovered_records


def _summarize_training_cache(cache_dir: Path, store_root: Path) -> dict[str, object] | None:
    """Return a JSON-friendly snapshot of the tokenized-section cache.

    None when the cache dir doesn't exist (store never trained with
    the cache, or pre-Sprint-31 layout). Cheap — reads the manifest
    only, not the entry files.
    """
    if not cache_dir.is_dir():
        return None
    from dlm.directives.cache import TokenizedCache
    from dlm.metrics import queries as _queries

    cache = TokenizedCache.open(cache_dir)
    last = _queries.latest_tokenization(store_root)
    return {
        "path": str(cache_dir),
        "entry_count": cache.entry_count,
        "bytes": cache.total_bytes,
        "last_run_hit_rate": last.hit_rate if last else None,
        "last_run_id": last.run_id if last else None,
    }


def _summarize_gate(store: object) -> dict[str, object] | None:
    """Return a JSON-friendly snapshot of the learned adapter gate.

    None when the store has no gate config (pre-Sprint-34 runs, or
    `training.gate.enabled` was false). Reads two sources: the
    on-disk `gate_config.json` for mode + adapter order, and the
    metrics `gate_events` table for per-adapter mean weight from the
    most recent run that recorded a gate.
    """
    import json as _json

    from dlm.store.paths import StorePath
    from dlm.train.gate.paths import gate_config_path

    assert isinstance(store, StorePath)
    cfg_path = gate_config_path(store)

    from dlm.metrics import queries as _queries
    from dlm.train.gate.module import GateMetadata

    events = _queries.latest_gate_events(store.root)
    # Divergence path: training raised before writing a config, but we
    # still emit one GateEvent per adapter with mode="diverged" so
    # operators can see the failure. Surface it even when the config
    # file is absent.
    if not cfg_path.exists():
        if events and events[0].mode == "diverged":
            return {
                "mode": "diverged",
                "adapter_names": [e.adapter_name for e in events],
                "input_dim": None,
                "hidden_proj_dim": None,
                "last_run_id": events[0].run_id,
                "per_adapter": [
                    {
                        "adapter_name": e.adapter_name,
                        "mean_weight": e.mean_weight,
                        "sample_count": e.sample_count,
                        "mode": e.mode,
                    }
                    for e in events
                ],
            }
        return None

    raw = _json.loads(cfg_path.read_text(encoding="utf-8"))
    meta = GateMetadata.from_json(raw)
    per_adapter: list[dict[str, object]] = []
    run_id: int | None = None
    if events:
        run_id = events[0].run_id
        per_adapter = [
            {
                "adapter_name": e.adapter_name,
                "mean_weight": e.mean_weight,
                "sample_count": e.sample_count,
                "mode": e.mode,
            }
            for e in events
        ]
    else:
        # No recorded events yet; fall back to the config so `dlm show`
        # still reports that a gate exists and in which mode.
        per_adapter = [{"adapter_name": name} for name in meta.adapter_names]
    return {
        "mode": meta.mode,
        "adapter_names": list(meta.adapter_names),
        "input_dim": meta.input_dim,
        "hidden_proj_dim": meta.hidden_proj_dim,
        "last_run_id": run_id,
        "per_adapter": per_adapter,
    }


def _summarize_preference_mining(store_root: Path) -> dict[str, object] | None:
    """Return the latest preference-mine summary for `dlm show --json`."""
    from dlm.metrics import queries as _queries

    totals = _queries.preference_mining_totals(store_root)
    if totals is None:
        return None
    last = _queries.latest_preference_mining(store_root)
    assert last is not None
    rows = _queries.preference_mining_for_run(store_root, last.run_id)
    return {
        "run_count": totals.run_count,
        "event_count": totals.event_count,
        "total_mined_pairs": totals.total_mined_pairs,
        "total_skipped_prompts": totals.total_skipped_prompts,
        "last_run_id": last.run_id,
        "last_run_event_count": len(rows),
        "last_event": _queries.preference_mining_to_dict([last])[0],
    }


def _summarize_base_security(base_model_key: str) -> dict[str, object] | None:
    """Surface security-sensitive base-model flags for `dlm show`.

    Today that's just `trust_remote_code` — a flag that causes the HF
    loader to execute Python from the model repo. We resolve the spec
    out of the in-process registry (no network: the resolver reads a
    frozen Python dict) so users can see which bases opt in without
    grepping source. Returns None when the key doesn't resolve (an
    `hf:...` escape hatch that isn't in the registry); the caller
    silently skips in that case.
    """
    from dlm.base_models import resolve as resolve_base_model
    from dlm.base_models.errors import BaseModelError

    try:
        spec = resolve_base_model(base_model_key, accept_license=True)
    except BaseModelError:
        return None
    return {
        "base_model": spec.key,
        "architecture": spec.architecture,
        "trust_remote_code": bool(spec.trust_remote_code),
    }


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
