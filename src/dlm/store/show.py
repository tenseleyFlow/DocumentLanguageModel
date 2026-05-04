"""Domain dispatcher for `dlm show`.

Aggregates the .dlm + store snapshot the CLI renders to text or JSON.
Callers (CLI, LSP doc-overview panel, future automation) build a
`StoreViewRequest`, call `gather_store_view`, and render the typed
`StoreView` themselves. The dispatcher does no console I/O;
`ManifestCorruptError` propagates so the caller can map it to its own
exit code or banner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dlm.directives import expand_sources as _expand_sources
from dlm.directives.errors import DirectiveError
from dlm.metrics import queries as _queries
from dlm.store.inspect import StoreInspection, inspect_store

if TYPE_CHECKING:
    from dlm.doc.parser import ParsedDlm
    from dlm.store.paths import StorePath


@dataclass(frozen=True)
class StoreViewRequest:
    """Inputs to `gather_store_view`."""

    parsed: ParsedDlm
    target_path: Path
    store: StorePath


@dataclass(frozen=True)
class StoreView:
    """Aggregated snapshot a `dlm show` caller renders to text or JSON.

    `inspection` is `None` when the store has no manifest yet
    (post-`dlm init`, pre-`dlm train`); the caller surfaces the
    "store: not yet initialized" path in that case. The summary dicts
    are JSON-safe and form the v1 contract for `dlm show --json`; any
    reshape is a version bump (recorded in `tests/golden/cli-json/`).
    """

    parsed_dlm_id: str
    parsed_base_model: str
    source_path: Path
    training_cache_config: dict[str, object]
    training_sources: list[dict[str, object]] | None
    discovered_configs: list[dict[str, object]]
    inspection: StoreInspection | None
    training_cache: dict[str, object] | None
    gate: dict[str, object] | None
    preference_mining: dict[str, object] | None
    base_security: dict[str, object] | None


def gather_store_view(req: StoreViewRequest) -> StoreView:
    """Walk the .dlm + its store and produce a `StoreView`.

    Raises `ManifestCorruptError` if the manifest is unparseable.
    """
    parsed = req.parsed
    base_path = req.target_path.resolve().parent
    training_sources, discovered_configs = _summarize_training_sources_and_discovered(
        parsed, base_path
    )
    cache_cfg = parsed.frontmatter.training.cache
    training_cache_config: dict[str, object] = {
        "enabled": cache_cfg.enabled,
        "max_bytes": cache_cfg.max_bytes,
        "prune_older_than_days": cache_cfg.prune_older_than_days,
    }

    if not req.store.manifest.exists():
        return StoreView(
            parsed_dlm_id=parsed.frontmatter.dlm_id,
            parsed_base_model=parsed.frontmatter.base_model,
            source_path=req.target_path.resolve(),
            training_cache_config=training_cache_config,
            training_sources=training_sources,
            discovered_configs=discovered_configs,
            inspection=None,
            training_cache=None,
            gate=None,
            preference_mining=None,
            base_security=None,
        )

    inspection = inspect_store(req.store, source_path=req.target_path.resolve())
    training_cache = _summarize_training_cache(req.store.tokenized_cache_dir, req.store.root)
    gate = _summarize_gate(req.store)
    preference_mining = _summarize_preference_mining(req.store.root)
    base_security = _summarize_base_security(parsed.frontmatter.base_model)

    return StoreView(
        parsed_dlm_id=parsed.frontmatter.dlm_id,
        parsed_base_model=parsed.frontmatter.base_model,
        source_path=req.target_path.resolve(),
        training_cache_config=training_cache_config,
        training_sources=training_sources,
        discovered_configs=discovered_configs,
        inspection=inspection,
        training_cache=training_cache,
        gate=gate,
        preference_mining=preference_mining,
        base_security=base_security,
    )


def _summarize_training_sources_and_discovered(
    parsed: ParsedDlm, base_path: Path
) -> tuple[list[dict[str, object]] | None, list[dict[str, object]]]:
    """Best-effort `training.sources` expansion + `.dlm/training.yaml` discovery.

    Returns `(training_sources, discovered_configs)`. `training_sources`
    is None when the frontmatter declares no directives; otherwise
    declared records are returned even when expansion fails (so the
    show output stays useful for debugging a misconfigured directive).
    `discovered_configs` is always a list (empty when nothing was
    found or the expansion failed).
    """
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
        result = _expand_sources(parsed, base_path=base_path)
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

    None when the cache dir doesn't exist (store never trained with the
    cache, or pre-Sprint-31 layout). Cheap — reads the manifest only,
    not the entry files.
    """
    if not cache_dir.is_dir():
        return None
    from dlm.directives.cache import TokenizedCache

    cache = TokenizedCache.open(cache_dir)
    last = _queries.latest_tokenization(store_root)
    return {
        "path": str(cache_dir),
        "entry_count": cache.entry_count,
        "bytes": cache.total_bytes,
        "last_run_hit_rate": last.hit_rate if last else None,
        "last_run_id": last.run_id if last else None,
    }


def _summarize_gate(store: StorePath) -> dict[str, object] | None:
    """Return a JSON-friendly snapshot of the learned adapter gate.

    None when the store has no gate config and no diverged-gate events.
    Reads `gate_config.json` for mode + adapter order, and the
    `gate_events` table for per-adapter mean weight from the most
    recent run that recorded a gate.
    """
    import json as _json

    from dlm.train.gate.module import GateMetadata
    from dlm.train.gate.paths import gate_config_path

    cfg_path = gate_config_path(store)

    events = _queries.latest_gate_events(store.root)
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
    """Latest preference-mine summary for the JSON contract."""
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
    """Surface `trust_remote_code` flag from the base-model registry.

    Returns None when the key doesn't resolve (an `hf:...` escape hatch
    that isn't in the registry); the caller silently skips in that case.
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
