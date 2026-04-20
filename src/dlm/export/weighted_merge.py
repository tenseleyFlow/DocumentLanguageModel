"""Weighted-merge helper for multi-adapter export.

`--adapter-mix knowledge:1.0,tone:0.5` composes the named adapters
into one exportable checkpoint. Two public surfaces:

- `parse_mix_spec("knowledge:1.0,tone:0.5") -> [("knowledge", 1.0),
  ("tone", 0.5)]` — pure string parsing, validates shape + ranges.
- `build_weighted_merged(base_model, store, spec, mix)` — heavy: loads
  each named adapter via `PeftModel.from_pretrained`/`add_adapter`,
  then `add_weighted_adapter(names, weights, "_export_merged",
  combination_type="linear")`. Returns the PEFT model with
  `_export_merged` active. Covered by the slow integration test.

Caveats (from `.docs/findings.md` §4):

- `add_weighted_adapter` is LoRA-only. QLoRA adapters must be loaded
  against a dequantized base, or the composition precision becomes
  unsafe — the runner's existing merge-safety gate (pitfall #3)
  catches this and refuses unless `--dequantize` is explicit.
- `combination_type="linear"` is the default; SVD is available but
  costly. We pin linear here; future sprints can expose the knob.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from dlm.export.errors import ExportError

if TYPE_CHECKING:
    from pathlib import Path

    from dlm.base_models import BaseModelSpec
    from dlm.store.paths import StorePath


_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
"""Same grammar the schema + parser + store layout share."""

_MERGED_ADAPTER_NAME = "_export_merged"


class InvalidMixSpecError(ExportError):
    """`--adapter-mix` string didn't parse or violated the grammar."""


@dataclass(frozen=True)
class MixEntry:
    name: str
    weight: float


def parse_mix_spec(spec_str: str) -> list[MixEntry]:
    """Parse `name:weight,name:weight` into typed entries.

    Rules:

    - Non-empty string; at least one entry.
    - Each entry is `<name>:<weight>`. `<name>` must match the
      adapter-name grammar. `<weight>` must parse to a float ≥ 0
      (negative weights compose surprisingly; we refuse up-front).
    - Duplicate names are rejected.

    Order is preserved so the caller can pass entries straight into
    `add_weighted_adapter` without resorting.
    """
    raw = spec_str.strip()
    if not raw:
        raise InvalidMixSpecError(
            "--adapter-mix: empty spec; pass at least one `name:weight` entry"
        )

    entries: list[MixEntry] = []
    seen: set[str] = set()
    for piece in raw.split(","):
        token = piece.strip()
        if not token:
            raise InvalidMixSpecError(
                f"--adapter-mix: empty entry in spec {spec_str!r}"
            )
        if ":" not in token:
            raise InvalidMixSpecError(
                f"--adapter-mix: entry {token!r} is missing a weight "
                "(shape: `name:weight`)"
            )
        name, _, weight_str = token.rpartition(":")
        name = name.strip()
        weight_str = weight_str.strip()
        if not _NAME_RE.fullmatch(name):
            raise InvalidMixSpecError(
                f"--adapter-mix: adapter name {name!r} is not valid "
                f"(must match {_NAME_RE.pattern})"
            )
        if name in seen:
            raise InvalidMixSpecError(
                f"--adapter-mix: adapter {name!r} appears twice"
            )
        seen.add(name)
        try:
            weight = float(weight_str)
        except ValueError as exc:
            raise InvalidMixSpecError(
                f"--adapter-mix: weight {weight_str!r} for adapter "
                f"{name!r} is not a number"
            ) from exc
        if weight < 0:
            raise InvalidMixSpecError(
                f"--adapter-mix: weight {weight} for adapter {name!r} "
                "is negative (must be >= 0)"
            )
        entries.append(MixEntry(name=name, weight=weight))

    return entries


def validate_mix_against_declared(
    entries: list[MixEntry], declared: set[str]
) -> None:
    """Refuse mix entries that reference adapters not in `training.adapters`.

    Single source of error messaging so the CLI and the runner both
    report the same shape.
    """
    unknown = [e.name for e in entries if e.name not in declared]
    if unknown:
        raise InvalidMixSpecError(
            f"--adapter-mix references adapter(s) {unknown} not declared "
            f"(declared: {sorted(declared)})"
        )


def build_weighted_merged(  # pragma: no cover - heavy path
    base_model: Any,
    store: StorePath,
    spec: BaseModelSpec,
    entries: list[MixEntry],
) -> Any:
    """Load each adapter, combine them via `add_weighted_adapter`.

    Returns a `PeftModel` with adapter `_export_merged` active. The
    caller writes the merged adapter to an ephemeral directory and
    hands it to the existing GGUF pipeline.
    """
    from peft import PeftModel

    if not entries:
        raise InvalidMixSpecError(
            "build_weighted_merged: received empty mix — validator did not run"
        )

    first = entries[0]
    first_path = _resolve_or_raise(store, first.name)
    model = PeftModel.from_pretrained(
        base_model, str(first_path), adapter_name=first.name
    )
    for extra in entries[1:]:
        path = _resolve_or_raise(store, extra.name)
        model.load_adapter(str(path), adapter_name=extra.name)

    model.add_weighted_adapter(
        adapters=[e.name for e in entries],
        weights=[e.weight for e in entries],
        adapter_name=_MERGED_ADAPTER_NAME,
        combination_type="linear",
    )
    model.set_adapter(_MERGED_ADAPTER_NAME)
    return model


def _resolve_or_raise(store: StorePath, name: str) -> Path:  # pragma: no cover
    """Return the current-adapter version dir for `name`, or raise ExportError."""
    path = store.resolve_current_adapter_for(name)
    if path is None or not path.exists():
        raise ExportError(
            f"adapter {name!r} has no committed version under "
            f"{store.adapter_current_pointer_for(name)}; run `dlm train` first."
        )
    return path


def resolve_first_source_path(store: StorePath, entries: list[MixEntry]) -> Path:
    """Return the on-disk version dir for the first mix entry.

    Used by the CLI to hand a `tokenizer_source` to `save_merged_to_tmp`
    so the merged output dir includes tokenizer files for the downstream
    preflight (audit-07 B2). All source adapters share the same base
    model + tokenizer (enforced by the frontmatter `base_model` being
    single-valued), so any source is interchangeable — we pick the first.
    """
    if not entries:
        raise InvalidMixSpecError(
            "resolve_first_source_path: empty mix"
        )
    return _resolve_or_raise(store, entries[0].name)


_TOKENIZER_FILES: Final[tuple[str, ...]] = (
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
)
"""Files we copy from a source adapter dir into the merged dir so the
downstream GGUF pipeline's preflight passes (audit-07 B2)."""


def save_merged_to_tmp(  # pragma: no cover - heavy path
    merged_model: Any,
    tmp_dir: Path,
    *,
    tokenizer_source: Path | None = None,
    training_run_source: Path | None = None,
) -> Path:
    """Save the composite `_export_merged` adapter to `tmp_dir`.

    Returns the path for consumption as `adapter_path_override` in
    `run_export`. Uses PEFT's `save_pretrained` with the merged
    adapter selected so the resulting dir mirrors a normal adapter
    checkpoint (adapter_config.json + adapter_model.safetensors).

    Additionally copies tokenizer files from `tokenizer_source` (one
    of the source adapter dirs) so the export preflight's
    `check_tokenizer_vocab` finds `tokenizer_config.json`. Without
    this, the merged export dies in preflight (audit-07 B2).

    `training_run_source` — when set, copies `training_run.json` into
    the merged dir so `check_was_adapter_qlora` fires the merge-safety
    gate correctly on QLoRA-derived composites.
    """
    import shutil

    tmp_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        str(tmp_dir), selected_adapters=[_MERGED_ADAPTER_NAME]
    )

    if tokenizer_source is not None:
        for fname in _TOKENIZER_FILES:
            src = tokenizer_source / fname
            if src.exists():
                shutil.copy2(src, tmp_dir / fname)

    if training_run_source is not None:
        src = training_run_source / "training_run.json"
        if src.exists():
            shutil.copy2(src, tmp_dir / "training_run.json")

    return tmp_dir
