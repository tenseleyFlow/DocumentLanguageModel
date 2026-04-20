"""Route sections to named adapters.

Rules, keyed off the `#adapter` fence suffix extracted by the parser:

- **PROSE, no suffix**: trains all adapters. Prose is the shared
  "domain vocabulary" layer — users don't want to copy it per adapter.
- **PROSE, `#name` suffix**: trains only `name`. Rare but supported —
  a knowledge-specific prose chunk that shouldn't leak into tone.
- **INSTRUCTION / PREFERENCE, no suffix**: trains the first-declared
  adapter (the implicit "default"). Preserves the single-adapter
  single-doc shape.
- **INSTRUCTION / PREFERENCE, `#name` suffix**: trains only `name`.
  Unknown names raise `UnknownAdapterError`.

The router does not turn sections into rows itself — it returns the
sections assigned to the adapter, and the caller pipes them through
`dlm.data.sections_to_rows`. Keeping the split lets unit tests assert
routing without exercising the row-shape translator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dlm.doc.sections import Section, SectionType

if TYPE_CHECKING:
    from dlm.doc.parser import ParsedDlm


class UnknownAdapterError(ValueError):
    """A section references an adapter name not declared in `training.adapters`."""


@dataclass(frozen=True)
class RoutingPlan:
    """Per-adapter section assignments — precomputable once per parse.

    `by_adapter[name]` is the list of sections that should train adapter
    `name`. `unrouted` is the residual (sections with a `#unknown`
    suffix) — empty after `build_plan` in the happy path; populated
    only on validation error paths that want to report what went wrong.
    """

    by_adapter: dict[str, list[Section]]
    unrouted: list[Section]


def declared_adapter_names(parsed: ParsedDlm) -> list[str]:
    """Return the declared adapter names in first-appearance (dict) order.

    Single-adapter docs (`training.adapters` absent) return a single
    entry `["default"]` — the implicit name the store layout uses.
    """
    adapters = parsed.frontmatter.training.adapters
    if adapters is None:
        return ["default"]
    return list(adapters)


def build_plan(parsed: ParsedDlm) -> RoutingPlan:
    """Compute the per-adapter section assignments for a parsed document.

    Raises `UnknownAdapterError` on the first section whose `#name`
    suffix doesn't appear in `training.adapters`. Single-adapter docs
    (no `adapters` block) accept no `#name` suffix on any section —
    any routed section in that shape is rejected.
    """
    names = declared_adapter_names(parsed)
    declared = set(names)
    default_name = names[0]

    by_adapter: dict[str, list[Section]] = {name: [] for name in names}
    unrouted: list[Section] = []

    for section in parsed.sections:
        target = section.adapter
        if target is not None and target not in declared:
            raise UnknownAdapterError(
                f"section at line {section.start_line} routes to adapter "
                f"'#{target}' which is not declared in training.adapters "
                f"(declared: {sorted(declared)})"
            )

        if target is None:
            if section.type is SectionType.PROSE:
                # Prose fans out to every adapter.
                for name in names:
                    by_adapter[name].append(section)
            else:
                # Instruction/preference rows default to the first-declared.
                by_adapter[default_name].append(section)
        else:
            by_adapter[target].append(section)

    return RoutingPlan(by_adapter=by_adapter, unrouted=unrouted)


def sections_for(parsed: ParsedDlm, adapter_name: str) -> list[Section]:
    """Shorthand: `build_plan(parsed).by_adapter[adapter_name]`.

    Raises `UnknownAdapterError` if `adapter_name` isn't declared, or
    `KeyError` equivalent behavior — callers can prefer `build_plan`
    when they need to iterate all adapters at once.
    """
    plan = build_plan(parsed)
    if adapter_name not in plan.by_adapter:
        raise UnknownAdapterError(
            f"adapter {adapter_name!r} not declared "
            f"(declared: {sorted(plan.by_adapter)})"
        )
    return plan.by_adapter[adapter_name]
