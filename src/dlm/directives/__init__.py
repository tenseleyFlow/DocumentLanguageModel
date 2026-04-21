"""Training-time source directives — expand frontmatter
`training.sources` into synthesized `Section` values.

Flow:

    parsed = parse_file(dlm_path)
    result = expand_sources(parsed, base_path=dlm_path.parent)
    all_sections = tuple(parsed.sections) + result.sections
    # result.provenance feeds TrainingRunSummary.source_directives

Lazy package surface — no torch / transformers import footprint, so
`dlm show` and tests that don't train stay fast.
"""

from __future__ import annotations

from dlm.directives.errors import (
    DirectiveError,
    DirectivePathError,
    DirectivePolicyError,
)
from dlm.directives.expand import (
    ExpandResult,
    SourceProvenance,
    expand_sources,
)

__all__ = [
    "DirectiveError",
    "DirectivePathError",
    "DirectivePolicyError",
    "ExpandResult",
    "SourceProvenance",
    "expand_sources",
]
