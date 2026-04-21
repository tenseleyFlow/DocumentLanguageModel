"""`.dlm/training.yaml` per-codebase config.

Distinct from the `.dlm` frontmatter schema — different file, different
shape, different namespace. A codebase drops this alongside its source
to declare what `dlm train` should ingest when a directive descends
into the tree. The nearest-ancestor `.dlm/training.yaml` wins for each
file under its subtree, matching `.gitignore`'s resolution semantics.

See `dlm.directives.merge` for the full precedence table and
`docs/format/dlm-training-yaml.md` for the user-facing reference.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DlmTrainingConfig(BaseModel):
    """Per-subtree training config discovered at `.dlm/training.yaml`.

    All fields optional — an empty config (just `dlm_training_version: 1`)
    is legal and means "no refinement at this level", which is useful
    as a placeholder marker while drafting.

    `exclude_defaults` controls whether `dlm.directives.defaults.DEFAULT_EXCLUDES`
    applies at this subtree. Most users want it True (secrets, VCS,
    build artifacts skipped automatically); trees that legitimately
    train on e.g. generated code can set it False to opt out.

    `metadata` flows onto every `Section` synthesized from this
    subtree via `Section.tags`. Tags do NOT affect `section_id`, so
    adjusting metadata doesn't invalidate the replay corpus.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dlm_training_version: Literal[1] = 1
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    exclude_defaults: bool = True
    metadata: Mapping[str, str] = Field(default_factory=dict)
