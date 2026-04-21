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

from pydantic import BaseModel, ConfigDict, Field, field_validator


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

    `weights` scales per-row training exposure by `(tag_key, tag_value)`.
    Resolution is multiplicative across tag keys and merges shallow-to-
    deep like `metadata` — deeper configs override shallower keys. A
    row with tags `{domain: auth, generated: true}` under a tree where
    the root sets `weights.domain.auth = 2.0` and a subtree sets
    `weights.generated.true = 0.5` ends up at effective weight 1.0.
    Rows with no matching tag/value get weight 1.0.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dlm_training_version: Literal[1] = 1
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    exclude_defaults: bool = True
    metadata: Mapping[str, str] = Field(default_factory=dict)
    weights: Mapping[str, Mapping[str, float]] = Field(default_factory=dict)

    @field_validator("weights")
    @classmethod
    def _validate_weights(
        cls, value: Mapping[str, Mapping[str, float]]
    ) -> Mapping[str, Mapping[str, float]]:
        for tag_key, inner in value.items():
            for tag_value, scale in inner.items():
                if scale < 0:
                    raise ValueError(
                        f"weights[{tag_key!r}][{tag_value!r}] must be ≥ 0, "
                        f"got {scale}. Negative weights don't have a "
                        f"well-defined meaning under row-repetition expansion."
                    )
        return value
