"""Template gallery — curated starter `.dlm` files plus metadata.

Public surface:

- `Template`, `TemplateMeta` — dataclass / pydantic model for a template
- `list_bundled` — enumerate the in-tree bundled gallery
- `load_template` — fetch a `Template` by name (bundled first, cache second)
- `apply_template` — write a new `.dlm` at a target path based on a named
  template; mints a fresh `dlm_id`
- Errors: `TemplateError`, `TemplateNotFoundError`, `TemplateMetaError`,
  `TemplateApplyError`

The registry is offline-first: the bundled gallery under
`<repo>/templates/` always loads. Remote refresh (sprint spec §"Remote
fetch") is scaffolded under `fetcher.py` but requires a pinned upstream
that doesn't exist yet — see the sprint file for the honest `[~]` surface.
"""

from __future__ import annotations

from dlm.templates.errors import (
    TemplateApplyError,
    TemplateError,
    TemplateMetaError,
    TemplateNotFoundError,
)
from dlm.templates.init import ApplyResult, apply_template
from dlm.templates.registry import (
    Template,
    bundled_templates_dir,
    list_bundled,
    load_template,
)
from dlm.templates.schema import TemplateMeta

__all__ = [
    "ApplyResult",
    "Template",
    "TemplateApplyError",
    "TemplateError",
    "TemplateMeta",
    "TemplateMetaError",
    "TemplateNotFoundError",
    "apply_template",
    "bundled_templates_dir",
    "list_bundled",
    "load_template",
]
