"""Apply a template to a target `.dlm` path.

Backing for `dlm init --template <name>`. The flow:

1. Load the template pair (body + meta) from the gallery.
2. Parse the bundled body as a `.dlm` to validate it and grab its schema.
3. Mint a fresh `dlm_id` — two users running `dlm init --template coding-tutor`
   must get distinct stores, never collide on the bundled ULID.
4. Serialize the updated frontmatter + body to the target path.

Refuses to overwrite unless the caller passes `force=True`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dlm.doc.parser import parse_text
from dlm.doc.schema import DlmFrontmatter
from dlm.doc.sections import Section
from dlm.doc.serializer import serialize
from dlm.io.ulid import mint_ulid
from dlm.templates.errors import TemplateApplyError
from dlm.templates.registry import Template, load_template


@dataclass(frozen=True)
class ApplyResult:
    """Outcome of `apply_template`: the template applied + the fresh ULID."""

    template: Template
    dlm_id: str


def apply_template(
    name: str,
    target: Path,
    *,
    force: bool = False,
    gallery_dir: Path | None = None,
) -> ApplyResult:
    """Write a new `.dlm` at `target` based on the named gallery template.

    Returns an `ApplyResult` with the template and the freshly-minted
    `dlm_id`. Callers (typically `dlm init --template`) use the id to
    provision the per-store manifest.

    Refuses when `target` exists and `force` is False. Never overwrites
    silently — same policy as `dlm init` without `--template`.
    """
    if target.exists() and not force:
        raise TemplateApplyError(
            f"{target} already exists — pass force=True to overwrite",
        )

    template = load_template(name, gallery_dir=gallery_dir)
    parsed = parse_text(template.dlm_text)

    fresh_id = mint_ulid()
    new_fm = _replace_dlm_id(parsed.frontmatter, fresh_id)
    rendered = _serialize_with_frontmatter(new_fm, parsed.sections)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    return ApplyResult(template=template, dlm_id=fresh_id)


def _replace_dlm_id(fm: DlmFrontmatter, new_id: str) -> DlmFrontmatter:
    """Return a copy of `fm` with `dlm_id` replaced by `new_id`.

    `DlmFrontmatter` is a Pydantic model; `model_copy(update=...)` gives
    us a frozen-by-construction replacement without mutating the input.
    """
    return fm.model_copy(update={"dlm_id": new_id})


def _serialize_with_frontmatter(
    fm: DlmFrontmatter,
    sections: tuple[Section, ...],
) -> str:
    """Render the updated frontmatter + the template body."""
    # `serialize` consumes a `ParsedDlm`; we reconstruct one with the
    # rotated frontmatter and the original sections.
    from dlm.doc.parser import ParsedDlm

    parsed = ParsedDlm(frontmatter=fm, sections=sections)
    return serialize(parsed)
