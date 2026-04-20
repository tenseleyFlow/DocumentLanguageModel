"""Template registry: enumerate and load templates from the bundled gallery.

The bundled gallery lives at `src/dlm/templates/gallery/` and is shipped
with the wheel via hatch's package layout (`packages = ["src/dlm"]`).
Each template is a pair of files with a shared stem:

    gallery/coding-tutor.dlm          # the .dlm body + frontmatter
    gallery/coding-tutor.meta.yaml    # the TemplateMeta sidecar

Invalid templates (missing sidecar, meta schema drift, unparseable .dlm)
are dropped from the listing with a log warning — never silently served.
This is the sprint-spec §"Validation at pull time" rule applied to the
local gallery as well.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Final

import yaml
from pydantic import ValidationError

from dlm.templates.errors import TemplateMetaError, TemplateNotFoundError
from dlm.templates.schema import TemplateMeta

_LOG = logging.getLogger(__name__)

_GALLERY_PACKAGE: Final[str] = "dlm.templates.gallery"


@dataclass(frozen=True)
class Template:
    """A gallery template: the `.dlm` body + its meta sidecar + source path."""

    name: str
    meta: TemplateMeta
    dlm_text: str
    source_path: Path


def bundled_templates_dir() -> Path:
    """Return the filesystem path to the bundled gallery directory.

    Uses `importlib.resources.files` so the path works both from an
    editable checkout and from an installed wheel.
    """
    return Path(str(resources.files(_GALLERY_PACKAGE)))


def list_bundled(gallery_dir: Path | None = None) -> list[Template]:
    """Return every validated template in the bundled gallery, name-sorted.

    Templates that fail meta validation or .dlm loading are logged and
    skipped — `list_bundled()` never raises on a single bad template.
    """
    root = gallery_dir if gallery_dir is not None else bundled_templates_dir()
    templates: list[Template] = []
    for dlm_path in sorted(root.glob("*.dlm")):
        name = dlm_path.stem
        try:
            templates.append(_load_pair(name, dlm_path, root))
        except TemplateMetaError as exc:
            _LOG.warning("templates: dropping %r — %s", name, exc)
    return templates


def load_template(name: str, gallery_dir: Path | None = None) -> Template:
    """Return a single validated template by name.

    Raises `TemplateNotFoundError` if the name has no matching `.dlm` in
    the gallery. Raises `TemplateMetaError` if the `.meta.yaml` sidecar
    is missing, malformed, or schema-invalid.
    """
    root = gallery_dir if gallery_dir is not None else bundled_templates_dir()
    dlm_path = root / f"{name}.dlm"
    if not dlm_path.exists():
        raise TemplateNotFoundError(
            f"no template named {name!r} in {root}",
        )
    return _load_pair(name, dlm_path, root)


def _load_pair(name: str, dlm_path: Path, gallery_dir: Path) -> Template:
    meta_path = gallery_dir / f"{name}.meta.yaml"
    if not meta_path.exists():
        raise TemplateMetaError(
            f"template {name!r} is missing its meta sidecar at {meta_path}",
        )

    try:
        raw = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TemplateMetaError(
            f"template {name!r} meta YAML is malformed: {exc}",
        ) from exc
    if not isinstance(raw, dict):
        raise TemplateMetaError(
            f"template {name!r} meta must be a YAML mapping, got {type(raw).__name__}",
        )

    try:
        meta = TemplateMeta.model_validate(raw)
    except ValidationError as exc:
        raise TemplateMetaError(
            f"template {name!r} meta failed schema validation: {exc}",
        ) from exc

    if meta.name != name:
        raise TemplateMetaError(
            f"template {name!r} meta.name field is {meta.name!r} — must match filename stem",
        )

    dlm_text = dlm_path.read_text(encoding="utf-8")
    return Template(name=name, meta=meta, dlm_text=dlm_text, source_path=dlm_path)
