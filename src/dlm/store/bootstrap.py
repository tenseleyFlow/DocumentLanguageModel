"""Domain dispatcher for `dlm init`.

Lifts the scaffold-write → store-create → manifest-write pipeline out of
the CLI. Callers (CLI, LSP "Initialize from Template" command, future
automation) build an `InitRequest`, call `run_init`, and render the
typed `InitResult`. The dispatcher does no console I/O; template
errors propagate as `TemplateError` so the caller can map them to its
own exit code or banner.

The CLI keeps the user-interactive concerns: flag-mutex validation,
multimodal/audio default-base swap, `--template` peek for license-prompt
target, the GatedModelError → interactive-prompt → retry loop, and the
modality-consistency check. This dispatcher takes an already-resolved
`BaseModelSpec` plus an already-built `LicenseAcceptance | None`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from dlm.io.ulid import mint_ulid
from dlm.store.manifest import Manifest, save_manifest
from dlm.store.paths import StorePath, for_dlm
from dlm.templates import init as _templates_init

if TYPE_CHECKING:
    from pathlib import Path

    from dlm.base_models.license import LicenseAcceptance
    from dlm.base_models.schema import BaseModelSpec
    from dlm.templates.init import ApplyResult


class ScaffoldKind(StrEnum):
    """Body shape to write when no `--template` is given."""

    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"


@dataclass(frozen=True)
class InitRequest:
    """Inputs to `run_init`.

    `template_name` and `scaffold_kind` are mutually informative:
    when `template_name is not None`, the template's body wins and
    `scaffold_kind` is ignored. The CLI rejects `--template` combined
    with `--multimodal` / `--audio` before constructing the request.
    """

    path: Path
    spec: BaseModelSpec
    acceptance: LicenseAcceptance | None
    force: bool
    template_name: str | None
    scaffold_kind: ScaffoldKind


@dataclass(frozen=True)
class InitResult:
    """Outcome of `run_init`. `applied_template` is set iff the request
    carried a `template_name` (the dispatcher applied a gallery template
    rather than writing a scaffold)."""

    dlm_id: str
    store: StorePath
    applied_template: ApplyResult | None


def run_init(req: InitRequest) -> InitResult:
    """Apply a template (or write a scaffold), then provision the store."""
    if req.template_name is not None:
        applied = _templates_init.apply_template(
            req.template_name,
            req.path,
            force=req.force,
            accept_license=True,
        )
        dlm_id = applied.dlm_id
    else:
        applied = None
        dlm_id = mint_ulid()
        if req.scaffold_kind is ScaffoldKind.VISION:
            _write_init_scaffold_multimodal(req.path, req.spec.key, dlm_id)
        elif req.scaffold_kind is ScaffoldKind.AUDIO:
            _write_init_scaffold_audio(req.path, req.spec.key, dlm_id)
        else:
            _write_init_scaffold(req.path, req.spec.key, dlm_id)

    store = for_dlm(dlm_id)
    store.ensure_layout()
    save_manifest(
        store.manifest,
        Manifest(
            dlm_id=dlm_id,
            base_model=req.spec.key,
            base_model_revision=req.spec.revision,
            source_path=req.path.resolve(),
            license_acceptance=req.acceptance,
        ),
    )

    return InitResult(dlm_id=dlm_id, store=store, applied_template=applied)


def _write_init_scaffold(path: Path, base_model_key: str, dlm_id: str) -> None:
    """Write a minimal-but-valid text-only `.dlm` at `path`.

    Body has one PROSE paragraph + a commented instruction section so
    users see both section shapes on first open.
    """
    scaffold = f"""---
dlm_id: {dlm_id}
dlm_version: 1
base_model: {base_model_key}
---

# Your document title

Write prose here. It will train via continued pretraining (CPT) loss.

::instruction::

### Q
Your example question.

### A
Your example answer.
"""
    path.write_text(scaffold, encoding="utf-8")


def _write_init_scaffold_multimodal(path: Path, base_model_key: str, dlm_id: str) -> None:
    """Write a vision-language `.dlm` at `path`.

    Body shows the `::image::` attribute fence + a caption so users see
    the v10 grammar on first open. The placeholder path
    `figures/your-image.png` is deliberately non-existent — first
    `dlm train` refuses with a clear file-missing error, prompting the
    user to drop a real image in. Friendlier than committing an inert
    sample users might not notice isn't theirs.

    `dlm_version: 10` because IMAGE sections require schema v10.
    """
    scaffold = f"""---
dlm_id: {dlm_id}
dlm_version: 10
base_model: {base_model_key}
---

# Your document title

Write prose here. It will train via continued pretraining (CPT) loss.

::image path="figures/your-image.png" alt="short description"::
Caption text describing the image. Training rows bundle the image
with this caption as `<image>\\n<caption>`.

::instruction::

### Q
What is in this image?

### A
Describe what the image shows.
"""
    path.write_text(scaffold, encoding="utf-8")


def _write_init_scaffold_audio(path: Path, base_model_key: str, dlm_id: str) -> None:
    """Write an audio-language `.dlm` at `path`.

    Body shows the `::audio::` attribute fence with the sibling-
    transcript-friendly `transcript="..."` form so users see the v11
    grammar on first open. The placeholder path `clips/your-clip.wav`
    is deliberately non-existent — first `dlm train` refuses with a
    clear file-missing error rather than silently training on an inert
    sample.

    `dlm_version: 11` because AUDIO sections require schema v11.
    """
    scaffold = f"""---
dlm_id: {dlm_id}
dlm_version: 11
base_model: {base_model_key}
---

# Your document title

Write prose here. It will train via continued pretraining (CPT) loss.

::audio path="clips/your-clip.wav" transcript="Transcript of the audio clip."::

::instruction::

### Q
What was said in this recording?

### A
Describe what you hear in the audio.
"""
    path.write_text(scaffold, encoding="utf-8")
