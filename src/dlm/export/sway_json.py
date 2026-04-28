"""Cross-repo bridge: emit a ready-to-run ``sway.yaml`` next to a dlm export.

Closes the gap where users who train via dlm then evaluate via sway
had to run two separate commands. With ``dlm export
--emit-sway-json``, the user runs::

    dlm export myadapter.dlm --target ollama --emit-sway-json

and finds both the GGUF/Modelfile *and* a ``sway.yaml`` at
``<export-dir>/sway.yaml`` ready for ``sway run`` against any
HF/MLX/HTTP backend.

## Why this lives in dlm and not sway

The autogen logic — translating ``.dlm`` sections to a sway suite spec
— belongs to sway and stays there: importing
``dlm_sway.integrations.dlm.autogen.build_spec_dict``. dlm's job is
just "after a successful export, also run that autogen and write the
result to disk." Keeping the orchestration on the dlm side means
``dlm export`` is the one CLI a user touches; sway is a runtime
collaborator, not a separate phase the user has to remember.

## Optional-dep posture

``dlm-sway`` is in dlm's ``[sway]`` optional extra. Without it
installed, the import below raises ``ImportError`` and we surface a
typed :class:`SwayJsonExportError` whose message tells the user
exactly what to install. dlm's ``[sway]`` extra pulls plain
``dlm-sway`` (NOT ``dlm-sway[dlm]``) — sway already optionally
depends on dlm via its own ``[dlm]`` extra, so pulling the round-trip
extra would create a resolver cycle. Plain ``dlm-sway`` is enough:
``build_spec_dict`` lives in ``integrations/dlm/`` but doesn't import
``dlm`` to do its work; it operates on the parsed ``.dlm`` file we
hand it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dlm.export.errors import ExportError

logger = logging.getLogger(__name__)


class SwayJsonExportError(ExportError):
    """Raised when ``--emit-sway-json`` can't produce a sway.yaml.

    Two common cases:

    - ``dlm-sway`` not installed: the user passed ``--emit-sway-json``
      but didn't ``pip install 'dlm[sway]'`` (or installed dlm without
      the sway extra). The error message tells them exactly what to
      install.
    - ``build_spec_dict`` raised: typically a parse failure on the
      ``.dlm`` itself, surfaced through ``SwayError``. We re-wrap so
      the dlm CLI's exception handler sees a familiar ``ExportError``
      subclass and exits cleanly.
    """


def write_sway_json(dlm_path: Path, export_dir: Path) -> Path:
    """Generate ``<export_dir>/sway.yaml`` from ``dlm_path``.

    Parameters
    ----------
    dlm_path:
        The source ``.dlm`` document being exported. Used as the
        autogen input and recorded as ``dlm_source`` in the emitted
        spec for downstream traceability.
    export_dir:
        Directory the GGUF artifacts already wrote into. The new
        ``sway.yaml`` lands at ``export_dir / "sway.yaml"``.

    Returns
    -------
    Path
        Absolute path to the written ``sway.yaml``.

    Raises
    ------
    SwayJsonExportError
        ``dlm-sway`` extra not installed, or the autogen call failed.
    """
    dlm_path = Path(dlm_path).expanduser().resolve()
    export_dir = Path(export_dir).expanduser().resolve()

    try:
        # Both imports are required: ``resolve_dlm`` parses the .dlm
        # into the DlmHandle that ``build_spec_dict`` consumes.
        # The combined ``[import-not-found,unused-ignore]`` keeps mypy
        # happy across two install postures: when dlm-sway is missing
        # (``import-not-found`` fires), and when CI installs the
        # ``[sway]`` extra (the import succeeds — without the
        # ``unused-ignore`` paired code, mypy then complains that the
        # ``import-not-found`` ignore is itself unused).
        from dlm_sway.integrations.dlm.autogen import (  # type: ignore[import-not-found,unused-ignore]
            build_spec_dict,
        )
        from dlm_sway.integrations.dlm.resolver import (  # type: ignore[import-not-found,unused-ignore]
            resolve_dlm,
        )
    except ImportError as exc:  # pragma: no cover — env-dep branch
        raise SwayJsonExportError(
            "--emit-sway-json requires the dlm-sway integration. Install "
            "with: pip install 'dlm[sway]' (or `pip install dlm-sway` if "
            "dlm is editable). dlm-sway must be on PyPI as `dlm-sway>=0.1.0`."
        ) from exc

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover — pyyaml is in dlm core
        raise SwayJsonExportError(
            "PyYAML missing; this should never happen — it's a core dlm dep."
        ) from exc

    try:
        handle = resolve_dlm(dlm_path)
        spec_dict = build_spec_dict(handle, dlm_source=str(dlm_path))
    except Exception as exc:
        # Catch broadly so any sway-side parse / format error surfaces
        # as a typed dlm error the CLI's existing handler can render.
        raise SwayJsonExportError(
            f"sway autogen failed for {dlm_path}: {type(exc).__name__}: {exc}"
        ) from exc

    out_path = export_dir / "sway.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ``sort_keys=False`` preserves the dict insertion order
    # ``build_spec_dict`` produced (version → models → defaults →
    # suite). Looks like a hand-authored sway.yaml.
    yaml_text = yaml.safe_dump(spec_dict, sort_keys=False)
    out_path.write_text(yaml_text, encoding="utf-8")
    logger.debug("wrote %s (%d bytes)", out_path, len(yaml_text))
    return out_path
