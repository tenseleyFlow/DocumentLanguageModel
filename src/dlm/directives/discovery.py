"""Walk a source tree and collect every `.dlm/` configuration.

One `DiscoveredConfig` per `.dlm/` directory found under the walk
root. Each config aggregates both `.dlm/training.yaml` (parsed as
`DlmTrainingConfig`) and `.dlm/ignore` (parsed as a tuple of
`IgnoreRule`). Either or both may be absent — the presence of the
`.dlm/` directory alone is enough to produce a record (useful when a
user wants a pure drive-by `.dlm/ignore` without writing YAML).

Results are sorted by anchor path length ascending, so parents
appear before descendants. This matches the resolution order in
`dlm.directives.merge.effective_config_for`.

Malformed YAML or broken lines in `.dlm/ignore` log + degrade — the
walk never fails. The CLI has no way to recover from a mid-train
discovery crash, so tolerance here is load-bearing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from dlm.directives.ignore_parser import IgnoreRule, parse_ignore_file
from dlm.directives.schema import DlmTrainingConfig
from dlm.io.text import DlmEncodingError, read_text

_LOG = logging.getLogger(__name__)

_CONFIG_FILENAME = "training.yaml"
_IGNORE_FILENAME = "ignore"


@dataclass(frozen=True)
class DiscoveredConfig:
    """Aggregated `.dlm/` config at one anchor directory.

    `anchor` is the directory that *contains* the `.dlm/` dir (i.e.
    the repo root, or a subtree root). Relative paths in
    `config.include` / `config.exclude` and ignore rules resolve
    against this anchor.

    Both `config` and `ignore_rules` can be empty — a bare `.dlm/`
    directory with no files inside still produces a (no-op)
    DiscoveredConfig, letting users mark subtrees explicitly without
    writing YAML.
    """

    anchor: Path
    config: DlmTrainingConfig | None
    ignore_rules: tuple[IgnoreRule, ...]


def discover_configs(root: Path) -> tuple[DiscoveredConfig, ...]:
    """Walk `root` top-down and return a `DiscoveredConfig` per `.dlm/`.

    `root` itself is included — if `<root>/.dlm/` exists, it becomes
    the first (shallowest) discovered config. Each deeper `.dlm/`
    dir produces an additional record.

    Results are sorted by anchor path depth ascending so callers
    iterating can apply parent rules before child rules.
    """
    discovered: list[DiscoveredConfig] = []

    if not root.is_dir():
        return ()

    for dlm_dir in sorted(root.rglob(".dlm")):
        if not dlm_dir.is_dir():
            continue
        anchor = dlm_dir.parent
        config = _load_training_yaml(dlm_dir / _CONFIG_FILENAME)
        ignore_rules = _load_ignore(dlm_dir / _IGNORE_FILENAME)
        discovered.append(
            DiscoveredConfig(
                anchor=anchor, config=config, ignore_rules=ignore_rules
            )
        )

    discovered.sort(key=lambda d: len(d.anchor.as_posix()))
    return tuple(discovered)


def _load_training_yaml(path: Path) -> DlmTrainingConfig | None:
    """Load + validate a `.dlm/training.yaml`. Missing file → None.

    Malformed YAML, schema violations, or encoding errors log one
    warning and return None. The anchor still produces a
    DiscoveredConfig (just with `config=None`), so a neighboring
    `.dlm/ignore` at the same anchor keeps working.
    """
    if not path.is_file():
        return None
    try:
        text = read_text(path)
    except DlmEncodingError as exc:
        _LOG.warning("discovery: %s: not UTF-8 (%s); skipping config", path, exc)
        return None

    try:
        raw = yaml.safe_load(text) if text.strip() else {}
    except yaml.YAMLError as exc:
        _LOG.warning("discovery: %s: invalid YAML (%s); skipping config", path, exc)
        return None

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        _LOG.warning(
            "discovery: %s: top-level must be a mapping, got %s; skipping config",
            path,
            type(raw).__name__,
        )
        return None

    try:
        return DlmTrainingConfig.model_validate(raw)
    except ValidationError as exc:
        _LOG.warning(
            "discovery: %s: schema violation (%s); skipping config", path, exc
        )
        return None


def _load_ignore(path: Path) -> tuple[IgnoreRule, ...]:
    """Load + parse a `.dlm/ignore`. Missing file → empty tuple.

    The parser itself never raises; malformed lines log + skip. An
    unreadable file (encoding error) logs once and degrades to empty
    rules.
    """
    if not path.is_file():
        return ()
    try:
        text = read_text(path)
    except DlmEncodingError as exc:
        _LOG.warning("discovery: %s: not UTF-8 (%s); skipping ignore", path, exc)
        return ()
    return parse_ignore_file(text)
