"""Canonical on-disk layout for the learned gate.

Per-store shape:

    <store>/adapter/_gate/
        gate.safetensors     # nn.Module weights
        gate_config.json     # adapter order + hidden dim + mode

`_gate` is prefixed with an underscore so it can't collide with any
user-declared adapter name (the adapter-name grammar refuses a leading
underscore via `_ADAPTER_NAME_RE`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from dlm.store.paths import StorePath

_GATE_DIRNAME = "_gate"
_GATE_WEIGHTS_FILENAME = "gate.safetensors"
_GATE_CONFIG_FILENAME = "gate_config.json"


def gate_dir(store: StorePath) -> Path:
    """Return the gate's on-disk directory (no mkdir)."""
    return store.root / "adapter" / _GATE_DIRNAME


def gate_save_path(store: StorePath) -> Path:
    """Path to the serialized gate weights file."""
    return gate_dir(store) / _GATE_WEIGHTS_FILENAME


def gate_config_path(store: StorePath) -> Path:
    """Path to the gate metadata JSON."""
    return gate_dir(store) / _GATE_CONFIG_FILENAME
