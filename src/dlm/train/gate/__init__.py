"""Learned MoE-style adapter gate.

The gate is a small MLP trained post-SFT that routes each prompt to a
weighted combination of the document's named adapters. First-pass
scope: one weight vector per input, applied uniformly across adapter
layers. Per-layer dynamic routing is research follow-up.

Public surface:

- `Gate`          — the `nn.Module` (Linear → GELU → Linear → Softmax)
- `GateConfig`    — runtime-persisted gate metadata (adapter order,
                    hidden dim, input dim, mode=trained/uniform)
- `train_gate`    — post-SFT training pass
- `load_gate`     — load a persisted gate for inference
- `gate_save_path` / `gate_config_path` — canonical store paths
- `GateTrainingError`, `GateConfigError` — raised from the trainer /
                    loader when inputs are malformed
"""

from __future__ import annotations

from dlm.train.gate.errors import GateConfigError, GateTrainingError
from dlm.train.gate.module import Gate, GateMetadata
from dlm.train.gate.paths import gate_config_path, gate_save_path
from dlm.train.gate.trainer import (
    GateTrainingResult,
    GateTrainingSample,
    load_gate,
    train_gate,
)

__all__ = [
    "Gate",
    "GateConfigError",
    "GateMetadata",
    "GateTrainingError",
    "GateTrainingResult",
    "GateTrainingSample",
    "gate_config_path",
    "gate_save_path",
    "load_gate",
    "train_gate",
]
