"""Optional metric sinks — TensorBoard + W&B.

Both packages are optional-dependency installs under the
`observability` extra. Sinks lazy-import; a caller that doesn't set
`--tensorboard` / `--wandb` pays nothing.
"""

from __future__ import annotations

from dlm.metrics.sinks.tensorboard import TensorBoardSink, tensorboard_available
from dlm.metrics.sinks.wandb import WandbSink, wandb_available

__all__ = [
    "TensorBoardSink",
    "WandbSink",
    "tensorboard_available",
    "wandb_available",
]
