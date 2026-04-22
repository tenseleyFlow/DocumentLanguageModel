"""GGUF export pipeline — convert trained adapters to Ollama-consumable files.

Heavy imports (`torch`, `peft`,
`transformers`) stay deferred; subprocess calls to the vendored
`llama.cpp` tools go through `dlm.export.quantize.run_checked`.
"""

from __future__ import annotations

from dlm.export.errors import (
    ExportError,
    ExportManifestError,
    PreflightError,
    SubprocessError,
    UnsafeMergeError,
    VendoringError,
)
from dlm.export.manifest import (
    EXPORT_MANIFEST_FILENAME,
    ExportArtifact,
    ExportManifest,
    load_export_manifest,
    save_export_manifest,
)
from dlm.export.plan import (
    DEFAULT_QUANT,
    QUANT_BYTES_PER_PARAM,
    ExportPlan,
    QuantLevel,
    resolve_export_plan,
    valid_quants,
)
from dlm.export.runner import ExportResult, run_export

__all__ = [
    "DEFAULT_QUANT",
    "EXPORT_MANIFEST_FILENAME",
    "ExportArtifact",
    "ExportError",
    "ExportManifest",
    "ExportManifestError",
    "ExportPlan",
    "ExportResult",
    "PreflightError",
    "QUANT_BYTES_PER_PARAM",
    "QuantLevel",
    "SubprocessError",
    "UnsafeMergeError",
    "VendoringError",
    "load_export_manifest",
    "resolve_export_plan",
    "run_export",
    "save_export_manifest",
    "valid_quants",
]
