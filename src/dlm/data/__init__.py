"""Dataset assembly — turn parsed `.dlm` sections into a ready-to-train dataset.

See Sprint 07 for the design. Heavy imports (`datasets`, `transformers`,
`trl`, `peft`) are deferred to the call sites that actually use them,
so `import dlm.data` stays cheap even when the training stack isn't
installed.
"""

from __future__ import annotations

from dlm.data.errors import (
    DataError,
    DataFormatError,
    InstructionParseError,
    PreferenceParseError,
    SectionParseError,
    TokenizerBringupError,
)

__all__ = [
    "DataError",
    "DataFormatError",
    "InstructionParseError",
    "PreferenceParseError",
    "SectionParseError",
    "TokenizerBringupError",
]
