"""Dataset assembly — turn parsed `.dlm` sections into a ready-to-train dataset.

Heavy imports (`datasets`, `transformers`, `trl`, `peft`) are deferred
to the call sites that actually use them, so `import dlm.data` stays
cheap even when the training stack isn't installed.
"""

from __future__ import annotations

from dlm.data.dataset_builder import build_dataset
from dlm.data.errors import (
    DataError,
    DataFormatError,
    InstructionParseError,
    PreferenceParseError,
    SectionParseError,
    TokenizerBringupError,
)
from dlm.data.formatter import FormattingFunc, make_formatting_func
from dlm.data.instruction_parser import QAPair, parse_instruction_body
from dlm.data.preference_parser import PreferenceTriple, parse_preference_body
from dlm.data.sections_to_rows import sections_to_rows
from dlm.data.splitter import split
from dlm.data.tokenizer_bringup import TokenizerBringup, prepare_tokenizer

__all__ = [
    "DataError",
    "DataFormatError",
    "FormattingFunc",
    "InstructionParseError",
    "PreferenceParseError",
    "PreferenceTriple",
    "QAPair",
    "SectionParseError",
    "TokenizerBringup",
    "TokenizerBringupError",
    "build_dataset",
    "make_formatting_func",
    "parse_instruction_body",
    "parse_preference_body",
    "prepare_tokenizer",
    "sections_to_rows",
    "split",
]
