"""Typed errors for the dataset-assembly pipeline.

Every error carries enough context to point the user back at a specific
section (and line offset within the section) of the source `.dlm`. The
section grammar errors (`InstructionParseError`, `PreferenceParseError`)
include the section id + the line inside the section's body where the
problem was detected; callers compose the message with the section's
`start_line` from `doc.sections.Section` to recover a file:line location.
"""

from __future__ import annotations


class DataError(Exception):
    """Base for all `dlm.data` errors."""


class DataFormatError(DataError):
    """Row does not have a recognized shape for SFT / CPT / DPO routing."""


class TokenizerBringupError(DataError):
    """Tokenizer load / fixup failed (missing chat_template, pad == EOS, etc)."""


class SectionParseError(DataError):
    """Base for section-body grammar errors.

    `section_id` is the 16-char content-hash of the source section;
    `section_line` is 1-indexed and measured from the first line *after*
    the opening fence so users can skim to the offending line inside
    their editor.
    """

    def __init__(self, message: str, *, section_id: str, section_line: int) -> None:
        super().__init__(message)
        self.section_id = section_id
        self.section_line = section_line


class InstructionParseError(SectionParseError):
    """`### Q` / `### A` grammar violation inside an `::instruction::` fence."""


class PreferenceParseError(SectionParseError):
    """`### Prompt` / `### Chosen` / `### Rejected` grammar violation."""
