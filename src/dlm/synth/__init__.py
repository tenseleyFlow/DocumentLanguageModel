"""Sprint 43 synthetic-instruction generation substrate."""

from dlm.synth.errors import (
    InvalidTeacherSpecError,
    SynthError,
    TeacherInvocationError,
    TeacherUnavailableError,
)
from dlm.synth.prompts import (
    DEFAULT_PROMPT_TEMPLATES,
    PromptParserKind,
    SynthPromptTemplate,
    SynthStrategy,
    get_prompt_template,
)

__all__ = [
    "DEFAULT_PROMPT_TEMPLATES",
    "InvalidTeacherSpecError",
    "PromptParserKind",
    "SynthError",
    "SynthPromptTemplate",
    "SynthStrategy",
    "TeacherInvocationError",
    "TeacherUnavailableError",
    "get_prompt_template",
]
