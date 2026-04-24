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
from dlm.synth.teachers import (
    AnthropicTeacher,
    HfTeacher,
    OpenAiTeacher,
    SelfTeacher,
    SynthTeacher,
    TeacherKind,
    TeacherRef,
    VllmServerTeacher,
    build_teacher,
    parse_teacher_ref,
)

__all__ = [
    "AnthropicTeacher",
    "DEFAULT_PROMPT_TEMPLATES",
    "HfTeacher",
    "InvalidTeacherSpecError",
    "OpenAiTeacher",
    "PromptParserKind",
    "SelfTeacher",
    "SynthError",
    "SynthPromptTemplate",
    "SynthTeacher",
    "SynthStrategy",
    "TeacherKind",
    "TeacherInvocationError",
    "TeacherRef",
    "TeacherUnavailableError",
    "VllmServerTeacher",
    "build_teacher",
    "get_prompt_template",
    "parse_teacher_ref",
]
