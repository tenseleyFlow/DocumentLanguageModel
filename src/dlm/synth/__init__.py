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
from dlm.synth.run import (
    PlannedSynthInstruction,
    SkippedSynthSection,
    SynthPair,
    SynthRunPlan,
    SynthSkipReason,
    SynthSourceSection,
    build_synth_plan,
    render_synth_plan,
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
    "PlannedSynthInstruction",
    "SelfTeacher",
    "SkippedSynthSection",
    "SynthError",
    "SynthPair",
    "SynthPromptTemplate",
    "SynthRunPlan",
    "SynthSkipReason",
    "SynthSourceSection",
    "SynthTeacher",
    "SynthStrategy",
    "TeacherKind",
    "TeacherInvocationError",
    "TeacherRef",
    "TeacherUnavailableError",
    "VllmServerTeacher",
    "build_synth_plan",
    "build_teacher",
    "get_prompt_template",
    "parse_teacher_ref",
    "render_synth_plan",
]
