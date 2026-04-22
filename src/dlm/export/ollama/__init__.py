"""Ollama integration — Modelfile emission + `ollama create`/`run`.

The `binary.py`, `register.py`, and
`smoke.py` modules call `subprocess`; `modelfile.py` and
`template_registry.py` are pure-python and fully unit-testable.
"""

from __future__ import annotations

from dlm.export.ollama.binary import (
    OLLAMA_MIN_VERSION,
    check_ollama_version,
    locate_ollama,
    ollama_version,
)
from dlm.export.ollama.errors import (
    ModelfileError,
    OllamaBinaryNotFoundError,
    OllamaCreateError,
    OllamaError,
    OllamaSmokeError,
    OllamaVersionError,
    TemplateRegistryError,
    VerificationError,
)
from dlm.export.ollama.modelfile import ModelfileContext, render_modelfile
from dlm.export.ollama.register import ollama_create, ollama_lock_path
from dlm.export.ollama.smoke import first_line, ollama_run
from dlm.export.ollama.template_registry import (
    DialectTemplate,
    get_template,
    load_template_text,
    registered_dialects,
)
from dlm.export.ollama.verify import (
    parse_prompt_eval_count,
    run_with_telemetry,
    verify_token_count,
)

__all__ = [
    "DialectTemplate",
    "ModelfileContext",
    "ModelfileError",
    "OLLAMA_MIN_VERSION",
    "OllamaBinaryNotFoundError",
    "OllamaCreateError",
    "OllamaError",
    "OllamaSmokeError",
    "OllamaVersionError",
    "TemplateRegistryError",
    "VerificationError",
    "check_ollama_version",
    "first_line",
    "get_template",
    "load_template_text",
    "locate_ollama",
    "ollama_create",
    "ollama_lock_path",
    "ollama_run",
    "ollama_version",
    "parse_prompt_eval_count",
    "registered_dialects",
    "render_modelfile",
    "run_with_telemetry",
    "verify_token_count",
]
