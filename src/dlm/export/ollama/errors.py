"""Typed errors for the Ollama integration path."""

from __future__ import annotations


class OllamaError(Exception):
    """Base for `dlm.export.ollama` errors."""


class OllamaBinaryNotFoundError(OllamaError):
    """`ollama` not found on PATH or standard install locations.

    Remediation: install from https://ollama.com/download, then re-run.
    """


class OllamaVersionError(OllamaError):
    """Installed Ollama is older than `OLLAMA_MIN_VERSION` (audit F16).

    Carries the detected and required versions so the CLI can render a
    specific upgrade message.
    """

    def __init__(
        self,
        *,
        detected: tuple[int, int, int],
        required: tuple[int, int, int],
    ) -> None:
        def _fmt(v: tuple[int, int, int]) -> str:
            return f"{v[0]}.{v[1]}.{v[2]}"

        super().__init__(
            f"Ollama {_fmt(detected)} is below the minimum supported version "
            f"{_fmt(required)}. Upgrade from https://ollama.com/download."
        )
        self.detected = detected
        self.required = required


class OllamaCreateError(OllamaError):
    """`ollama create` exited non-zero.

    Captures the subprocess stdout + stderr so the CLI can surface the
    real remediation (often "base GGUF missing" or "duplicate name").
    """

    def __init__(self, *, stdout: str, stderr: str) -> None:
        tail = stderr.strip() or stdout.strip() or "(no output)"
        super().__init__(f"`ollama create` failed:\n{tail}")
        self.stdout = stdout
        self.stderr = stderr


class OllamaSmokeError(OllamaError):
    """`ollama run` produced no coherent output or exited non-zero.

    Smoke failures are a hard stop for the default `dlm export` flow;
    users who know the model works but want to skip smoke can pass
    `--no-smoke`.
    """

    def __init__(self, *, stdout: str, stderr: str) -> None:
        super().__init__(
            f"smoke test failed — `ollama run` returned empty or errored:\n{stderr.strip() or stdout.strip() or '(no output)'}"
        )
        self.stdout = stdout
        self.stderr = stderr


class ModelfileError(OllamaError):
    """Modelfile generation or validation failed.

    Typically means the adapter dir is missing tokenizer metadata the
    Modelfile needs (stops, chat template). Sprint 07's bringup should
    have written these; surfacing this at export is the fail-fast gate.
    """


class TemplateRegistryError(OllamaError):
    """Requested template dialect not in the registry.

    Registry ships one entry per `BaseModelSpec.template` Literal value
    (`chatml`, `llama3`, `phi3`, `mistral`). Unknown dialect usually
    means an hf:-escape-hatch base whose template inference picked a
    dialect we haven't templated — remedy is to add it to the registry.
    """
