"""`BaseModelSpec` — curated metadata for a single pretrained base model.

Every field is strict: `extra="forbid"`, frozen, and validated on
instantiation. Values pack everything the rest of the project needs to
know about a base without re-fetching HF metadata at every decision
point:

- `revision`: 40-char commit SHA. Enforced non-None so retrains under the
  same spec pin at exactly the same weights.
- `target_modules`: per-architecture LoRA target list (see findings §8;
  `"all-linear"` is avoided because it bloats small models).
- `template`: the chat-template dialect used by Sprint 12's Go-template
  registry for Modelfile generation.
- `gguf_arch` / `tokenizer_pre`: identifiers the llama.cpp converter
  matches against; Sprint 11's export preflight uses them.
- License / gating (audit-02 F04 + F21): separate fields for SPDX,
  acceptance gating, and re-distribution — each consumed by a different
  gate (Sprint 12b license UX; Sprint 14 pack `--include-base`;
  Sprint 28 share-protocol push refusal).
"""

from __future__ import annotations

import re
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SHA_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")


class BaseModelSpec(BaseModel):
    """Curated registry metadata for one base model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(..., min_length=1, description="Registry slug (e.g. `qwen2.5-1.5b`).")
    hf_id: str = Field(
        ..., min_length=1, description="HuggingFace id, e.g. `Qwen/Qwen2.5-1.5B-Instruct`."
    )
    revision: str = Field(..., description="40-char commit SHA; never a branch.")
    architecture: str = Field(..., description="transformers `config.architectures[0]` value.")
    params: int = Field(..., gt=0, description="Parameter count; drives hardware doctor.")
    target_modules: list[str] = Field(..., min_length=1)
    template: Literal["chatml", "llama3", "phi3", "mistral"]
    gguf_arch: str = Field(..., min_length=1, description="Name llama.cpp's converter uses.")
    tokenizer_pre: str = Field(..., min_length=1, description="Pre-tokenizer label.")

    # License + acceptance (audit-02 F04 / F21).
    license_spdx: str = Field(..., min_length=1)
    license_url: str | None = None
    requires_acceptance: bool = False
    redistributable: bool = Field(
        ...,
        description="True iff the license allows bundling the base inside a .dlm.pack.",
    )

    # Size + context hints.
    size_gb_fp16: float = Field(..., gt=0)
    context_length: int = Field(..., gt=0)
    recommended_seq_len: int = Field(..., gt=0)

    @field_validator("revision")
    @classmethod
    def _validate_revision(cls, value: str) -> str:
        if not _SHA_RE.fullmatch(value):
            raise ValueError(f"revision must be a 40-char lowercase hex SHA, got {value!r}")
        return value

    @field_validator("hf_id")
    @classmethod
    def _validate_hf_id(cls, value: str) -> str:
        if "/" not in value or value.startswith("/") or value.endswith("/"):
            raise ValueError(f"hf_id must be 'org/name', got {value!r}")
        org, _, name = value.partition("/")
        if not org or not name or "/" in name:
            raise ValueError(f"hf_id must be 'org/name' (single `/`), got {value!r}")
        return value

    @field_validator("target_modules")
    @classmethod
    def _validate_target_modules(cls, value: list[str]) -> list[str]:
        if any(not m for m in value):
            raise ValueError("target_modules must not contain empty strings")
        return value
