"""`BaseModelSpec` — curated metadata for a single pretrained base model.

Every field is strict: `extra="forbid"`, frozen, and validated on
instantiation. Values pack everything the rest of the project needs to
know about a base without re-fetching HF metadata at every decision
point:

- `revision`: 40-char commit SHA. Enforced non-None so retrains under the
  same spec pin at exactly the same weights.
- `target_modules`: per-architecture LoRA target list (see findings §8;
  `"all-linear"` is avoided because it bloats small models).
- `template`: the chat-template dialect used by the Go-template
  registry for Modelfile generation.
- `gguf_arch` / `tokenizer_pre`: identifiers the llama.cpp converter
  matches against; export preflight uses them.
- `reasoning_tuned` / `context_length_effective`: additive registry
  hints for prompt defaults and realistic doctor estimates. The
  effective length defaults to the nominal context window when unset.
- `refresh_check_hf_gating` / `provenance_url` /
  `provenance_match_text`: live-registry refresh hints for entries
  whose fetch mirror and first-party provenance page are not the same
  system.
- License / gating: separate fields for SPDX, acceptance gating, and
  re-distribution — each consumed by a different policy gate (license
  acceptance, pack `--include-base`, share-protocol refusal).
"""

from __future__ import annotations

import re
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_PROMPT_TEMPERATURE: Final[float] = 0.7
DEFAULT_REASONING_PROMPT_TEMPERATURE: Final[float] = 0.6


class VlPreprocessorPlan(BaseModel):
    """Per-base vision-preprocessing parameters.

    Pinned at registry-build time so `dlm export` + the VL cache key
    stay stable across reruns. HF's `AutoProcessor` is the source of
    truth at runtime; this block records the *expected* shape for
    preflight checks + cache keying.

    `target_size` is `(height, width)` in pixels. `resize_policy`
    defaults to `"fixed"` because that's what the current launch
    registry ships. `image_token` is the textual placeholder inserted
    into prompts before the processor expands it into
    `num_image_tokens` copies.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_size: tuple[int, int] = Field(..., description="(height, width) in pixels")
    resize_policy: Literal["fixed", "dynamic"] = "fixed"
    image_token: str = Field(..., min_length=1, description="Placeholder token string")
    num_image_tokens: int = Field(..., gt=0, description="Tokens consumed per image")

    @field_validator("target_size")
    @classmethod
    def _validate_target_size(cls, value: tuple[int, int]) -> tuple[int, int]:
        h, w = value
        if h <= 0 or w <= 0:
            raise ValueError(f"target_size must be positive, got {value!r}")
        return value


class AudioPreprocessorPlan(BaseModel):
    """Per-base audio-preprocessing parameters.

    Mirrors `VlPreprocessorPlan` — pinned at registry-build time so
    the audio cache key stays stable. Current releases refuse audio at
    non-target `sample_rate`; resampling lands as a follow-up.

    `sample_rate` is the model's training rate in Hz (Qwen2-Audio:
    16000). `max_length_seconds` caps the per-clip duration the
    processor sees; longer clips are truncated (the processor's
    built-in policy). `audio_token` is the textual placeholder that
    expands into the model's fixed audio-token window.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_rate: int = Field(..., gt=0, description="Hz — refuse on mismatch")
    max_length_seconds: float = Field(..., gt=0.0)
    audio_token: str = Field(..., min_length=1, description="Placeholder token string")
    num_audio_tokens: int = Field(..., gt=0, description="Tokens reserved per clip")


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
    template: Literal[
        "chatml",
        "qwen3thinking",
        "gemma2",
        "smollm3",
        "olmo2",
        "llama3",
        "phi3",
        "phi4mini",
        "mistral",
        "paligemma",
        "qwen2-audio",
        "qwen2-vl",
        "internvl2",
    ]
    gguf_arch: str = Field(..., min_length=1, description="Name llama.cpp's converter uses.")
    tokenizer_pre: str = Field(..., min_length=1, description="Pre-tokenizer label.")

    # License + acceptance.
    license_spdx: str = Field(..., min_length=1)
    license_url: str | None = None
    requires_acceptance: bool = False
    redistributable: bool = Field(
        ...,
        description="True iff the license allows bundling the base inside a .dlm.pack.",
    )
    # trust_remote_code: `True` for bases whose HF class lives in the
    # model's own repo (custom `modeling_*.py` files) rather than in
    # transformers. Picking such a base as `base_model:` in a .dlm is
    # the user's informed acknowledgment — the registry entry carries
    # a docstring caveat, vl-memory.md + the cookbook flag it, and the
    # loader only passes `trust_remote_code=True` to HF when this is
    # `True` on the spec. Defaults to False so non-custom bases can
    # never accidentally opt into remote code.
    trust_remote_code: bool = False

    # Size + context hints.
    size_gb_fp16: float = Field(..., gt=0)
    context_length: int = Field(..., gt=0)
    context_length_effective: int | None = Field(None, gt=0)
    recommended_seq_len: int = Field(..., gt=0)
    reasoning_tuned: bool = False
    refresh_check_hf_gating: bool = True
    provenance_url: str | None = None
    provenance_match_text: str | None = None

    # Modality + multi-modal preprocessing (schema v10 + v11, plus the
    # additive `text-moe` discriminator).
    # Text-family bases leave `modality in {"text", "text-moe"}`
    # with both plans None;
    # `modality="vision-language"` requires a `vl_preprocessor_plan`
    # and rejects an audio plan; `modality="audio-language"` requires
    # an `audio_preprocessor_plan` and rejects a vl plan. Every
    # invariant is enforced below at validate time.
    modality: Literal["text", "text-moe", "vision-language", "audio-language"] = "text"
    vl_preprocessor_plan: VlPreprocessorPlan | None = None
    audio_preprocessor_plan: AudioPreprocessorPlan | None = None

    @model_validator(mode="after")
    def _modality_matches_plan(self) -> BaseModelSpec:
        if self.modality == "vision-language":
            if self.vl_preprocessor_plan is None:
                raise ValueError(
                    f"base {self.key!r}: modality='vision-language' requires "
                    "a vl_preprocessor_plan (pinned image size + token shape)"
                )
            if self.audio_preprocessor_plan is not None:
                raise ValueError(
                    f"base {self.key!r}: audio_preprocessor_plan is invalid "
                    "on a vision-language base"
                )
        elif self.modality == "audio-language":
            if self.audio_preprocessor_plan is None:
                raise ValueError(
                    f"base {self.key!r}: modality='audio-language' requires "
                    "an audio_preprocessor_plan (pinned sample_rate + token shape)"
                )
            if self.vl_preprocessor_plan is not None:
                raise ValueError(
                    f"base {self.key!r}: vl_preprocessor_plan is invalid on an audio-language base"
                )
        else:  # "text" or "text-moe"
            if self.vl_preprocessor_plan is not None:
                raise ValueError(
                    f"base {self.key!r}: vl_preprocessor_plan only valid with "
                    "modality='vision-language'"
                )
            if self.audio_preprocessor_plan is not None:
                raise ValueError(
                    f"base {self.key!r}: audio_preprocessor_plan only valid "
                    "with modality='audio-language'"
                )
        return self

    @model_validator(mode="after")
    def _effective_context_length_is_bounded(self) -> BaseModelSpec:
        if (
            self.context_length_effective is not None
            and self.context_length_effective > self.context_length
        ):
            raise ValueError(
                f"base {self.key!r}: context_length_effective={self.context_length_effective} "
                f"cannot exceed context_length={self.context_length}"
            )
        return self

    @model_validator(mode="after")
    def _provenance_probe_is_complete(self) -> BaseModelSpec:
        url_set = self.provenance_url is not None
        text_set = self.provenance_match_text is not None
        if url_set != text_set:
            raise ValueError(
                f"base {self.key!r}: provenance_url and provenance_match_text must be set together"
            )
        if not self.refresh_check_hf_gating and not url_set:
            raise ValueError(
                f"base {self.key!r}: refresh_check_hf_gating=False requires a "
                "first-party provenance_url + provenance_match_text"
            )
        return self

    @property
    def suggested_prompt_temperature(self) -> float:
        """Default sampling temperature for `dlm prompt`.

        Most instruct bases keep the long-standing 0.7 default.
        Reasoning-tuned bases run slightly cooler by default so the
        chain-of-thought control tokens they were tuned around stay
        stable when the user omits `--temp`.
        """
        if self.reasoning_tuned:
            return DEFAULT_REASONING_PROMPT_TEMPERATURE
        return DEFAULT_PROMPT_TEMPERATURE

    @property
    def effective_context_length(self) -> int:
        """Context window `dlm doctor` should estimate against.

        Registry rows can pin a lower practical ceiling than the
        model's advertised nominal context length. When no such hint is
        set, the nominal context window remains the source of truth.
        """
        return self.context_length_effective or self.context_length

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
