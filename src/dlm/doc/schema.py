"""Pydantic models for `.dlm` frontmatter.

Every model is `extra="forbid"` and `frozen=True` — strict validation and
immutable values. Default values must match the shape produced by
`tests/fixtures/dlm_factory.py` so round-trips are stable.

Sprint 12b introduces a versioned dispatcher in `dlm.doc.versioned`;
until then this module ships only the v1 `DlmFrontmatter`.
"""

from __future__ import annotations

import re
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Crockford base32 alphabet used by ULID: 0-9, A-Z minus I L O U.
_ULID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9A-HJ-KM-NP-TV-Z]{26}$")

# Adapter names: lowercase alpha start, alphanumeric + underscore tail.
# Keeps store paths safe (adapter/<name>/versions/) and log lines readable.
_ADAPTER_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

CURRENT_SCHEMA_VERSION: Final[int] = 9
"""Schema version this parser implements.

New fields bump the version and register a migrator in the same
commit — enforced by `test_all_versions_have_migrator_up_to_latest`.
v2 renamed `training.dpo` → `training.preference` to accommodate both
DPO and ORPO under one `method`-switched config. v3 added the
additive `training.cpt` block (DAPT schedule + embedding warm-up)
for continued-pretraining refinements. v4 added the additive
`training.adapters` map for named multi-adapter composition; flat
`adapter`/`lora_*` keys remain the single-adapter shorthand. v5
added the additive `training.precision` override (opt-in fp16/bf16
on MPS after the NaN-adapter bug). v6 adds the additive
`training.sources` block — declarative file-tree directives that
synthesize PROSE sections at train time, letting a `.dlm` act as a
training plan over content stored elsewhere on disk.
"""


class PreferenceHyperparams(BaseModel):
    """Hyperparameters shared across preference methods.

    Some fields are method-specific (`beta` for DPO, `alpha` for
    ORPO); the trainer reads whichever applies. Keeping both on one
    flat block simplifies migration and lets users switch methods
    without reshaping their document.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    beta: float = Field(0.1, ge=0.0, le=1.0)
    alpha: float = Field(0.1, ge=0.0, le=1.0)
    learning_rate: float = Field(5e-6, gt=0.0)
    num_epochs: int = Field(1, ge=1)


class PreferenceConfig(BaseModel):
    """Preference-phase knobs (DPO or ORPO). Additive to `TrainingConfig`;
    default disabled. `enabled` flips to `True` automatically when the
    document contains `::preference::` sections unless the user has
    explicitly set it to `False` — the phase orchestrator reads that
    signal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    method: Literal["dpo", "orpo"] = "dpo"
    hyperparams: PreferenceHyperparams = Field(default_factory=lambda: PreferenceHyperparams())
    # DPO-only fields — ignored for ORPO but kept on the config so a
    # user switching methods doesn't have to delete them.
    loss_type: Literal["sigmoid", "hinge", "ipo"] = "sigmoid"
    reference: Literal["base", "pre_adapter"] = "pre_adapter"


def _default_preference() -> PreferenceConfig:
    return PreferenceConfig()


class CptConfig(BaseModel):
    """Continued-pretraining refinements.

    `schedule="auto"` lets the trainer pick: `dapt` when CPT rows
    dominate (>70% of training rows), otherwise the SFT default. A
    user who wants the DAPT curve regardless of the row mix pins
    `schedule="dapt"`; `schedule="sft"` opts out entirely.

    `embed_warmup_steps>0` unfreezes `embed_tokens` + `lm_head` for
    the first N steps and adds them to `modules_to_save`, which
    inflates adapter size by `vocab_size * hidden_dim`. The trainer
    warns loudly when this is enabled.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schedule: Literal["auto", "dapt", "sft"] = "auto"
    embed_warmup_steps: int = Field(0, ge=0)


def _default_cpt() -> CptConfig:
    return CptConfig()


class GateConfig(BaseModel):
    """Learned MoE-style adapter gate (Sprint 34).

    When `enabled`, a small MLP trained post-SFT routes each prompt to
    a weighted combination of the document's named adapters. Applied
    uniformly across adapter layers (per-layer routing is the research
    follow-up).

    `cold_start_floor` is the minimum number of supervising sections
    per adapter below which gate training is skipped and inference
    defaults to uniform weights — small corpora overfit a tiny router.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    hidden_proj_dim: int = Field(64, ge=8, le=2048)
    steps: int = Field(200, ge=1, le=10000)
    lr: float = Field(3e-4, gt=0.0, le=1.0)
    cold_start_floor: int = Field(4, ge=1, le=1024)
    # Entropy-regularization weight on the gate loss. Higher values
    # discourage mode collapse (one adapter takes all the weight);
    # lower values let the gate commit harder when data justifies it.
    entropy_lambda: float = Field(0.01, ge=0.0, le=1.0)


def _default_gate() -> GateConfig:
    return GateConfig()


class CacheConfig(BaseModel):
    """Tokenized-section cache tuning (Sprint 31.6).

    The cache lives at `~/.dlm/store/<dlm_id>/tokenized-cache/` and
    trades disk for tokenization wall-clock on directive-sourced runs.
    Defaults cover the typical case: cache on, 10 GiB cap, 90-day
    retention. Per-document overrides here let authors tune for their
    corpus size.

    All fields are independent — no cross-field validation. The three
    knobs map to three distinct operator concerns:
    - ``enabled`` is the off-switch.
    - ``max_bytes`` is the disk ceiling.
    - ``prune_older_than_days`` is the retention window.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    # Default: 10 GiB (10 * 1024^3). Per-document cap that supersedes
    # the cache module's built-in default when the trainer opens the
    # cache. Lower for small personal corpora, higher for 50K+ file
    # codebases.
    max_bytes: int = Field(10 * 1024 * 1024 * 1024, ge=1)
    # Default cutoff for `dlm cache prune` when the user doesn't pass
    # `--older-than`. Overridable by the CLI flag on a per-command
    # basis.
    prune_older_than_days: int = Field(90, ge=1)


def _default_cache() -> CacheConfig:
    return CacheConfig()


class AdapterConfig(BaseModel):
    """One named adapter in a multi-adapter document.

    A subset of the flat config — only the per-adapter LoRA knobs plus
    `learning_rate`. Hyperparameters that are intrinsically run-scoped
    (`sequence_len`, `num_epochs`, `seed`, `optimizer`, `lr_scheduler`,
    `warmup_ratio`, `micro_batch_size`, `grad_accum`) stay at the
    `TrainingConfig` top level because mixing them per-adapter makes
    schedules and batching incoherent.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter: Literal["lora", "qlora"] = "lora"
    lora_r: int = Field(8, ge=1, le=256)
    lora_alpha: int = Field(16, ge=1)
    lora_dropout: float = Field(0.05, ge=0.0, le=0.5)
    target_modules: Literal["auto"] | list[str] = "auto"
    learning_rate: float = Field(2e-4, gt=0.0)


class SourceDirective(BaseModel):
    """A directive to ingest file(s) as synthetic PROSE sections at train
    time.

    Paths resolve relative to the `.dlm` file's parent directory when
    not absolute; `~` expands via `Path.expanduser()`. Under
    `training.sources_policy="strict"` the resolved path must stay
    under the `.dlm` parent dir (symlinks included — containment is
    checked after `Path.resolve()`). `permissive` lets absolute paths
    anywhere on disk.

    `include` / `exclude` are POSIX-glob patterns relative to each
    source root (default `("**/*",)` + `()` — every file matches).
    Size caps apply per-file and per-directive; binary files (first-
    KiB NUL scan) and non-UTF-8 bytes are skipped with a log warning,
    never a fatal error, because mixed trees are the common case.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(..., min_length=1)
    include: tuple[str, ...] = ("**/*",)
    exclude: tuple[str, ...] = ()
    max_bytes_per_file: int | None = Field(default=None, ge=1)
    max_files: int | None = Field(default=None, ge=1)


class TrainingConfig(BaseModel):
    """Training-time knobs. `auto` values are resolved by the hardware doctor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter: Literal["lora", "qlora"] = "lora"
    lora_r: int = Field(8, ge=1, le=256)
    lora_alpha: int = Field(16, ge=1)
    lora_dropout: float = Field(0.05, ge=0.0, le=0.5)
    target_modules: Literal["auto"] | list[str] = "auto"
    sequence_len: int = Field(2048, ge=64, le=32768)
    micro_batch_size: Literal["auto"] | int = "auto"
    grad_accum: Literal["auto"] | int = "auto"
    learning_rate: float = Field(2e-4, gt=0.0)
    num_epochs: int = Field(3, ge=1)
    optimizer: Literal["adamw_torch", "adamw_bnb_8bit", "paged_adamw_8bit"] = "adamw_torch"
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float = Field(0.1, ge=0.0, le=0.5)
    # Advanced: override the hardware doctor's auto-picked precision.
    # `None` (default) lets the planner pick per backend — bf16 on
    # Ampere+, fp16 on older CUDA, fp32 on MPS (the last pin is
    # defensive: MPS fp16 attention kernels produce NaN LoRA weights
    # on tiny-data runs; see `.docs/bugs/01-nan-adapter-on-mps.md`).
    # Users who want fp16 on MPS for memory (e.g. running an 8B base
    # on a 24 GB unified-memory budget) can opt in here, accepting
    # the stability risk on small datasets.
    precision: Literal["bf16", "fp16", "fp32"] | None = None
    seed: int = 42
    preference: PreferenceConfig = Field(default_factory=_default_preference)
    cpt: CptConfig = Field(default_factory=_default_cpt)
    # Learned adapter gate (Sprint 34). Only meaningful when `adapters`
    # declares two or more named adapters — a gate over a single
    # adapter is a tautology. Enforced at validate-time below.
    gate: GateConfig = Field(default_factory=_default_gate)
    # Tokenized-section cache tuning (Sprint 31.6). Defaults preserve
    # pre-v9 behavior: cache on, 10 GiB cap, 90-day prune window.
    cache: CacheConfig = Field(default_factory=_default_cache)
    # Named adapters for multi-adapter composition. When set, the flat
    # `adapter`/`lora_*`/`target_modules`/`learning_rate` fields must
    # stay at their defaults — mixing the two shapes creates ambiguous
    # "which config wins?" semantics. An empty/None `adapters` keeps
    # the single-adapter shorthand fully backward-compatible.
    adapters: dict[str, AdapterConfig] | None = None
    # Source directives (v6). Declarative file-tree ingestion — each
    # entry becomes a walk-and-read at train time, synthesizing PROSE
    # sections for the CPT path. `None` / empty keeps the `.dlm` as a
    # self-contained training corpus; populated lets the document
    # reference external codebases, notes directories, etc. See
    # `dlm.directives.expand_sources`.
    sources: tuple[SourceDirective, ...] | None = None
    # `permissive` (default) lets directive paths point anywhere on
    # disk. `strict` confines them to the `.dlm` parent subtree —
    # useful when a `.dlm` travels with a project and the author wants
    # training content to stay project-local regardless of where a
    # downstream user places the file.
    sources_policy: Literal["permissive", "strict"] = "permissive"

    @field_validator("micro_batch_size", "grad_accum")
    @classmethod
    def _validate_auto_or_positive(cls, v: int | str) -> int | str:
        if v == "auto":
            return v
        if not isinstance(v, int) or v < 1:
            raise ValueError(f"must be a positive int or 'auto', got {v!r}")
        return v

    @field_validator("adapters")
    @classmethod
    def _validate_adapter_names(
        cls, v: dict[str, AdapterConfig] | None
    ) -> dict[str, AdapterConfig] | None:
        if v is None:
            return v
        if not v:
            raise ValueError(
                "training.adapters: at least one adapter must be declared "
                "(omit the block entirely for the single-adapter shorthand)"
            )
        for name in v:
            if not _ADAPTER_NAME_RE.fullmatch(name):
                raise ValueError(
                    f"training.adapters: {name!r} is not a valid adapter "
                    f"name (must match {_ADAPTER_NAME_RE.pattern})"
                )
        return v

    @model_validator(mode="after")
    def _gate_requires_multiple_adapters(self) -> TrainingConfig:
        if self.gate.enabled and (self.adapters is None or len(self.adapters) < 2):
            raise ValueError(
                "training.gate.enabled=true requires training.adapters "
                "with two or more named adapters (a gate over a single "
                "adapter has nothing to route between)"
            )
        return self

    @model_validator(mode="after")
    def _flat_and_named_are_mutually_exclusive(self) -> TrainingConfig:
        if self.adapters is None:
            return self
        # A set flat-adapter field would silently lose to the named
        # block at train time. Refuse at parse time instead.
        flat_defaults = {
            "adapter": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": "auto",
            "learning_rate": 2e-4,
        }
        drift = [key for key, default in flat_defaults.items() if getattr(self, key) != default]
        if drift:
            raise ValueError(
                "training.adapters is declared; flat per-adapter fields "
                f"{drift} must stay at their defaults (move them into the "
                "per-adapter block instead)"
            )
        return self


class ExportConfig(BaseModel):
    """Export-time defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_quant: Literal["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"] = "Q4_K_M"
    # Optional per-document sampling overrides. When set, the Modelfile
    # emits `PARAMETER temperature <v>` / `PARAMETER top_p <v>` in place
    # of the dialect default — a Q&A document prefers temperature=0.2,
    # a creative one prefers 0.9. `None` keeps the dialect default
    # (audit-04 Q5).
    default_temperature: float | None = Field(None, gt=0.0, le=2.0)
    default_top_p: float | None = Field(None, gt=0.0, le=1.0)


# Named factories so mypy can type-check the field defaults correctly.
def _default_training() -> TrainingConfig:
    return TrainingConfig()


def _default_export() -> ExportConfig:
    return ExportConfig()


class DlmFrontmatter(BaseModel):
    """Top-level frontmatter: the YAML block between `---` delimiters.

    `dlm_id` is a canonical 26-character ULID. It is assigned by
    `dlm init` (Sprint 13) and never regenerated by the parser.
    `base_model` is either a registry key (e.g. `qwen2.5-1.5b`) or an
    `hf:org/name` escape hatch — the registry (Sprint 06) validates the
    actual lookup; this module only validates that the string is non-empty.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dlm_id: str
    dlm_version: int = CURRENT_SCHEMA_VERSION
    base_model: str = Field(..., min_length=1)
    training: TrainingConfig = Field(default_factory=_default_training)
    export: ExportConfig = Field(default_factory=_default_export)
    system_prompt: str | None = None

    @field_validator("dlm_id")
    @classmethod
    def _validate_ulid(cls, v: str) -> str:
        if not _ULID_RE.fullmatch(v):
            raise ValueError(
                f"dlm_id must be a 26-char Crockford base32 ULID, got {v!r}",
            )
        return v

    @field_validator("dlm_version")
    @classmethod
    def _validate_version(cls, v: int) -> int:
        # Defense in depth (audit-07 M6): the `versioned` dispatcher is
        # the intended entry point, but direct `DlmFrontmatter.model_validate`
        # callers (tests, tooling) need the same guard. Reject both
        # under-1 and beyond-current values at the field level.
        if v < 1:
            raise ValueError(f"dlm_version must be ≥1, got {v}")
        if v > CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"dlm_version {v} is newer than this CLI supports "
                f"(CURRENT_SCHEMA_VERSION={CURRENT_SCHEMA_VERSION})."
            )
        return v
