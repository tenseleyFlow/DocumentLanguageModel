"""Pre-flight compatibility checks — run BEFORE any subprocess launches.

Four probes, each returning `None` on success and raising
`PreflightError` on failure with actionable remediation:

1. **Adapter config match.** `adapter_config.json` in the checkpoint
   references the same base HF id as the spec. Catches the "exported
   adapter from a different base" footgun when users hand-manage
   multiple adapters.
2. **Tokenizer vocab match.** The adapter's tokenizer (source of
   truth per Sprint 12b) has the same vocab size as the model that
   was trained. Catches Sprint 07's pad-fallback resize case: if the
   tokenizer grew to 32001 but the adapter only has 32000 rows, the
   extra `<|pad|>` embedding is undefined.
3. **Chat template presence.** Unless `--no-template`, the tokenizer
   config carries a non-empty `chat_template` — Sprint 12's Modelfile
   emitter needs it.
4. **Arch / pre-tokenizer probes** (delegated to Sprint 06's probe
   suite). These are a courtesy re-check at export time, since
   registry state can drift between `dlm init` and `dlm export`.

Pure-python and mock-friendly — no subprocess calls in this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from dlm.export.errors import PreflightError

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec


def check_adapter_config(adapter_dir: Path, spec: BaseModelSpec) -> None:
    """Assert `adapter_config.json` references `spec.hf_id`.

    PEFT writes the base model id into `adapter_config.json`. A
    mismatch usually means the user trained against one base but is
    trying to export against another — a silent-data-corruption path
    we refuse upfront.
    """
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise PreflightError(
            probe="adapter_config",
            detail=(
                f"missing {config_path}. This directory does not look like a "
                "PEFT adapter checkpoint; has `dlm train` run successfully?"
            ),
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(
            probe="adapter_config",
            detail=f"cannot parse {config_path}: {exc}",
        ) from exc

    saved = config.get("base_model_name_or_path")
    if saved != spec.hf_id:
        raise PreflightError(
            probe="adapter_config",
            detail=(
                f"adapter was trained against {saved!r}, but current spec is "
                f"{spec.hf_id!r}. Re-train fresh against the correct base, or "
                "switch back to the original base in the .dlm frontmatter."
            ),
        )


def check_tokenizer_vocab(adapter_dir: Path) -> int:
    """Read the vocab size from the adapter's tokenizer config.

    Returns the vocab size for downstream consumers (`assert_gguf_vocab_matches`
    after base conversion). Raises `PreflightError` if the adapter dir
    lacks a `tokenizer_config.json` or the file is malformed — the
    export pipeline needs vocab info to validate the emitted GGUF.
    """
    cfg_path = adapter_dir / "tokenizer_config.json"
    if not cfg_path.exists():
        raise PreflightError(
            probe="tokenizer_vocab",
            detail=(
                f"adapter dir {adapter_dir} is missing tokenizer_config.json. "
                "Sprint 07 bringup writes this at training end; a checkpoint "
                "predating Sprint 07 can't be exported — re-train."
            ),
        )
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(
            probe="tokenizer_vocab",
            detail=f"cannot parse {cfg_path}: {exc}",
        ) from exc

    # `vocab_size` key isn't always present in tokenizer_config.json;
    # fall back to the companion tokenizer.json which always carries it.
    vocab_size = cfg.get("vocab_size")
    if not isinstance(vocab_size, int):
        tokenizer_json = adapter_dir / "tokenizer.json"
        if tokenizer_json.exists():
            try:
                t = json.loads(tokenizer_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise PreflightError(
                    probe="tokenizer_vocab",
                    detail=f"cannot parse {tokenizer_json}: {exc}",
                ) from exc
            model = t.get("model") or {}
            vocab = model.get("vocab")
            if isinstance(vocab, dict):
                vocab_size = len(vocab)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise PreflightError(
            probe="tokenizer_vocab",
            detail=(
                f"cannot determine vocab size from {adapter_dir}. "
                "Inspect tokenizer_config.json + tokenizer.json; export "
                "needs a trustworthy vocab count to validate GGUF output."
            ),
        )
    return vocab_size


def check_chat_template(adapter_dir: Path, *, required: bool = True) -> None:
    """Assert the tokenizer config has a non-empty `chat_template`.

    `--no-template` on the CLI sets `required=False` (Sprint 11 spec
    line 165); the default requires one because Sprint 12's Modelfile
    emitter hardcodes `TEMPLATE "..."` which needs source text.
    """
    if not required:
        return
    cfg_path = adapter_dir / "tokenizer_config.json"
    if not cfg_path.exists():
        raise PreflightError(
            probe="chat_template",
            detail=f"missing {cfg_path}; cannot check chat_template presence.",
        )
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(
            probe="chat_template",
            detail=f"cannot parse {cfg_path}: {exc}",
        ) from exc
    template = cfg.get("chat_template")
    if not template or not str(template).strip():
        raise PreflightError(
            probe="chat_template",
            detail=(
                "tokenizer has no chat_template. Pass --no-template to skip "
                "this check (Modelfile emission will fall back to the base "
                "model's default), or attach a template via frontmatter."
            ),
        )


def check_was_adapter_qlora(adapter_dir: Path) -> bool:
    """Read the audit-05 M1 `training_run.json` flag.

    Returns False when the file is missing (legacy adapter trained
    before audit-05 landed the explicit flag). A malformed file is
    NOT silently treated as "not QLoRA" — that would let a corrupt
    adapter bypass the pitfall-3 merge gate; we raise instead.
    """
    training_run_path = adapter_dir / "training_run.json"
    if not training_run_path.exists():
        return False
    try:
        data = json.loads(training_run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(
            probe="training_run_json",
            detail=(
                f"adapter {adapter_dir}/training_run.json is unreadable "
                f"({exc}); cannot determine use_qlora, refusing merge-safety "
                "bypass. Re-run `dlm train` or fix the file."
            ),
        ) from exc
    return bool(data.get("use_qlora", False))
