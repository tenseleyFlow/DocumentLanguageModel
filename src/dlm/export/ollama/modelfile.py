"""`Modelfile` generator — the thing `ollama create` consumes.

Assembles a complete Modelfile from:

- The `ExportPlan` (quant, merged flag)
- The `BaseModelSpec` (for license + registry metadata in header)
- The adapter directory (source of truth for stops + chat template
  per the tokenizer contract — audit F06)
- The user's system prompt from frontmatter, if any

Output shape is fixed. Stops are always emitted — missing stops
cause runaway generation (findings §9).

Directive builders live in `modelfile_shared.py` so the VL variant
can reuse them without reaching across `_`-prefixed names.

Security: `SYSTEM "..."` content is JSON-escaped via `json.dumps`
to defend against `"` / newline injection in user-supplied prompts.
The Modelfile's `SYSTEM` directive treats its string as a quoted
literal, so a naïve string interpolation would be exploitable.
JSON's string-literal grammar is a subset of Modelfile's, so the
escapes (`\"`, `\n`, `\\`) round-trip cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dlm.export.ollama.modelfile_shared import (
    build_header,
    build_license_line,
    build_param_lines,
    build_system_line,
    resolve_num_ctx,
    resolve_stops,
)
from dlm.export.ollama.template_registry import DialectTemplate, get_template

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.export.plan import ExportPlan


@dataclass(frozen=True)
class ModelfileContext:
    """Everything `render_modelfile` needs to produce one Modelfile."""

    spec: BaseModelSpec
    plan: ExportPlan
    adapter_dir: Path
    base_gguf_name: str  # e.g., "base.Q4_K_M.gguf" or "merged.Q4_K_M.gguf"
    adapter_gguf_name: str | None  # None on --merged path
    dlm_id: str
    adapter_version: int
    system_prompt: str | None = None
    source_dlm_path: Path | None = None
    dlm_version: str = "dev"
    # `sequence_len` from the document's training config. When set, we
    # emit `PARAMETER num_ctx <min(seq_len, spec.context_length)>` so
    # Ollama respects the window the adapter was trained for —
    # otherwise it defaults to 2048 and a document trained at 8192
    # effectively loses 75% of its context.
    training_sequence_len: int | None = None
    # Per-document sampling overrides from frontmatter's `export:`
    # block. When set, they replace the dialect's defaults in the
    # emitted `PARAMETER temperature` / `PARAMETER top_p` lines
    # when building the Modelfile.
    override_temperature: float | None = None
    override_top_p: float | None = None
    # Speculative-decoding draft. When set, emit
    # `PARAMETER draft_model <tag>` so Ollama ≥ 0.5 runs this small
    # model as a speculative drafter. Tag is an Ollama community
    # reference (e.g. `qwen2.5:0.5b`); users `ollama pull` it once.
    draft_model_ollama_name: str | None = None


def render_modelfile(ctx: ModelfileContext) -> str:
    """Return the full Modelfile text.

    Raises `ModelfileError` when the adapter dir is missing metadata
    required for a correct Modelfile (template file, tokenizer config).
    """
    dialect = ctx.spec.template
    template_row = get_template(dialect)

    stops = resolve_stops(ctx.adapter_dir, template_row)
    header = build_header(
        dlm_version=ctx.dlm_version,
        dlm_id=ctx.dlm_id,
        adapter_version=ctx.adapter_version,
        base_key=ctx.spec.key,
        base_revision=ctx.spec.revision,
        quant=ctx.plan.quant,
        merged=ctx.plan.merged,
        source_dlm_path=ctx.source_dlm_path,
    )
    from_line = f"FROM ./{ctx.base_gguf_name}"
    adapter_line = f"ADAPTER ./{ctx.adapter_gguf_name}" if ctx.adapter_gguf_name else None
    template_block = _build_template_block(template_row)
    num_ctx = resolve_num_ctx(ctx.training_sequence_len, ctx.spec.context_length)
    temperature = (
        ctx.override_temperature
        if ctx.override_temperature is not None
        else template_row.default_temperature
    )
    top_p = ctx.override_top_p if ctx.override_top_p is not None else template_row.default_top_p
    param_lines = build_param_lines(
        stops=stops,
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        draft_model=ctx.draft_model_ollama_name,
    )
    system_line = build_system_line(ctx.system_prompt)
    license_line = build_license_line(ctx.spec)

    # Assemble with blank-line separators for readability; the Ollama
    # parser doesn't care about whitespace between directives.
    parts: list[str] = [header, "", from_line]
    if adapter_line is not None:
        parts.append(adapter_line)
    parts.extend(["", template_block, ""])
    parts.extend(param_lines)
    if system_line is not None:
        parts.extend(["", system_line])
    if license_line is not None:
        parts.extend(["", license_line])
    # Trailing newline: every text file.
    return "\n".join(parts) + "\n"


def _build_template_block(template_row: DialectTemplate) -> str:
    """Emit the Modelfile TEMPLATE directive with the Go template body.

    Ollama accepts triple-double-quoted strings for multi-line TEMPLATE
    values. We assume no literal triple-double-quote appears inside the
    Go template (ours don't).
    """
    body = template_row.read_template()
    return f'TEMPLATE """{body}"""'
