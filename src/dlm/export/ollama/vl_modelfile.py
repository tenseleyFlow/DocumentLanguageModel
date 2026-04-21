"""VL-aware `Modelfile` generator for vision-language GGUF exports.

Separate from `modelfile.py` (the text-only path) because Ollama's
`{{ .Image }}` directive is VL-specific and the template-registry
rows for chatml/llama3/phi3/mistral don't carry it. This module owns
the VL variant: a dialect-agnostic template that prepends an image
slot before the user's prompt, plus the standard PARAMETER block
from the text path.

Shared directive builders live in `modelfile_shared.py` — header,
stops resolution, param block, system/license lines, num_ctx cap.
This module only owns the VL-specific TEMPLATE shape and sampling
defaults.

Today's vendored llama.cpp tag doesn't fully support PaliGemma or
InternVL2 GGUF export (see `dlm.export.arch_probe`), so this module
produces output that isn't exercised end-to-end — only the render
path is covered.
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
from dlm.export.ollama.template_registry import DialectTemplate

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.export.plan import ExportPlan


@dataclass(frozen=True)
class VlModelfileContext:
    """Render-time inputs for a VL Modelfile.

    Mirrors `ModelfileContext` but carries no dialect — VL bases use
    their own fixed template shape. The image token pulled from
    `spec.vl_preprocessor_plan.image_token` drives nothing at render
    time (Ollama's `{{ .Image }}` is the handoff point), but it
    lives on the spec so the emitter can surface it in comments.
    """

    spec: BaseModelSpec
    plan: ExportPlan
    adapter_dir: Path
    base_gguf_name: str
    adapter_gguf_name: str | None
    dlm_id: str
    adapter_version: int
    system_prompt: str | None = None
    source_dlm_path: Path | None = None
    dlm_version: str = "dev"
    training_sequence_len: int | None = None
    override_temperature: float | None = None
    override_top_p: float | None = None


# Default sampling parameters for VL generation. Chart/doc-QA tends
# to want low temperature for deterministic answers; freeform caption
# generation wants higher. We default conservative (temp=0.2, top_p=0.9)
# and let the user override via the frontmatter `export:` block.
_VL_DEFAULT_TEMPERATURE: float = 0.2
_VL_DEFAULT_TOP_P: float = 0.9

# Ollama 0.4+ accepts `{{ .Image }}` to inject image bytes into the
# prompt. Earlier versions silently drop it — the doctor's
# `ollama --version` check belongs to the VL export DoD but is
# tracked separately; this template assumes 0.4+.
_VL_TEMPLATE_BODY: str = "{{ if .System }}{{ .System }}\n\n{{ end }}{{ .Image }}\n{{ .Prompt }}"

# Minimal stop-token set for VL generation when the adapter's
# tokenizer config doesn't ship explicit stops. PaliGemma's end-
# of-turn marker is `<eos>`; Qwen2-VL uses `<|im_end|>`. The adapter's
# `special_tokens_map.json` is the source of truth at runtime; this
# is the last-resort default.
_VL_FALLBACK_STOPS: tuple[str, ...] = ("<|im_end|>", "<eos>")


def render_vl_modelfile(ctx: VlModelfileContext) -> str:
    """Return the full VL Modelfile text.

    Parallel to `render_modelfile` but emits a VL-specific TEMPLATE
    block and uses VL-appropriate sampling defaults. The adapter
    directory supplies stop tokens + chat template just like the text
    path.
    """
    stops = resolve_stops(ctx.adapter_dir, _vl_template_row())
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
    template_block = f'TEMPLATE """{_VL_TEMPLATE_BODY}"""'
    num_ctx = resolve_num_ctx(ctx.training_sequence_len, ctx.spec.context_length)
    temperature = (
        ctx.override_temperature
        if ctx.override_temperature is not None
        else _VL_DEFAULT_TEMPERATURE
    )
    top_p = ctx.override_top_p if ctx.override_top_p is not None else _VL_DEFAULT_TOP_P
    param_lines = build_param_lines(
        stops=stops,
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        draft_model=None,
    )
    system_line = build_system_line(ctx.system_prompt)
    license_line = build_license_line(ctx.spec)

    parts: list[str] = [header, "", from_line]
    if adapter_line is not None:
        parts.append(adapter_line)
    parts.extend(["", template_block, ""])
    parts.extend(param_lines)
    if system_line is not None:
        parts.extend(["", system_line])
    if license_line is not None:
        parts.extend(["", license_line])
    return "\n".join(parts) + "\n"


def _vl_template_row() -> DialectTemplate:
    """Fallback DialectTemplate used by `resolve_stops` for VL bases.

    The existing stops resolver reads from the adapter's
    `special_tokens_map.json` first, then falls back to the dialect
    row. VL bases don't map to one of the registered text dialects,
    so we pass a synthetic row whose `default_stops` are the VL
    fallback set.
    """
    return DialectTemplate(
        dialect="chatml",  # placeholder — resolve_stops only reads defaults
        template_path=Path("/dev/null"),
        default_stops=_VL_FALLBACK_STOPS,
        default_temperature=_VL_DEFAULT_TEMPERATURE,
        default_top_p=_VL_DEFAULT_TOP_P,
        extra_stop_hints=(),
    )
