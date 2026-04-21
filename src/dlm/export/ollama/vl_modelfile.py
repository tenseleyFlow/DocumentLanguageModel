"""VL-aware `Modelfile` generator for vision-language GGUF exports.

Separate from `modelfile.py` (the text-only path) because Ollama's
`{{ .Image }}` directive is VL-specific and the template-registry
rows for chatml/llama3/phi3/mistral don't carry it. This module owns
the VL variant: a dialect-agnostic template that prepends an image
slot before the user's prompt, plus the standard PARAMETER block
from the text path.

Sprint 35.4 scope: scaffold + unit-test the renderer so the day VL
GGUF conversion lands in llama.cpp, the emitter is ready. The
current vendored tag doesn't fully support PaliGemma or InternVL2
GGUF export (see `dlm.export.arch_probe`), so this module produces
output that isn't exercised end-to-end today — only the render path
is covered.

Reuses the following from `modelfile.py`:
- `_build_header`, `_build_param_lines`, `_build_system_line`,
  `_build_license_line` — the non-template directives are identical
  across text + VL Modelfiles.
- `_resolve_stops`, `_resolve_num_ctx` — stops + context come from
  the same adapter-tokenizer sources regardless of modality.

What's different:
- No chat-dialect template row lookup; the TEMPLATE block is a fixed
  VL shape with `{{ .System }} {{ .Image }} {{ .Prompt }}`.
- `base_gguf_name` points at the full VL GGUF (LM + vision tower) —
  the caller is responsible for handling the single-file-vs-mmproj
  split when it matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dlm.export.ollama.modelfile import (
    _build_header,
    _build_license_line,
    _build_param_lines,
    _build_system_line,
    _resolve_num_ctx,
    _resolve_stops,
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
# `ollama --version` check belongs to Sprint 35.4's DoD but is
# tracked separately; this template assumes 0.4+.
_VL_TEMPLATE_BODY: str = (
    "{{ if .System }}{{ .System }}\n\n{{ end }}"
    "{{ .Image }}\n"
    "{{ .Prompt }}"
)

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
    stops = _resolve_stops(ctx.adapter_dir, _vl_template_row())
    header = _build_header(_as_modelfile_ctx(ctx))
    from_line = f"FROM ./{ctx.base_gguf_name}"
    adapter_line = (
        f"ADAPTER ./{ctx.adapter_gguf_name}" if ctx.adapter_gguf_name else None
    )
    template_block = f'TEMPLATE """{_VL_TEMPLATE_BODY}"""'
    num_ctx = _resolve_num_ctx(_as_modelfile_ctx(ctx))
    temperature = (
        ctx.override_temperature
        if ctx.override_temperature is not None
        else _VL_DEFAULT_TEMPERATURE
    )
    top_p = (
        ctx.override_top_p
        if ctx.override_top_p is not None
        else _VL_DEFAULT_TOP_P
    )
    param_lines = _build_param_lines(
        stops=stops,
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        draft_model=None,
    )
    system_line = _build_system_line(ctx.system_prompt)
    license_line = _build_license_line(ctx.spec)

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
    """Fallback DialectTemplate used by `_resolve_stops` for VL bases.

    The existing stops resolver reads from the adapter's
    `special_tokens_map.json` first, then falls back to the dialect
    row. VL bases don't map to one of the registered text dialects,
    so we pass a synthetic row whose `default_stops` are the VL
    fallback set.
    """
    return DialectTemplate(
        dialect="chatml",  # placeholder — _resolve_stops only reads defaults
        template_path=Path("/dev/null"),
        default_stops=_VL_FALLBACK_STOPS,
        default_temperature=_VL_DEFAULT_TEMPERATURE,
        default_top_p=_VL_DEFAULT_TOP_P,
        extra_stop_hints=(),
    )


def _as_modelfile_ctx(ctx: VlModelfileContext):  # type: ignore[no-untyped-def]
    """Adapter → the text ModelfileContext shape for shared helpers.

    `_build_header`, `_build_param_lines`, etc. accept ModelfileContext.
    Constructing one with the same field values lets us reuse the
    helpers without duplicating their logic here. The dialect field
    isn't read by those helpers — only the header + params blocks.
    """
    from dlm.export.ollama.modelfile import ModelfileContext

    return ModelfileContext(
        spec=ctx.spec,
        plan=ctx.plan,
        adapter_dir=ctx.adapter_dir,
        base_gguf_name=ctx.base_gguf_name,
        adapter_gguf_name=ctx.adapter_gguf_name,
        dlm_id=ctx.dlm_id,
        adapter_version=ctx.adapter_version,
        system_prompt=ctx.system_prompt,
        source_dlm_path=ctx.source_dlm_path,
        dlm_version=ctx.dlm_version,
        training_sequence_len=ctx.training_sequence_len,
        override_temperature=ctx.override_temperature,
        override_top_p=ctx.override_top_p,
    )
