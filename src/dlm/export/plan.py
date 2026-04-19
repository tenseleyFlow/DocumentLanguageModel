"""`ExportPlan` — what the user wants, resolved against frontmatter defaults.

One ExportPlan per `dlm export` invocation. Drives both the runner
(which subprocesses to launch, in what order) and the layout (which
directory names the outputs land in). Frozen + `extra="forbid"` so
a typo in the CLI can't silently produce a subtly-different export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from dlm.export.errors import UnsafeMergeError

# llama.cpp supports many more quant types; we ship the five that
# appear in the Sprint 11 spec. Extending this Literal requires a
# bump to vendored llama.cpp + a registry-probe pass.
QuantLevel = Literal["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]

# Bytes-per-param table — used by `dlm export` to estimate output size
# upfront. Matches llama.cpp's documented ratios; not load-bearing for
# correctness.
QUANT_BYTES_PER_PARAM: dict[QuantLevel, float] = {
    "Q4_K_M": 0.56,
    "Q5_K_M": 0.66,
    "Q6_K": 0.83,
    "Q8_0": 1.0,
    "F16": 2.0,
}

DEFAULT_QUANT: QuantLevel = "Q4_K_M"


def valid_quants() -> tuple[str, ...]:
    """Return the tuple of accepted quant strings (for CLI help text)."""
    return get_args(QuantLevel)


@dataclass(frozen=True)
class ExportPlan:
    """Resolved export configuration.

    `dequantize_confirmed` is the user's explicit opt-in to merging a
    QLoRA adapter (pitfall #3). Without it, `--merged` on a QLoRA
    checkpoint raises `UnsafeMergeError`.

    `ollama_name` is the tag Sprint 12 registers with; Sprint 11 just
    stamps it into the export manifest for later lookup.
    """

    quant: QuantLevel = DEFAULT_QUANT
    merged: bool = False
    dequantize_confirmed: bool = False
    include_template: bool = True
    ollama_name: str | None = None

    def __post_init__(self) -> None:
        if self.quant not in get_args(QuantLevel):
            raise ValueError(f"unknown quant {self.quant!r}; expected one of {valid_quants()}")
        if self.dequantize_confirmed and not self.merged:
            raise ValueError(
                "--dequantize only makes sense with --merged; drop the flag or add --merged."
            )

    def assert_merge_safe(self, *, was_qlora: bool) -> None:
        """Gate the merged-QLoRA path behind an explicit opt-in (pitfall #3).

        Merging LoRA deltas into a 4-bit base silently loses precision.
        We refuse unless the user passed `--dequantize`, which commits
        to a fp16 merge (dequantize base → merge → re-quantize on export).
        """
        if self.merged and was_qlora and not self.dequantize_confirmed:
            raise UnsafeMergeError(
                "This adapter was trained on a 4-bit base (QLoRA). "
                "Merging loses precision silently.\n"
                "Re-run with `--merged --dequantize` to proceed in fp16, "
                "or drop `--merged` to use the default separate-GGUF path."
            )

    def estimated_base_bytes(self, base_params: int) -> int:
        """Rough output size estimate for `base.<quant>.gguf`."""
        return int(base_params * QUANT_BYTES_PER_PARAM[self.quant])


def resolve_export_plan(
    *,
    cli_quant: str | None,
    cli_merged: bool,
    cli_dequantize: bool,
    cli_no_template: bool,
    cli_ollama_name: str | None,
    frontmatter_default_quant: str | None,
) -> ExportPlan:
    """Compose an `ExportPlan` from CLI + frontmatter + built-in defaults.

    Precedence: CLI flag > frontmatter default > built-in default.
    The resolver validates the quant string early so the CLI can
    surface "unknown quant" before any subprocess launches.
    """
    chosen = cli_quant or frontmatter_default_quant or DEFAULT_QUANT
    if chosen not in get_args(QuantLevel):
        raise ValueError(f"unknown quant {chosen!r}; expected one of {valid_quants()}")

    return ExportPlan(
        quant=chosen,  # type: ignore[arg-type]
        merged=cli_merged,
        dequantize_confirmed=cli_dequantize,
        include_template=not cli_no_template,
        ollama_name=cli_ollama_name,
    )
