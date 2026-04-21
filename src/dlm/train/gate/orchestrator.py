"""Post-SFT orchestration for the learned adapter gate.

Consumes a `ParsedDlm` whose `training.gate.enabled == True` and at
least two named adapters are declared, plus a loaded base model +
tokenizer. Extracts adapter-tagged sections, mean-pools prompt
embeddings from the base model, calls `train_gate`, and records the
resulting per-adapter statistics via the metrics recorder.

Separated from `trainer.run` so the consumer-side wiring in `run_all`
stays small and so unit tests can exercise the sample-extraction logic
against a stubbed embedder without loading a real HF base.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dlm.train.gate.errors import GateTrainingError
from dlm.train.gate.trainer import (
    GateTrainingResult,
    GateTrainingSample,
    train_gate,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from dlm.doc.parser import ParsedDlm
    from dlm.metrics.recorder import MetricsRecorder
    from dlm.store.paths import StorePath

_LOG = logging.getLogger(__name__)

# Cap the prompt text we feed through the embedder. Sections can be
# arbitrarily long (especially PROSE); we only need a stable prefix to
# condition the gate — longer inputs just increase the embedding cost.
_MAX_PROMPT_CHARS = 2048


@dataclass(frozen=True)
class GateProbe:
    """One (adapter_name, prompt_text) pair harvested from a section."""

    adapter_name: str
    prompt: str


def probes_from_sections(parsed: ParsedDlm) -> list[GateProbe]:
    """Extract adapter-tagged supervising prompts from `parsed.sections`.

    Shape per section type:

    - INSTRUCTION — the first `### Q` block's text.
    - PREFERENCE  — the Prompt field from the first turn.
    - PROSE       — first `_MAX_PROMPT_CHARS` chars of content.

    Sections without an `adapter` tag are dropped — they train into the
    SFT adapter but have no per-adapter routing signal for the gate.
    Malformed instruction / preference bodies are skipped (not fatal)
    so a single authoring slip doesn't kill the gate pass.
    """
    from dlm.data.errors import InstructionParseError, PreferenceParseError
    from dlm.data.instruction_parser import parse_instruction_body
    from dlm.data.preference_parser import parse_preference_body
    from dlm.doc.sections import SectionType

    probes: list[GateProbe] = []
    for section in parsed.sections:
        tag = section.adapter
        if not tag:
            continue
        try:
            if section.type is SectionType.INSTRUCTION:
                pairs = parse_instruction_body(section.content, section_id=section.section_id)
                if pairs:
                    probes.append(GateProbe(tag, pairs[0].question))
            elif section.type is SectionType.PREFERENCE:
                turns = parse_preference_body(section.content, section_id=section.section_id)
                if turns:
                    probes.append(GateProbe(tag, turns[0].prompt))
            else:  # PROSE
                probes.append(GateProbe(tag, section.content[:_MAX_PROMPT_CHARS]))
        except (InstructionParseError, PreferenceParseError) as exc:
            _LOG.warning(
                "gate: skipping unparseable %s section %s: %s",
                section.type.value,
                section.section_id,
                exc,
            )
    return probes


def run_post_sft_gate(
    store: StorePath,
    parsed: ParsedDlm,
    *,
    run_id: int,
    recorder: MetricsRecorder,
    embed: Callable[[str], torch.Tensor],
    input_dim: int,
    seed: int | None = None,
) -> GateTrainingResult | None:
    """Train the gate on supervising prompts and record per-adapter stats.

    Returns `None` when the gate is disabled or the document declares
    fewer than two adapters; callers don't need to check the config
    themselves. Embedding is injected as a callable so tests can stub
    it without loading an HF model. Any `GateTrainingError` is logged
    and swallowed — Sprint 34 treats gate training as best-effort so
    an SFT commit is never undone by a gate hiccup.
    """
    training = parsed.frontmatter.training
    gate_cfg = training.gate
    if not gate_cfg.enabled:
        return None
    adapters = training.adapters
    if adapters is None or len(adapters) < 2:
        return None
    adapter_names = tuple(adapters.keys())

    probes = probes_from_sections(parsed)
    samples = [
        GateTrainingSample(embedding=embed(p.prompt), adapter_name=p.adapter_name)
        for p in probes
        if p.adapter_name in adapter_names
    ]

    effective_seed = seed if seed is not None else parsed.frontmatter.training.seed

    try:
        result = train_gate(
            store,
            samples,
            adapter_names=adapter_names,
            input_dim=input_dim,
            hidden_proj_dim=gate_cfg.hidden_proj_dim,
            steps=gate_cfg.steps,
            lr=gate_cfg.lr,
            cold_start_floor=gate_cfg.cold_start_floor,
            entropy_lambda=gate_cfg.entropy_lambda,
            seed=effective_seed,
        )
    except GateTrainingError as exc:
        _LOG.warning("gate: training failed, leaving store gate-less: %s", exc)
        return None

    _record_gate_events(
        recorder=recorder,
        run_id=run_id,
        adapter_names=adapter_names,
        result=result,
    )
    return result


def _record_gate_events(
    *,
    recorder: MetricsRecorder,
    run_id: int,
    adapter_names: tuple[str, ...],
    result: GateTrainingResult,
) -> None:
    """Emit one `GateEvent` row per declared adapter.

    Uniform-mode results record `mean_weight = 1/N` across adapters so
    `dlm show` can still render a per-adapter table (the gate exists,
    it just hasn't been trained yet).
    """
    from dlm.metrics.events import GateEvent

    n = len(adapter_names)
    uniform_weight = 1.0 / n if n else 0.0
    for name in adapter_names:
        sample_count = result.per_adapter_samples.get(name, 0)
        if result.mode == "trained":
            mean_weight = result.per_adapter_mean_weight.get(name, uniform_weight)
        else:
            mean_weight = uniform_weight
        recorder.record_gate(
            GateEvent(
                run_id=run_id,
                adapter_name=name,
                mean_weight=mean_weight,
                sample_count=sample_count,
                mode=result.mode,
            )
        )
