"""Probe prompt derivation.

Two paths, picked in order:

1. **Explicit `!probe` markers.** Any `::instruction::` section whose
   question header is `### Q !probe` is a user-declared probe. The
   question body is the prompt; the answer body is the reference (not
   enforced at eval time — generation is compared to the reference
   only by human inspection in logs).
2. **Auto-sample from val split.** If no explicit probes exist, pick
   up to `k` questions from INSTRUCTION sections via a seed-stable
   sample. This guarantees every training run logs *something*; the
   user graduates to explicit `!probe` markers once they know which
   questions matter.

The emitted probes are just strings — `dlm.inference.generate` consumes
them with deterministic settings (temperature 0, do_sample=False) so
the output diff between runs is meaningful.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from dlm.data.errors import InstructionParseError
from dlm.data.instruction_parser import QAPair, parse_instruction_body
from dlm.doc.sections import Section, SectionType

_PROBE_MARKER = "!probe"
_PROBE_HEADER = f"### Q {_PROBE_MARKER}"
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Probe:
    """One probe prompt + its reference answer (for log inspection)."""

    prompt: str
    reference: str | None = None
    section_id: str = ""


def extract_probes(sections: list[Section], *, k: int = 3, seed: int = 0) -> list[Probe]:
    """Return up to `k` probes derived from `sections`.

    Explicit `!probe`-marked questions take priority; if `k` explicit
    probes are found, auto-sampling is skipped. Otherwise the remainder
    is filled from INSTRUCTION section Q/A pairs via a deterministic
    sample.
    """
    parsed_pairs = _parse_instruction_sections(sections)
    explicit = list(_extract_explicit_probes(sections, parsed_pairs=parsed_pairs))
    if len(explicit) >= k:
        return explicit[:k]

    needed = k - len(explicit)
    seen_prompts = {p.prompt for p in explicit}
    auto = _auto_sample_probes(
        sections,
        k=needed,
        seed=seed,
        exclude=seen_prompts,
        parsed_pairs=parsed_pairs,
    )
    return [*explicit, *auto]


# --- internals ---------------------------------------------------------------


def _extract_explicit_probes(
    sections: list[Section],
    *,
    parsed_pairs: dict[str, list[QAPair]],
) -> list[Probe]:
    """Find INSTRUCTION Q/A pairs whose question starts with `!probe`.

    The `!probe` marker appears on the Q header line; the Q body is the
    prompt text. We rewrite the body by stripping the leading `!probe`
    token and any whitespace so the prompt itself doesn't carry the
    marker token into model input.
    """
    out: list[Probe] = []
    for section in sections:
        if section.type is not SectionType.INSTRUCTION:
            continue
        pairs = parsed_pairs.get(section.section_id, [])
        for pair in pairs:
            # After normalization every probe pair sits in a private
            # namespace; we flag them via a sentinel prefix in the body.
            if pair.question.startswith(f"{_PROBE_MARKER}:"):
                prompt = pair.question[len(_PROBE_MARKER) + 1 :].strip()
                out.append(
                    Probe(
                        prompt=prompt,
                        reference=pair.answer,
                        section_id=section.section_id,
                    )
                )
    return out


def _normalize_probe_markers(body: str) -> str:
    """Rewrite `### Q !probe` headers so the instruction parser accepts them.

    The base parser treats anything after `### Q` as inline content and
    rejects it. We want `!probe` as a prefix marker rather than part of
    the grammar, so pre-process by stripping the marker off the header
    line and planting it on the first body line with a `!probe:` prefix.
    """
    lines = body.splitlines()
    rewritten: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == _PROBE_HEADER:
            rewritten.append("### Q")
            # Find the first non-blank body line and prefix it.
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                rewritten.append(lines[i])
                i += 1
            if i < len(lines):
                rewritten.append(f"{_PROBE_MARKER}:{lines[i]}")
                i += 1
            continue
        rewritten.append(line)
        i += 1
    return "\n".join(rewritten)


def _auto_sample_probes(
    sections: list[Section],
    *,
    k: int,
    seed: int,
    exclude: set[str],
    parsed_pairs: dict[str, list[QAPair]],
) -> list[Probe]:
    """Deterministically pick `k` questions from INSTRUCTION sections.

    Hashes `(seed, question)` and keeps the top-k by hash — a stable
    weighted sample without needing `random.Random`. Excludes any
    prompt already in `exclude` (typically explicit probes).

    Parses the *normalized* section body so sections containing
    `### Q !probe` headers don't trip the strict instruction parser
    — we strip the marker, then filter out `!probe:`-prefixed bodies
    (those are the explicit probes, which the caller has already
    captured).
    """
    if k <= 0:
        return []

    candidates: list[Probe] = []
    for section in sections:
        if section.type is not SectionType.INSTRUCTION:
            continue
        pairs = parsed_pairs.get(section.section_id, [])
        for pair in pairs:
            # Skip explicit probes (their question body was prefixed
            # with `!probe:` by the normalizer) — the caller handles
            # them separately.
            if pair.question.startswith(f"{_PROBE_MARKER}:"):
                continue
            if pair.question in exclude:
                continue
            candidates.append(
                Probe(
                    prompt=pair.question,
                    reference=pair.answer,
                    section_id=section.section_id,
                )
            )

    if not candidates:
        return []

    # Stable hash-based ordering.
    keyed = sorted(candidates, key=lambda p: _probe_sort_key(p.prompt, seed))
    return keyed[:k]


def _probe_sort_key(prompt: str, seed: int) -> str:
    h = hashlib.sha256(f"{seed}\x00{prompt}".encode())
    return h.hexdigest()


def _parse_instruction_sections(sections: list[Section]) -> dict[str, list[QAPair]]:
    """Parse instruction sections once so malformed blocks warn once."""
    parsed: dict[str, list[QAPair]] = {}
    for section in sections:
        if section.type is not SectionType.INSTRUCTION:
            continue
        try:
            parsed[section.section_id] = parse_instruction_body(
                _normalize_probe_markers(section.content),
                section_id=section.section_id,
            )
        except InstructionParseError as exc:
            _LOG.warning(
                "probe extraction skipped malformed instruction section %s at line %d: %s",
                exc.section_id,
                exc.section_line,
                exc,
            )
            parsed[section.section_id] = []
    return parsed
