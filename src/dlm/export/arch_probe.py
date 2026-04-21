"""Probe the vendored `convert_hf_to_gguf.py` for VL arch coverage.

VL GGUF conversion is moving upstream; some VL architectures are
fully registered in `convert_hf_to_gguf.py` (LM + vision tower emit
together), some are partial (LM-only via the TextModel base, vision
tower ships separately via an mmproj class), and some aren't
registered at all.

`probe_gguf_arch(arch_class)` scans the vendored conversion script
for `@ModelBase.register(...)` decorators that name the arch, then
looks at the decorated class's base(s) to decide the verdict:

- **SUPPORTED** — at least one register binds to a class that does
  NOT inherit from `MmprojModel` (i.e. a TextModel or similar that
  emits the LM + combined VL path cleanly).
- **PARTIAL** — the arch appears only on `MmprojModel` subclasses,
  meaning llama.cpp handles the vision tower separately via an
  mmproj sidecar but no single-file GGUF covers the full VL model.
- **UNSUPPORTED** — the arch isn't registered anywhere; GGUF
  conversion would fail.

Callers (the export dispatcher) use this verdict to choose between
emitting GGUF, emitting GGUF with a warning banner about vision-
tower caveats, or falling back cleanly to the HF-snapshot path.

The probe is cheap (one read + regex scan) but we memoize on
`(llama_cpp_sha, arch_class)` so a single export run doesn't re-parse
the 200+ KB script on every registry lookup.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dlm.export.vendoring import convert_hf_to_gguf_py, pinned_tag


class SupportLevel(enum.StrEnum):
    """Verdict for one VL arch class against a vendored llama.cpp tree.

    StrEnum so manifests can store the value as a plain JSON string
    ("SUPPORTED" / "PARTIAL" / "UNSUPPORTED") without custom encoders.
    """

    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class ArchProbeResult:
    """Outcome of probing one arch class.

    `reason` is the human-readable explanation the dispatcher surfaces
    in banner/error messages; `llama_cpp_tag` records which vendored
    build the verdict came from (so users bumping the submodule can
    tell when the answer changes).
    """

    arch_class: str
    support: SupportLevel
    reason: str
    llama_cpp_tag: str | None


# Matches a `@ModelBase.register(...)` decorator and captures the arg
# list verbatim so we can look for the quoted arch-class name inside.
# Multi-line decorators are supported via DOTALL on the arg-list span.
_REGISTER_DECORATOR: Final[re.Pattern[str]] = re.compile(
    r"@ModelBase\.register\((?P<args>[^)]*)\)",
    re.DOTALL,
)

# Captures the first `class Foo(Bar, Baz):` line after a register
# decorator — we read the base-class list to decide SUPPORTED vs
# PARTIAL (MmprojModel-only registration → PARTIAL).
_CLASS_DEFINITION: Final[re.Pattern[str]] = re.compile(
    r"^\s*class\s+\w+\((?P<bases>[^)]*)\)\s*:",
    re.MULTILINE,
)

_MMPROJ_BASE: Final[str] = "MmprojModel"

_CACHE: dict[tuple[str | None, str], ArchProbeResult] = {}


def probe_gguf_arch(
    arch_class: str,
    *,
    llama_cpp_root: Path | None = None,
) -> ArchProbeResult:
    """Return the SUPPORTED/PARTIAL/UNSUPPORTED verdict for `arch_class`.

    `llama_cpp_root` overrides the default vendored root (used by
    tests that point the probe at a fixture tree); production callers
    omit it and let `dlm.export.vendoring` resolve via the env var or
    `vendor/llama.cpp/`.

    Raises `VendoringError` (surfaced by `convert_hf_to_gguf_py`) when
    the vendored tree doesn't contain the conversion script at all —
    a pre-Sprint-11 layout or an uninitialized submodule.
    """
    tag = pinned_tag(llama_cpp_root)
    cache_key = (tag, arch_class)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    script_path = convert_hf_to_gguf_py(llama_cpp_root)
    result = _probe_from_script(
        arch_class=arch_class,
        script_path=script_path,
        llama_cpp_tag=tag,
    )
    _CACHE[cache_key] = result
    return result


def clear_cache() -> None:
    """Drop the memoized probe results. Test-only; production never calls this."""
    _CACHE.clear()


def _probe_from_script(
    *,
    arch_class: str,
    script_path: Path,
    llama_cpp_tag: str | None,
) -> ArchProbeResult:
    """Core scanner: read the script, locate registrations, classify."""
    text = script_path.read_text(encoding="utf-8")
    bindings = _find_arch_bindings(text, arch_class)

    if not bindings:
        return ArchProbeResult(
            arch_class=arch_class,
            support=SupportLevel.UNSUPPORTED,
            reason=(
                f"{arch_class!r} not found in any @ModelBase.register(...) "
                f"decorator — vendored llama.cpp "
                f"(tag={llama_cpp_tag or 'unknown'}) does not know this "
                "architecture. GGUF conversion would fail."
            ),
            llama_cpp_tag=llama_cpp_tag,
        )

    # If any binding targets a non-Mmproj class, the LM converts as a
    # standalone GGUF. The vision tower may still ship separately via
    # a different registration, but single-file GGUF is viable.
    non_mmproj_bindings = [b for b in bindings if not _is_mmproj(b.bases)]
    if non_mmproj_bindings:
        reason = (
            f"{arch_class!r} registered on "
            f"{', '.join(sorted({b.class_name for b in non_mmproj_bindings}))} "
            f"in llama.cpp tag={llama_cpp_tag or 'unknown'}; LM converts "
            "cleanly via convert_hf_to_gguf.py."
        )
        return ArchProbeResult(
            arch_class=arch_class,
            support=SupportLevel.SUPPORTED,
            reason=reason,
            llama_cpp_tag=llama_cpp_tag,
        )

    # All bindings are Mmproj-only — vision tower ships as a
    # separate GGUF and the LM side needs a different arch string.
    mmproj_names = sorted({b.class_name for b in bindings})
    return ArchProbeResult(
        arch_class=arch_class,
        support=SupportLevel.PARTIAL,
        reason=(
            f"{arch_class!r} registered only on MmprojModel class(es) "
            f"{', '.join(mmproj_names)} in llama.cpp "
            f"tag={llama_cpp_tag or 'unknown'}. The vision tower converts "
            "but no single-file GGUF covers the full VL model."
        ),
        llama_cpp_tag=llama_cpp_tag,
    )


@dataclass(frozen=True)
class _ArchBinding:
    """One `@ModelBase.register(...)` → `class Foo(Bar):` pairing."""

    class_name: str
    bases: str  # raw comma-separated base-class list


def _find_arch_bindings(text: str, arch_class: str) -> list[_ArchBinding]:
    """Return every class registration that lists `arch_class` as an arg."""
    bindings: list[_ArchBinding] = []
    quoted_needles = (f'"{arch_class}"', f"'{arch_class}'")
    for match in _REGISTER_DECORATOR.finditer(text):
        args = match.group("args")
        if not any(needle in args for needle in quoted_needles):
            continue
        class_match = _CLASS_DEFINITION.search(text, match.end())
        if class_match is None:
            # Decorator at end of file with no following class — treat
            # as if it didn't bind to anything recognizable.
            continue
        # Pull the class name by rewinding from the `(` to `class `.
        class_name = _extract_class_name(text, class_match.start())
        if class_name is None:
            continue
        bindings.append(
            _ArchBinding(
                class_name=class_name,
                bases=class_match.group("bases"),
            )
        )
    return bindings


_CLASS_NAME_RE: Final[re.Pattern[str]] = re.compile(r"class\s+(\w+)\s*\(")


def _extract_class_name(text: str, class_def_start: int) -> str | None:
    """Parse `class Foo(...)` starting at `class_def_start`."""
    # The MULTILINE match on _CLASS_DEFINITION starts at the first
    # leading whitespace; re-match from there to capture the name.
    segment_end = text.find("(", class_def_start)
    if segment_end == -1:
        return None
    segment = text[class_def_start : segment_end + 1]
    name_match = _CLASS_NAME_RE.search(segment)
    return name_match.group(1) if name_match else None


def _is_mmproj(bases: str) -> bool:
    """True when the class inherits from MmprojModel (direct or indirect).

    We use a plain substring check rather than importing the class
    hierarchy — the script lives in the vendored tree and we don't
    want to import it into dlm's process just to classify. Good
    enough because the base list is short and MmprojModel is a
    distinctive name that won't false-match another base.
    """
    return _MMPROJ_BASE in bases
