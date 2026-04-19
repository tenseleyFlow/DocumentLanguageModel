"""Compatibility probes run against a `BaseModelSpec`.

Each probe is an independent function returning `ProbeResult`. `run_all`
aggregates them into a `ProbeReport`. Probes must be non-destructive
(read-only) and offline-safe where possible — the refresh-registry
script exercises them online.

Four probes:

1. `probe_architecture` — `AutoConfig(hf_id).architectures[0]` matches
   `spec.architecture`. Catches model-surgery mismatches and wrong
   revisions.
2. `probe_chat_template` — tokenizer has a non-empty `chat_template`
   attribute. Essential for Sprint 12's Modelfile emission.
3. `probe_gguf_arch_supported` — scans the vendored
   `convert_hf_to_gguf.py` for a `@Model.register("<arch>")` matching
   `spec.gguf_arch`. Sprint 11 owns the vendored submodule; until then
   the probe skips with a clear message.
4. `probe_pretokenizer_hash` — reads `vendor/llama_cpp_pretokenizer_hashes.json`
   (populated by `scripts/bump-llama-cpp.sh`) and checks the spec's
   `tokenizer_pre` is a known label. Silent drift here causes silent
   GGUF export failures per findings §9; the probe catches it early.

Heavy imports (`transformers.AutoConfig`, `AutoTokenizer`) happen
inside each probe so the module loads cheaply.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Final

from dlm.base_models.errors import GatedModelError, ProbeReport, ProbeResult
from dlm.base_models.schema import BaseModelSpec

_LOG = logging.getLogger(__name__)

# Vendored artifact locations (Sprint 11 populates `vendor/llama.cpp`).
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
VENDOR_LLAMA_CPP_DEFAULT: Final[Path] = _REPO_ROOT / "vendor" / "llama.cpp"
VENDOR_PRETOKENIZER_HASHES_DEFAULT: Final[Path] = (
    _REPO_ROOT / "vendor" / "llama_cpp_pretokenizer_hashes.json"
)


# --- individual probes --------------------------------------------------------


def probe_architecture(spec: BaseModelSpec) -> ProbeResult:
    """`AutoConfig.from_pretrained(hf_id, revision).architectures[0]` matches."""
    try:
        from huggingface_hub.errors import GatedRepoError
        from transformers import AutoConfig
    except ImportError as exc:  # pragma: no cover — dev env always has transformers
        return ProbeResult(
            name="architecture",
            passed=True,
            detail=f"skipped: transformers unavailable ({exc})",
            skipped=True,
        )

    try:
        cfg = AutoConfig.from_pretrained(spec.hf_id, revision=spec.revision)
    except GatedRepoError as exc:
        raise GatedModelError(spec.hf_id, spec.license_url) from exc
    except Exception as exc:
        return ProbeResult(
            name="architecture",
            passed=False,
            detail=f"load failed: {type(exc).__name__}: {exc}",
        )

    architectures = getattr(cfg, "architectures", None)
    if not architectures:
        return ProbeResult(
            name="architecture",
            passed=False,
            detail="config.json has no `architectures` entry",
        )

    observed = architectures[0]
    if observed != spec.architecture:
        return ProbeResult(
            name="architecture",
            passed=False,
            detail=f"expected {spec.architecture!r}, got {observed!r}",
        )
    return ProbeResult(
        name="architecture",
        passed=True,
        detail=f"matched {observed!r}",
    )


def probe_chat_template(spec: BaseModelSpec) -> ProbeResult:
    """Tokenizer carries a non-empty `chat_template` attribute."""
    try:
        from huggingface_hub.errors import GatedRepoError
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        return ProbeResult(
            name="chat_template",
            passed=True,
            detail=f"skipped: transformers unavailable ({exc})",
            skipped=True,
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision)
    except GatedRepoError as exc:
        raise GatedModelError(spec.hf_id, spec.license_url) from exc
    except Exception as exc:
        return ProbeResult(
            name="chat_template",
            passed=False,
            detail=f"load failed: {type(exc).__name__}: {exc}",
        )

    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return ProbeResult(
            name="chat_template",
            passed=False,
            detail="tokenizer has no chat_template",
        )
    return ProbeResult(
        name="chat_template",
        passed=True,
        detail=f"present ({len(template)} chars)",
    )


def probe_gguf_arch_supported(
    spec: BaseModelSpec,
    *,
    vendor_path: Path | None = None,
) -> ProbeResult:
    """Scan vendored `convert_hf_to_gguf.py` for `@Model.register("<gguf_arch>")`.

    Until Sprint 11 lands the submodule, this probe skips.
    """
    script = (vendor_path or VENDOR_LLAMA_CPP_DEFAULT) / "convert_hf_to_gguf.py"
    if not script.exists():
        return ProbeResult(
            name="gguf_arch",
            passed=True,
            detail=f"skipped: {script} not present (Sprint 11 vendors llama.cpp)",
            skipped=True,
        )

    try:
        source = script.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ProbeResult(
            name="gguf_arch",
            passed=False,
            detail=f"read failed: {exc}",
        )

    # `@Model.register("qwen2", …)` or `@Model.register("qwen2")`
    pattern = re.compile(r"""@Model\.register\(\s*["']([^"']+)["']""")
    found_archs = set(pattern.findall(source))
    if spec.gguf_arch in found_archs:
        return ProbeResult(
            name="gguf_arch",
            passed=True,
            detail=f"converter registers {spec.gguf_arch!r}",
        )
    return ProbeResult(
        name="gguf_arch",
        passed=False,
        detail=(
            f"{spec.gguf_arch!r} not in convert_hf_to_gguf.py (found: {sorted(found_archs)[:5]}…)"
        ),
    )


def probe_pretokenizer_hash(
    spec: BaseModelSpec,
    *,
    hashes_path: Path | None = None,
) -> ProbeResult:
    """Check `spec.tokenizer_pre` is a known pre-tokenizer label.

    The vendored table is a JSON array of label strings that llama.cpp
    recognizes in `get_vocab_base_pre()`. Missing table → skip.
    """
    path = hashes_path or VENDOR_PRETOKENIZER_HASHES_DEFAULT
    if not path.exists():
        return ProbeResult(
            name="pretokenizer_hash",
            passed=True,
            detail=f"skipped: {path} not present (bump-llama-cpp.sh maintains it)",
            skipped=True,
        )

    try:
        labels = set(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        return ProbeResult(
            name="pretokenizer_hash",
            passed=False,
            detail=f"table unreadable: {exc}",
        )
    except TypeError as exc:
        return ProbeResult(
            name="pretokenizer_hash",
            passed=False,
            detail=f"table has wrong shape (expected list[str]): {exc}",
        )

    if spec.tokenizer_pre in labels:
        return ProbeResult(
            name="pretokenizer_hash",
            passed=True,
            detail=f"{spec.tokenizer_pre!r} known to llama.cpp",
        )
    return ProbeResult(
        name="pretokenizer_hash",
        passed=False,
        detail=(
            f"{spec.tokenizer_pre!r} not in vendored hash table; "
            "run scripts/bump-llama-cpp.sh or pick another base"
        ),
    )


# --- aggregate ---------------------------------------------------------------


def run_all(spec: BaseModelSpec) -> ProbeReport:
    """Run every probe; aggregate into a `ProbeReport`.

    `GatedModelError` from an individual probe propagates immediately —
    it's not a "probe failure" in the registry-drift sense; it's an
    acceptance-flow signal.
    """
    results = (
        probe_architecture(spec),
        probe_chat_template(spec),
        probe_gguf_arch_supported(spec),
        probe_pretokenizer_hash(spec),
    )
    return ProbeReport(hf_id=spec.hf_id, results=results)
