"""Compatibility probes run against a `BaseModelSpec`.

Each probe is an independent function returning `ProbeResult`. `run_all`
aggregates them into a `ProbeReport`. Probes must be non-destructive
(read-only) and offline-safe where possible — the refresh-registry
script exercises them online.

Five probes:

1. `probe_architecture` — `AutoConfig(hf_id).architectures[0]` matches
   `spec.architecture`. Catches model-surgery mismatches and wrong
   revisions.
2. `probe_chat_template` — tokenizer has a non-empty `chat_template`
   attribute. Essential for Sprint 12's Modelfile emission.
3. `probe_gguf_arch_supported` — scans the vendored
   `convert_hf_to_gguf.py` for a `@Model.register("<arch>")` matching
   `spec.gguf_arch`. Sprint 11 owns the vendored submodule; until then
   the probe skips with a clear message.
4. `probe_pretokenizer_label` — reads `vendor/llama_cpp_pretokenizer_hashes.json`
   (populated by `scripts/bump-llama-cpp.sh`) and checks the spec's
   `tokenizer_pre` is a known **label**. Silent drift here causes
   silent GGUF export failures per findings §9; the probe catches it
   early. This is the offline fast-check.
5. `probe_pretokenizer_hash` — real fingerprint check (audit-04 B8 /
   CLAUDE.md pitfall #5). Tokenizes `_LLAMA_CPP_CHKTXT` and compares
   the sha256 of the stringified token sequence against a vendored
   per-label fingerprint table. Detects silent upstream tokenization
   changes that the label probe would miss. Requires a local HF
   cache; skipped cleanly otherwise.

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
VENDOR_PRETOKENIZER_FINGERPRINTS_DEFAULT: Final[Path] = (
    _REPO_ROOT / "vendor" / "llama_cpp_pretokenizer_fingerprints.json"
)

# The canonical test string llama.cpp uses at `convert_hf_to_gguf.py::
# get_vocab_base_pre`. Tokenize this under the model's BPE tokenizer,
# stringify the resulting token-id list, sha256 it — that digest is
# the fingerprint llama.cpp maps to one of its pre-tokenizer types.
# Keep verbatim; any edit here desynchronizes us from llama.cpp's
# identification logic (audit-04 B8 + CLAUDE.md pitfall #5).
_LLAMA_CPP_CHKTXT: Final[str] = (
    "\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n"
    "🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ "
    "🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 "
    "កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ "
    "------======= нещо на Български '''''''```````\"\"\"\"......!!!!!!?????? "
    "I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, "
    "'D you like some tea? We'Ve a'lL"
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
    """Scan vendored ``convert_hf_to_gguf.py`` for
    ``@Model.register("<gguf_arch>")`` or ``@ModelBase.register(...)``.

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

    # Match both the pre-rename ``@Model.register(...)`` and the current
    # ``@ModelBase.register(...)`` decorator forms. Upstream renamed the
    # class in
    # https://github.com/ggml-org/llama.cpp/commit/46e3556 (mid-2024);
    # the pre-rename form is preserved so this probe stays tolerant if
    # the vendored copy is ever pinned to an older tag.
    pattern = re.compile(r"""@(?:Model|ModelBase)\.register\(\s*["']([^"']+)["']""")
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


def probe_pretokenizer_label(
    spec: BaseModelSpec,
    *,
    hashes_path: Path | None = None,
) -> ProbeResult:
    """Check `spec.tokenizer_pre` is a known pre-tokenizer label.

    The vendored table is a JSON array of label strings that llama.cpp
    recognizes in `get_vocab_base_pre()`. Missing table → skip.

    NOTE (audit-04 M7): this is a *label* probe, not a hash probe.
    Sprint 11 will add real `probe_pretokenizer_hash` that canonically
    digests `tokenizer.json` and compares against llama.cpp's fingerprint
    table. For now we check coarse compatibility via the label.
    """
    path = hashes_path or VENDOR_PRETOKENIZER_HASHES_DEFAULT
    if not path.exists():
        return ProbeResult(
            name="pretokenizer_label",
            passed=True,
            detail=f"skipped: {path} not present (bump-llama-cpp.sh maintains it)",
            skipped=True,
        )

    try:
        labels = set(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        return ProbeResult(
            name="pretokenizer_label",
            passed=False,
            detail=f"table unreadable: {exc}",
        )
    except TypeError as exc:
        return ProbeResult(
            name="pretokenizer_label",
            passed=False,
            detail=f"table has wrong shape (expected list[str]): {exc}",
        )

    if spec.tokenizer_pre in labels:
        return ProbeResult(
            name="pretokenizer_label",
            passed=True,
            detail=f"{spec.tokenizer_pre!r} known to llama.cpp",
        )
    return ProbeResult(
        name="pretokenizer_label",
        passed=False,
        detail=(
            f"{spec.tokenizer_pre!r} not in vendored label table; "
            "run scripts/bump-llama-cpp.sh or pick another base"
        ),
    )


def probe_pretokenizer_hash(
    spec: BaseModelSpec,
    *,
    fingerprints_path: Path | None = None,
) -> ProbeResult:
    """Compute the real llama.cpp pre-tokenizer fingerprint and compare.

    Audit-04 B8 / CLAUDE.md pitfall #5. The label probe (above) only
    checks membership in a string table; llama.cpp itself identifies
    the pre-tokenizer by sha256-hashing the token-id sequence produced
    by tokenizing a stable test string (`_LLAMA_CPP_CHKTXT`). We do
    the same here — if the upstream tokenizer changes behavior (new
    revision, silently different merges), the fingerprint drifts and
    this probe fails loudly *before* a broken GGUF reaches Ollama.

    The fingerprint table at
    `vendor/llama_cpp_pretokenizer_fingerprints.json` is maintained by
    `scripts/bump-llama-cpp.sh`. Missing table or no entry for the
    spec's `tokenizer_pre` label → skip (the label probe still runs).

    Requires a local HF cache (`local_files_only=True`); skipped
    cleanly in CI environments without the tokenizer downloaded.
    """
    import hashlib

    path = fingerprints_path or VENDOR_PRETOKENIZER_FINGERPRINTS_DEFAULT
    if not path.exists():
        return ProbeResult(
            name="pretokenizer_hash",
            passed=True,
            detail=f"skipped: {path} not present (bump-llama-cpp.sh maintains it)",
            skipped=True,
        )

    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ProbeResult(
            name="pretokenizer_hash",
            passed=False,
            detail=f"fingerprint table unreadable: {exc}",
        )
    if not isinstance(table, dict):
        return ProbeResult(
            name="pretokenizer_hash",
            passed=False,
            detail="fingerprint table has wrong shape (expected {label: sha256})",
        )

    expected = table.get(spec.tokenizer_pre)
    if not isinstance(expected, str):
        return ProbeResult(
            name="pretokenizer_hash",
            passed=True,
            detail=(
                f"skipped: no fingerprint recorded for {spec.tokenizer_pre!r}; "
                "run scripts/bump-llama-cpp.sh to refresh the table"
            ),
            skipped=True,
        )

    try:
        from huggingface_hub.errors import GatedRepoError
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover — dev env always has transformers
        return ProbeResult(
            name="pretokenizer_hash",
            passed=True,
            detail=f"skipped: transformers unavailable ({exc})",
            skipped=True,
        )

    try:
        tok = AutoTokenizer.from_pretrained(
            spec.hf_id, revision=spec.revision, local_files_only=True
        )
    except GatedRepoError as exc:
        raise GatedModelError(spec.hf_id, spec.license_url) from exc
    except Exception as exc:
        # Not a probe *failure* — tokenizer simply isn't cached locally.
        # Online refresh-registry runs will exercise the real check.
        return ProbeResult(
            name="pretokenizer_hash",
            passed=True,
            detail=f"skipped: cannot load tokenizer offline ({type(exc).__name__})",
            skipped=True,
        )

    try:
        tokens = tok.encode(_LLAMA_CPP_CHKTXT)
    except Exception as exc:
        return ProbeResult(
            name="pretokenizer_hash",
            passed=False,
            detail=f"tokenizer.encode failed on chktxt: {type(exc).__name__}: {exc}",
        )

    digest = hashlib.sha256(str(tokens).encode()).hexdigest()
    if digest != expected:
        return ProbeResult(
            name="pretokenizer_hash",
            passed=False,
            detail=(
                f"pre-tokenizer drifted for {spec.tokenizer_pre!r}: "
                f"expected {expected[:12]}…, got {digest[:12]}…. "
                "Upstream may have changed tokenization; re-pin revision "
                "or run scripts/bump-llama-cpp.sh to refresh the fingerprint."
            ),
        )
    return ProbeResult(
        name="pretokenizer_hash",
        passed=True,
        detail=f"fingerprint matches {spec.tokenizer_pre!r} ({digest[:12]}…)",
    )


# --- aggregate ---------------------------------------------------------------


def run_all(spec: BaseModelSpec, *, skip_export_probes: bool = False) -> ProbeReport:
    """Run every probe; aggregate into a `ProbeReport`.

    `GatedModelError` from an individual probe propagates immediately —
    it's not a "probe failure" in the registry-drift sense; it's an
    acceptance-flow signal.

    `skip_export_probes=True` drops the three llama.cpp / GGUF-conversion
    checks (`gguf_arch_supported`, `pretokenizer_label`,
    `pretokenizer_hash`). Users opt into this when they want training
    + HF inference on a base whose architecture ships faster than our
    vendored llama.cpp can absorb (e.g. brand-new Qwen3 on a llama.cpp
    pin from last month). They forfeit `dlm export` to Ollama until
    the vendored copy catches up.
    """
    core = (
        probe_architecture(spec),
        probe_chat_template(spec),
    )
    if skip_export_probes:
        return ProbeReport(hf_id=spec.hf_id, results=core)
    results = (
        *core,
        probe_gguf_arch_supported(spec),
        probe_pretokenizer_label(spec),
        probe_pretokenizer_hash(spec),
    )
    return ProbeReport(hf_id=spec.hf_id, results=results)
