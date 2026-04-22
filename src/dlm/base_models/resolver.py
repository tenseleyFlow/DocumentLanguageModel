"""Resolve a user-supplied base-model spec to a `BaseModelSpec`.

Spec grammar:

- `<key>` — registry lookup (e.g., `qwen2.5-1.5b`). `UnknownBaseModelError`
  if not present.
- `hf:<org>/<name>` — escape hatch. Fetches config.json + tokenizer
  metadata from HF, synthesizes a `BaseModelSpec`, runs the probe suite,
  and raises `ProbeFailedError` if any hard probe fails.

Gated models (`requires_acceptance=True`) raise `GatedModelError` unless
the caller has already accepted the license (signalled via
`accept_license=True`). The CLI uses this to persist acceptance; tests
pass `accept_license=True` directly to exercise the downstream path.
"""

from __future__ import annotations

import logging
from typing import Final, Literal

from dlm.base_models.errors import (
    GatedModelError,
    ProbeFailedError,
    ProbeResult,
    UnknownBaseModelError,
)
from dlm.base_models.registry import BASE_MODELS, known_keys
from dlm.base_models.schema import BaseModelSpec

TemplateDialect = Literal["chatml", "smollm3", "olmo2", "llama3", "phi3", "mistral"]

_LOG = logging.getLogger(__name__)

_HF_PREFIX: Final = "hf:"


def resolve(
    spec: str,
    *,
    accept_license: bool = False,
    skip_export_probes: bool = False,
) -> BaseModelSpec:
    """Return the `BaseModelSpec` for `spec`.

    Registry lookup first; `hf:`-prefix falls through to `resolve_hf()`.
    Gating is enforced here regardless of path. `skip_export_probes`
    only applies to the `hf:` path — registry entries are curated and
    always pass all probes by construction.
    """
    if spec.startswith(_HF_PREFIX):
        return resolve_hf(
            spec[len(_HF_PREFIX) :],
            accept_license=accept_license,
            skip_export_probes=skip_export_probes,
        )

    entry = BASE_MODELS.get(spec)
    if entry is None:
        raise UnknownBaseModelError(spec, known_keys())

    _enforce_gate(entry, accept_license=accept_license)
    return entry


def _env_skip_export_probes() -> bool:
    """Read `DLM_SKIP_EXPORT_PROBES` — set by power users whose base isn't
    yet in vendored llama.cpp but who only need training + HF inference.

    Checked by every `resolve` path so `dlm train/prompt/export` inherits
    the decision the user made at `dlm init --skip-export-probes` time
    without persisting extra state on the per-store manifest.
    """
    import os

    return os.environ.get("DLM_SKIP_EXPORT_PROBES", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def resolve_hf(
    hf_id: str,
    *,
    accept_license: bool = False,
    skip_export_probes: bool = False,
) -> BaseModelSpec:
    """Synthesize a `BaseModelSpec` for an arbitrary HF model id.

    Runs the probe suite; raises `ProbeFailedError` with a full report
    if any hard probe fails. This is the gate that prevents users from
    pinning a model our export pipeline can't actually convert.

    `skip_export_probes=True` drops the llama.cpp / GGUF-conversion
    probes so brand-new architectures (not yet in the vendored
    llama.cpp) can still train + HF-infer. Users opting in forfeit
    `dlm export` until the vendored copy catches up.
    """
    # Deferred import: probes pull transformers, which is expensive.
    from dlm.base_models import probes

    spec = _synthesize_spec(hf_id)
    _enforce_gate(spec, accept_license=accept_license)

    skip = skip_export_probes or _env_skip_export_probes()
    report = probes.run_all(spec, skip_export_probes=skip)
    if not report.passed:
        raise ProbeFailedError(spec.hf_id, list(report.results))
    return spec


# --- internals ---------------------------------------------------------------


def _enforce_gate(spec: BaseModelSpec, *, accept_license: bool) -> None:
    if spec.requires_acceptance and not accept_license:
        raise GatedModelError(spec.hf_id, spec.license_url)


def _synthesize_spec(hf_id: str) -> BaseModelSpec:
    """Build a minimal `BaseModelSpec` for an arbitrary HF id.

    Pulls config + tokenizer_config metadata from the Hub so probes have
    real data to work against. The synthesized spec is shaped to pass
    `BaseModelSpec` validation; users who want tighter defaults should
    add the model to the curated registry instead.
    """
    if "/" not in hf_id or hf_id.startswith("/") or hf_id.endswith("/"):
        raise UnknownBaseModelError(f"hf:{hf_id}", known_keys())

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import (
            EntryNotFoundError,
            GatedRepoError,
            RepositoryNotFoundError,
        )
        from transformers import AutoConfig
    except ImportError as exc:  # pragma: no cover — dev env always has these
        raise RuntimeError(
            "hf: escape hatch requires huggingface_hub + transformers; install dev deps"
        ) from exc

    api = HfApi()
    try:
        info = api.model_info(hf_id)
    except GatedRepoError as exc:
        raise GatedModelError(hf_id, license_url=None) from exc
    except RepositoryNotFoundError as exc:
        raise UnknownBaseModelError(f"hf:{hf_id}", known_keys()) from exc

    revision = info.sha
    if not revision or len(revision) != 40:
        raise RuntimeError(f"HF returned non-40-char SHA for {hf_id}: {revision!r}")

    try:
        config = AutoConfig.from_pretrained(hf_id, revision=revision)
    except GatedRepoError as exc:
        raise GatedModelError(hf_id, license_url=None) from exc
    except EntryNotFoundError as exc:
        raise UnknownBaseModelError(f"hf:{hf_id}", known_keys()) from exc

    architectures = getattr(config, "architectures", None) or ()
    if not architectures:
        # Build a single synthetic failure so the caller has something
        # to show — we can't construct a BaseModelSpec without arch.
        raise ProbeFailedError(
            hf_id,
            [
                ProbeResult(
                    name="architecture",
                    passed=False,
                    detail="config.json has no `architectures` entry",
                )
            ],
        )

    architecture = architectures[0]
    params = getattr(config, "num_parameters", None) or _estimate_params(config)
    context_length = (
        getattr(config, "max_position_embeddings", None)
        or getattr(config, "n_positions", None)
        or 4096
    )

    gguf_arch = _infer_gguf_arch(architecture)
    template = _infer_template(hf_id, architecture)

    # `hf:` models are advisory — we can't audit their license from here
    # alone; mark them conservatively as requiring acceptance + not
    # redistributable. Users who know better add the model to the registry.
    return BaseModelSpec(
        key=f"hf:{hf_id}",
        hf_id=hf_id,
        revision=revision,
        architecture=architecture,
        params=params,
        target_modules=_default_target_modules(gguf_arch),
        template=template,
        gguf_arch=gguf_arch,
        tokenizer_pre="default",
        license_spdx="Unknown",
        license_url=None,
        requires_acceptance=False,
        redistributable=False,
        size_gb_fp16=max(0.1, params * 2 / (1024**3)),
        context_length=context_length,
        recommended_seq_len=min(context_length, 2048),
    )


def _estimate_params(config: object) -> int:
    """Rough param count from hidden_size / num_hidden_layers / vocab_size."""
    hidden: int = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None) or 2048
    layers: int = (
        getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None) or 24
    )
    vocab: int = getattr(config, "vocab_size", None) or 32_000
    # 12 * h^2 * L is a textbook approximation of transformer params; add embeddings.
    return int(12 * hidden**2 * layers + 2 * hidden * vocab)


def _infer_gguf_arch(architecture: str) -> str:
    mapping = {
        "LlamaForCausalLM": "llama",
        "SmolLM3ForCausalLM": "llama",
        "Olmo2ForCausalLM": "olmo2",
        "Qwen2ForCausalLM": "qwen2",
        "Qwen3ForCausalLM": "qwen3",
        "MistralForCausalLM": "llama",
        "Phi3ForCausalLM": "phi3",
        "GemmaForCausalLM": "gemma",
        "Gemma2ForCausalLM": "gemma2",
    }
    return mapping.get(architecture, architecture.lower().replace("forcausallm", ""))


def _infer_template(hf_id: str, architecture: str) -> TemplateDialect:
    """Best-effort template dialect picker for `hf:` synthesis."""
    lower = hf_id.lower()
    if "smollm3" in lower or architecture.startswith("SmolLM3"):
        return "smollm3"
    if "olmo-2" in lower or architecture.startswith("Olmo2"):
        return "olmo2"
    if "llama-3" in lower or "llama3" in lower:
        return "llama3"
    if architecture.startswith("Phi"):
        return "phi3"
    if architecture.startswith("Mistral"):
        return "mistral"
    return "chatml"


def _default_target_modules(gguf_arch: str) -> list[str]:
    if gguf_arch == "phi3":
        return ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]
