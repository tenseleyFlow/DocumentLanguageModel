"""Static mean-gate fallback for Ollama / llama.cpp export.

The learned gate (Sprint 34) runs in PyTorch at `dlm prompt` time. The
GGUF runtime (Ollama, llama.cpp) can't evaluate a torch module at
inference, so when the user runs `dlm export` on a document with
`training.gate.enabled: true` we fall back to:

1. Compute the gate's softmax output on every training prompt.
2. Average those probability vectors across the corpus → one fixed
   weight per adapter.
3. Emit the averaged weights as the Modelfile's `--adapter-mix`
   coefficients.

The exported model is a statically-weighted merge of the named
adapters — lossless vs today's shipped behavior, and strictly better
than asking the user to guess coefficients. Dynamic per-prompt routing
is the `dlm prompt` / `dlm repl` path only.

The export manifest records ``gate_mode: "static_mean"`` so downstream
tooling can tell an exported-with-mean-gate build apart from a
hand-picked `--adapter-mix`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from dlm.train.gate.module import Gate, GateMetadata


def mean_gate_weights(
    gate: Gate,
    metadata: GateMetadata,
    prompt_embeddings: list[torch.Tensor],
) -> list[tuple[str, float]]:
    """Average ``gate(embedding)`` across the training prompts.

    Returns ``[(adapter_name, weight), ...]`` suitable for direct
    substitution into ``dlm export --adapter-mix``. Weights sum to
    1.0 (gate output is softmax; average of softmax is still on the
    simplex) but we don't renormalize defensively — a numeric-drift
    renorm would mask bugs.

    Raises ``ValueError`` if ``prompt_embeddings`` is empty — a
    zero-prompt corpus has nothing to average.
    """
    import torch

    if not prompt_embeddings:
        raise ValueError("mean_gate_weights requires >= 1 prompt embedding")

    with torch.no_grad():
        stacked = torch.stack([e.detach().to(torch.float32).reshape(-1) for e in prompt_embeddings])
        if stacked.shape[1] != metadata.input_dim:
            raise ValueError(
                f"prompt embedding dim {stacked.shape[1]} != gate input_dim "
                f"{metadata.input_dim} (base model mismatch?)"
            )
        probs = gate(stacked)  # (N, n_adapters)
        mean = probs.mean(dim=0)

    return [(name, float(mean[i].item())) for i, name in enumerate(metadata.adapter_names)]


def uniform_adapter_mix(adapter_names: tuple[str, ...]) -> list[tuple[str, float]]:
    """Mean-gate fallback for uniform-mode gates (cold-start).

    Returns ``[(name, 1/N), ...]`` — the export path for a doc that has
    a gate declared but where the gate trainer chose the uniform
    fallback because the corpus was too small.
    """
    n = len(adapter_names)
    if n == 0:
        return []
    w = 1.0 / n
    return [(name, w) for name in adapter_names]


def resolve_gate_mix(
    store: object,
    parsed: object,
) -> list[tuple[str, float]] | None:
    """Derive a static ``--adapter-mix`` from the learned gate's state.

    Returns ``None`` when the document has no enabled gate, declares
    fewer than two adapters, or the store has no persisted
    ``gate_config.json``. Otherwise returns one of:

    - **uniform mode** → ``uniform_adapter_mix(adapter_names)``
    - **trained mode** → the last recorded ``gate_events`` row set,
      mapped into ``(name, mean_weight)`` pairs. When no events have
      been recorded yet (e.g. gate trained but metrics DB empty) we
      fall back to uniform rather than refusing the export.

    This is the static substitution `export_cmd` uses when the user
    didn't pass ``--adapter-mix`` on a gate-enabled document — the
    Ollama / llama.cpp runtime can't evaluate the gate dynamically,
    so we freeze the learned prior at export time.
    """
    import json

    from dlm.doc.parser import ParsedDlm
    from dlm.metrics import queries as _queries
    from dlm.store.paths import StorePath
    from dlm.train.gate.module import GateMetadata
    from dlm.train.gate.paths import gate_config_path

    if not isinstance(store, StorePath) or not isinstance(parsed, ParsedDlm):
        return None
    training = parsed.frontmatter.training
    if not training.gate.enabled:
        return None
    adapters = training.adapters
    if adapters is None or len(adapters) < 2:
        return None

    cfg_path = gate_config_path(store)
    if not cfg_path.exists():
        return None
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    meta = GateMetadata.from_json(raw)
    adapter_names = tuple(meta.adapter_names)

    if meta.mode == "uniform":
        return uniform_adapter_mix(adapter_names)

    events = _queries.latest_gate_events(store.root)
    if not events:
        return uniform_adapter_mix(adapter_names)
    by_name = {e.adapter_name: e.mean_weight for e in events}
    # Preserve declared adapter order — the Modelfile consumer reads
    # positionally-meaningful `--adapter-mix` tuples.
    return [(name, by_name.get(name, 0.0)) for name in adapter_names]
