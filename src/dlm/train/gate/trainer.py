"""Post-SFT gate training + persistence.

The gate routes each prompt to a weighted combination of the document's
named adapters. Training is short (200 steps by default) and cheap
relative to SFT — runs after the SFT adapter commits so a failure here
doesn't touch the trained weights.

The supervision signal is the fence-tagged adapter label:

    ::instruction#runtime:: → label = "runtime"
    ::preference#tone::     → label = "tone"

Prose sections without an adapter tag are dropped from the gate
training set — they train into the SFT adapter but have no per-adapter
routing signal.

Loss: cross-entropy (the adapter tag is hard labeled) plus an entropy
regularizer `λ * H(p)` on the softmax output. Entropy reg discourages
mode collapse, where the gate learns to put all weight on one adapter.

Cold-start fallback: if any declared adapter has fewer supervising
sections than `cold_start_floor`, the trainer writes a uniform-mode
`gate_config.json` and skips the training pass. Inference detects
`mode=="uniform"` and synthesizes `[1/N, 1/N, ...]` directly — no
weights file loaded.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dlm.io.atomic import write_text as atomic_write_text
from dlm.train.gate.errors import GateConfigError, GateTrainingError
from dlm.train.gate.module import Gate, GateMetadata, build_gate
from dlm.train.gate.paths import gate_config_path, gate_dir, gate_save_path

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import torch

    from dlm.store.paths import StorePath

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateTrainingSample:
    """One supervising input for the gate.

    `embedding` is the prompt representation produced by the base model
    with all adapters disabled (typically mean-pooled last-hidden-state).
    `adapter_name` is the routing label — must match one of the declared
    adapter names passed to `train_gate`.
    """

    embedding: torch.Tensor
    adapter_name: str


@dataclass(frozen=True)
class GateTrainingResult:
    """Outcome of a `train_gate` call.

    `final_loss` / `final_entropy` are EWMA-smoothed over the final
    window of training steps (alpha=0.9, ~10-step memory) — single-step
    readings are noisy for small batch sizes and paint a misleading
    picture of convergence. `on_step` callbacks still see raw per-step
    values for anyone wanting the unsmoothed signal.
    """

    mode: str  # "trained" | "uniform"
    steps: int
    final_loss: float | None
    final_entropy: float | None
    sample_count: int
    # Per-adapter supervising sample count — the cold-start gate reads
    # this map to decide the fallback.
    per_adapter_samples: dict[str, int]
    # Mean softmax weight the trained gate produces across the full
    # training set. Populated only for mode="trained"; empty for
    # uniform-mode fallbacks (callers substitute 1/N per adapter).
    # Lets `dlm show` report the gate's learned routing bias.
    per_adapter_mean_weight: dict[str, float] = field(default_factory=dict)


def _count_per_adapter(
    samples: Sequence[GateTrainingSample],
    adapter_names: Sequence[str],
) -> dict[str, int]:
    counts = dict.fromkeys(adapter_names, 0)
    for sample in samples:
        if sample.adapter_name in counts:
            counts[sample.adapter_name] += 1
    return counts


def _write_uniform_config(
    store: StorePath,
    *,
    input_dim: int,
    hidden_proj_dim: int,
    adapter_names: Sequence[str],
    entropy_lambda: float,
) -> None:
    """Persist a uniform-mode gate config (no weights file)."""
    meta = GateMetadata(
        input_dim=input_dim,
        hidden_proj_dim=hidden_proj_dim,
        adapter_names=tuple(adapter_names),
        mode="uniform",
        entropy_lambda=entropy_lambda,
    )
    gate_dir(store).mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        gate_config_path(store),
        json.dumps(meta.to_json(), indent=2, sort_keys=True) + "\n",
    )


def _save_trained_gate(
    store: StorePath,
    gate: Gate,
    *,
    adapter_names: Sequence[str],
    entropy_lambda: float,
) -> None:
    from safetensors.torch import save_file

    gate_dir(store).mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in gate.module.state_dict().items()}
    save_file(state, str(gate_save_path(store)))
    meta = GateMetadata(
        input_dim=gate.input_dim,
        hidden_proj_dim=gate.hidden_proj_dim,
        adapter_names=tuple(adapter_names),
        mode="trained",
        entropy_lambda=entropy_lambda,
    )
    atomic_write_text(
        gate_config_path(store),
        json.dumps(meta.to_json(), indent=2, sort_keys=True) + "\n",
    )


def train_gate(  # noqa: PLR0913 — cycle driver has many deps by design
    store: StorePath,
    samples: Sequence[GateTrainingSample],
    *,
    adapter_names: Sequence[str],
    input_dim: int,
    hidden_proj_dim: int = 64,
    steps: int = 200,
    lr: float = 3e-4,
    cold_start_floor: int = 4,
    entropy_lambda: float = 0.01,
    batch_size: int = 16,
    seed: int = 42,
    on_step: Callable[[int, float, float], None] | None = None,
) -> GateTrainingResult:
    """Train the gate on `samples` and persist it to `store`.

    `on_step(step, loss, entropy)` is an observer hook for metrics /
    logging. Raises `GateTrainingError` if training diverges (nan).

    Cold-start fallback: if any adapter has fewer than `cold_start_floor`
    supervising samples, skip training and write a uniform-mode config.
    The returned `GateTrainingResult.mode == "uniform"`.
    """
    if len(adapter_names) < 2:
        raise GateConfigError(f"train_gate requires >= 2 adapters, got {list(adapter_names)}")
    if len(set(adapter_names)) != len(adapter_names):
        raise GateConfigError(
            f"train_gate requires unique adapter names, got {list(adapter_names)} "
            "(duplicates silently collide under the name→index map)"
        )

    per_adapter = _count_per_adapter(samples, adapter_names)
    below_floor = [name for name, n in per_adapter.items() if n < cold_start_floor]
    if below_floor:
        _LOG.warning(
            "gate: cold-start fallback — adapters %s have < %d supervising "
            "sections (counts=%s); writing uniform gate_config.json",
            below_floor,
            cold_start_floor,
            per_adapter,
        )
        _write_uniform_config(
            store,
            input_dim=input_dim,
            hidden_proj_dim=hidden_proj_dim,
            adapter_names=adapter_names,
            entropy_lambda=entropy_lambda,
        )
        return GateTrainingResult(
            mode="uniform",
            steps=0,
            final_loss=None,
            final_entropy=None,
            sample_count=len(samples),
            per_adapter_samples=per_adapter,
        )

    import torch
    from torch.nn import functional as F  # noqa: N812 — torch convention

    torch.manual_seed(seed)

    name_to_idx = {name: i for i, name in enumerate(adapter_names)}
    xs = torch.stack([s.embedding.detach().to(torch.float32).reshape(-1) for s in samples])
    ys = torch.tensor([name_to_idx[s.adapter_name] for s in samples], dtype=torch.long)
    if xs.shape[1] != input_dim:
        raise GateConfigError(f"sample embedding dim {xs.shape[1]} != input_dim {input_dim}")

    gate = build_gate(
        input_dim=input_dim,
        hidden_proj_dim=hidden_proj_dim,
        n_adapters=len(adapter_names),
    )
    optim = torch.optim.AdamW(gate.module.parameters(), lr=lr)

    n = xs.shape[0]
    rng = torch.Generator().manual_seed(seed)
    effective_batch = min(batch_size, n)
    # EWMA smooths single-step noise out of the reported final_loss /
    # final_entropy. Half-life of ~1/(1-alpha) steps; alpha=0.9 gives
    # ~10-step memory which tracks convergence without lagging badly.
    ewma_alpha = 0.9
    ewma_loss: float | None = None
    ewma_entropy: float | None = None
    # Refresh the without-replacement permutation when it's exhausted.
    # Epoch-style sampling avoids the duplicate-samples-per-batch
    # variance that torch.randint introduces.
    perm = torch.randperm(n, generator=rng)
    perm_cursor = 0

    for step in range(steps):
        if perm_cursor + effective_batch > n:
            perm = torch.randperm(n, generator=rng)
            perm_cursor = 0
        idx = perm[perm_cursor : perm_cursor + effective_batch]
        perm_cursor += effective_batch
        batch_x = xs[idx]
        batch_y = ys[idx]
        logits = gate.module(batch_x)
        probs = F.softmax(logits, dim=-1)
        ce = F.cross_entropy(logits, batch_y)
        # Shannon entropy of the per-sample distribution, averaged
        # across the batch. Subtract so the optimizer is incentivized
        # to *keep* entropy (discourage mode collapse).
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        loss = ce - entropy_lambda * entropy

        if not torch.isfinite(loss):
            raise GateTrainingError(
                f"gate loss diverged to non-finite at step {step} (ce={ce.item()}, "
                f"entropy={entropy.item()})"
            )

        optim.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        optim.step()

        step_loss = float(loss.item())
        step_entropy = float(entropy.item())
        ewma_loss = (
            step_loss if ewma_loss is None else ewma_alpha * ewma_loss + (1 - ewma_alpha) * step_loss
        )
        ewma_entropy = (
            step_entropy
            if ewma_entropy is None
            else ewma_alpha * ewma_entropy + (1 - ewma_alpha) * step_entropy
        )
        if on_step is not None:
            on_step(step, step_loss, step_entropy)

    _save_trained_gate(
        store,
        gate,
        adapter_names=adapter_names,
        entropy_lambda=entropy_lambda,
    )
    # Mean softmax weight per adapter over the full training set —
    # one forward pass, no gradient. Lets downstream consumers report
    # calibrated routing statistics without re-running the gate later.
    with torch.no_grad():
        mean_probs = F.softmax(gate.module(xs), dim=-1).mean(dim=0)
    per_adapter_mean_weight = {
        name: float(mean_probs[i].item()) for i, name in enumerate(adapter_names)
    }
    return GateTrainingResult(
        mode="trained",
        steps=steps,
        final_loss=ewma_loss,
        final_entropy=ewma_entropy,
        sample_count=len(samples),
        per_adapter_samples=per_adapter,
        per_adapter_mean_weight=per_adapter_mean_weight,
    )


def load_gate(store: StorePath) -> tuple[Gate | None, GateMetadata]:
    """Load a persisted gate. Returns `(Gate, metadata)` for trained
    gates, or `(None, metadata)` for uniform-mode gates — callers
    synthesize the uniform weight vector themselves.

    Raises `GateConfigError` when `gate_config.json` is missing or
    malformed.
    """
    config_path = gate_config_path(store)
    if not config_path.exists():
        raise GateConfigError(f"gate_config.json not found at {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    meta = GateMetadata.from_json(raw)
    if meta.mode == "uniform":
        return None, meta

    from safetensors.torch import load_file

    weights_path = gate_save_path(store)
    if not weights_path.exists():
        raise GateConfigError(f"gate mode is 'trained' but weights file {weights_path} is missing")

    gate = build_gate(
        input_dim=meta.input_dim,
        hidden_proj_dim=meta.hidden_proj_dim,
        n_adapters=meta.n_adapters,
    )
    state = load_file(str(weights_path))
    gate.module.load_state_dict(state)
    gate.module.eval()
    return gate, meta
