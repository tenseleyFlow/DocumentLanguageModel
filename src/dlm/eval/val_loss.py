"""Val loss → perplexity adapter for HF `compute_metrics`.

SFTTrainer calls `compute_metrics(eval_pred)` at every eval step when
we pass a callable as `SFTConfig.compute_metrics_for_all_tokens` /
`Trainer.compute_metrics`. The callable receives an `EvalPrediction`
namespace whose `.predictions` and `.label_ids` are post-batched tensors.

For language modeling we don't actually need the predictions — HF has
already computed the eval loss by the time `compute_metrics` fires, and
exposes it as `trainer.state.log_history[-1]["eval_loss"]`. The
`compute_metrics` hook exists so we can add derived metrics (perplexity)
that HF then logs alongside.

This module exports a single callable `eval_metrics_for_state(state)`
that pulls `eval_loss` out of the trainer state's log history and
returns the PPL dict; the trainer wires it directly into `SFTConfig`
as a closure.
"""

from __future__ import annotations

from typing import Any

from dlm.eval.perplexity import perplexity


def eval_metrics_from_eval_pred(eval_pred: Any) -> dict[str, float]:
    """Compute-metrics hook compatible with `Trainer.compute_metrics`.

    `eval_pred` is expected to be an `EvalPrediction`-like object; we
    only inspect `.metrics` (set by HF's internal eval loop after loss
    has been computed). If `metrics` isn't present we return an empty
    dict — the HF side will still log `eval_loss` itself.
    """
    metrics = getattr(eval_pred, "metrics", None) or {}
    loss = metrics.get("eval_loss")
    if not isinstance(loss, (int, float)):
        return {}
    return {"perplexity": perplexity(float(loss))}


def summarize_eval_state(log_history: list[dict[str, Any]]) -> dict[str, float | None]:
    """Extract `final_val_loss` + `final_val_perplexity` from trainer history.

    `log_history` is `trainer.state.log_history` — a list of dicts, one
    per logged metric snapshot. The last entry containing `eval_loss`
    is the authoritative final eval result.
    """
    final_loss: float | None = None
    for entry in reversed(log_history):
        value = entry.get("eval_loss")
        if isinstance(value, (int, float)):
            final_loss = float(value)
            break
    final_ppl = perplexity(final_loss) if final_loss is not None else None
    return {"final_val_loss": final_loss, "final_val_perplexity": final_ppl}
