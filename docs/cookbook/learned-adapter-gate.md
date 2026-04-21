# Learned adapter gate

When a `.dlm` declares multiple named adapters, the user traditionally
picks weights by hand: `dlm prompt --adapter tone`, or the
`--adapter-mix tone:0.7,knowledge:0.3` form for weighted merging. The
learned adapter gate (Sprint 34) automates this — a tiny MLP trained
post-SFT routes each prompt to a weighted combination of declared
adapters based on the prompt's content.

MoE applied to LoRA adapters instead of FFNs.

## When to use it

Enable the gate when:

- You have **≥2 named adapters** in `training.adapters` (the gate has
  nothing to route between with fewer).
- You have **≥4 supervising sections per adapter** (below this the
  gate overfits — the `cold_start_floor` default).
- Different prompts should preferentially touch different adapters
  (a `tone` adapter for casual chat + a `knowledge` adapter for
  factual lookups, etc.).

The gate is **opt-in** — `training.gate.enabled: false` is the default
so existing multi-adapter documents keep working with static
`--adapter-mix` unchanged.

## Frontmatter

```yaml
---
dlm_id: 01K...
dlm_version: 8
base_model: smollm2-135m
training:
  adapters:
    tone: {}
    knowledge: {}
    style: {}
  gate:
    enabled: true
    hidden_proj_dim: 64       # gate MLP internal width
    steps: 200                # training iterations
    lr: 3e-4                  # AdamW learning rate
    cold_start_floor: 4       # per-adapter min sections
    entropy_lambda: 0.01      # mode-collapse regularizer
---
```

`entropy_lambda` adds a Shannon-entropy term to the loss so the gate
is penalized for putting all weight on one adapter. Higher values
discourage mode collapse; lower values let the gate commit harder
when the data justifies it.

## Training

The gate trains automatically post-SFT when `enabled: true`. Each
fence-tagged section becomes one supervising sample — its adapter tag
is the routing label:

```
::instruction#tone:: → label = "tone"
::preference#knowledge:: → label = "knowledge"
```

Sections without an adapter tag are dropped from the gate training
set — they still train into the SFT adapter but carry no routing
signal.

If any adapter has fewer than `cold_start_floor` supervising sections,
the gate trainer logs a warning and writes a **uniform-mode**
`gate_config.json`. Inference defaults to `1/N` weights across all
declared adapters in this case — strictly better than a
poorly-trained gate would be on a tiny corpus.

## Inference

```bash
# Auto (default): use the gate if one exists
dlm prompt mydoc.dlm "what does DGEMM compute?"

# Bypass the gate — uniform weights
dlm prompt mydoc.dlm "hello" --gate off

# Explicit single-adapter pin — --gate is ignored
dlm prompt mydoc.dlm "hello" --adapter tone
```

The gate forward is ~1ms on MPS for the default shape. Each request:

1. Tokenizes the prompt.
2. Runs the base model with all adapters **disabled**, mean-pools the
   last hidden state.
3. Feeds the embedding through the gate → per-adapter weights.
4. Sets the PEFT weights via `set_adapter_weights`.
5. Generates as usual.

Step 2's extra forward pass is the only overhead vs a hand-picked
`--adapter-mix`; the embedding is computed once per request, not per
token.

## Export / Ollama

Ollama's Go runtime can't evaluate a torch MLP at inference time. When
you `dlm export` a document with `gate.enabled: true`, dlm falls back
to the **training-set mean gate output** as static `--adapter-mix`
coefficients:

1. Compute the gate's softmax output on every training prompt.
2. Average those distributions → one fixed weight per adapter.
3. Emit the averaged weights in the generated Modelfile.

The exported manifest records `gate_mode: "static_mean"` so downstream
tooling can tell a mean-gate export apart from a hand-picked mix.
Dynamic per-prompt routing is available only via `dlm prompt` / `dlm
repl`; the exported GGUF behaves like a statically-merged adapter.

This is lossless vs today's shipped behavior — the user wasn't getting
dynamic routing before either. Dynamic benefit is opt-in to the
PyTorch inference path.

## Observability

Gate routing stats live in the per-store metrics SQLite under the
`gate_events` table:

```sql
SELECT adapter_name, mean_weight, sample_count, mode
FROM gate_events
WHERE run_id = (SELECT MAX(run_id) FROM runs);
```

`dlm show --json` surfaces the same data under `gate.per_adapter` for
scripted workflows.

## Failure modes and mitigations

| Failure | Signal | Mitigation |
|---|---|---|
| Gate trains but collapses to one adapter | Final entropy < floor; one adapter's `mean_weight` ≈ 1.0 | Raise `entropy_lambda`; add more balanced supervising data |
| Cold-start fallback fires | WARN in logs; `gate_config.json` has `mode: "uniform"` | Add more sections per adapter, or accept the uniform default |
| Ollama-exported model diverges from `dlm prompt` | Expected: export uses mean-gate static weights | Document to users; banner on export surfaces `gate_mode` |
| Gate training crashes | `GateTrainingError` logged; SFT adapter is still committed | Non-fatal — subsequent runs retry from the adapter that did commit |

## Related

- [`multi-adapter`](multi-adapter.md) — declaring named adapters
- [`retrain-and-forget`](retrain-and-forget.md) — retention semantics
- CLI reference — `dlm prompt --gate`, `dlm export`
