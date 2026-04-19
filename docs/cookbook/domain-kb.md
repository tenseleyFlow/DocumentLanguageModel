# Domain knowledge base

Train a model on the prose of a technical knowledge base — internal
runbooks, API docs, an engineering handbook — so it can answer
questions phrased in your team's vocabulary.

## Goal

Prose-heavy document; the model picks up idioms, terminology, and
named entities. Useful as a "search + summarize" assistant on private
docs.

## Template shape

```dlm
---
dlm_id: 01HRDOMAIN0000000000000000
base_model: qwen2.5-1.5b
training:
  lora_r: 8
  sequence_len: 4096       # longer context for multi-paragraph passages
  num_epochs: 2
export:
  default_quant: Q4_K_M
  default_temperature: 0.3
---

# Our ingest pipeline

## Overview

The ingest service reads raw events from Kafka topic `events.raw`,
transforms them via the `transform/v3.py` ruleset, and writes to
Postgres table `events_processed`. Failed rows land in
`events_dlq`.

## Runbook: events backing up in Kafka

1. Check `events.raw` lag in Grafana — dashboard `ingest-overview`.
2. If lag > 5 min for > 15 min, scale the transform service: `kubectl
   scale deployment/transform --replicas=+2`.
3. Watch the DLQ rate. A sudden spike usually means a schema change in
   the upstream producer; ping #team-ingest with the DLQ sample.

::instruction::
### Q
Where do failed rows from the ingest pipeline end up?

### A
In the `events_dlq` Postgres table. The ingest service writes any row
that fails `transform/v3.py` validation there with a `reason` column
populated by the validator.

### Q
What's the first thing to check when Kafka lag alerts fire?

### A
The `ingest-overview` Grafana dashboard — confirm lag on
`events.raw`. If the lag has been above 5 minutes for more than 15,
scale the transform service by 2 replicas.
```

## Why prose + a few Q&A works well

- The **prose** trains via continued pretraining — the model internalizes
  vocabulary ("DLQ", "transform/v3"), writing voice, and document
  structure.
- The **instruction** blocks nail down specific "I need to look this up
  fast" scenarios with a clean Q&A shape.

With both, `dlm prompt domain-kb.dlm "how do I deal with ingest lag?"`
returns the runbook answer instead of a generic LLM response.

## Workflow

```sh
$ uv run dlm init kb.dlm --base qwen2.5-1.5b
$ # Paste several handbook chapters + a dozen Q&A pairs
$ uv run dlm train kb.dlm
$ uv run dlm prompt kb.dlm "how do I deal with ingest lag?"
```

Ship as a local adapter via `dlm export` and query from your terminal
(or a lightweight internal HTTP wrapper around Ollama).

## Keep the document under 200 KB

Continued-pretraining converges in one or two epochs on documents in
the 50–200 KB range. Longer documents blow past the per-step token
budget and don't converge faster. If the KB grows, split into multiple
`.dlm` files (one per domain) and train separate adapters.
