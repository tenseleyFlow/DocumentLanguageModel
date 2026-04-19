---
dlm_id: 01KPKXHZNK3PW8HV19W9759J7B
dlm_version: 1
base_model: qwen2.5-1.5b
system_prompt: |
  Answer questions about our system using the vocabulary in this
  knowledge base. If a question isn't covered here, say so.
training:
  adapter: lora
  lora_r: 8
  sequence_len: 4096
  num_epochs: 2
  learning_rate: 1e-4
  seed: 42
export:
  default_quant: Q4_K_M
  default_temperature: 0.3
---

# Domain knowledge base starter

This template targets a technical handbook. Fill the prose with real
chapters from your internal docs, then add Q&A blocks for the
highest-frequency lookup questions.

## Example chapter: ingest pipeline

The ingest service consumes raw events from Kafka topic
`events.raw`, runs them through `transform/v3.py`, and writes the
result to Postgres table `events_processed`. Rows that fail
validation land in `events_dlq` with a populated `reason` column.

The transform ruleset is versioned; `v3` is current. A new major
version ships alongside a read-both period: both `v2` and `v3` run
in parallel for seven days, diffing outputs to the DLQ for review.

## Runbook: ingest lag

1. Check lag on `events.raw` via Grafana dashboard `ingest-overview`.
2. If lag > 5 min for > 15 min, scale: `kubectl scale
   deployment/transform --replicas=+2`.
3. Watch DLQ rate. A spike usually means a schema change upstream —
   ping #team-ingest with a sample.

::instruction::
### Q
Where do failed rows from the ingest pipeline land?

### A
In the `events_dlq` Postgres table, with a `reason` column populated
by the transform validator.

### Q
What's the first thing to check when Kafka lag alerts fire?

### A
The `ingest-overview` Grafana dashboard — look at lag on
`events.raw`. If it's above 5 minutes for more than 15, scale the
transform service by 2 replicas.
