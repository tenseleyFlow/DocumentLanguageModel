# Probe-driven training

Close the loop between a differential-testing eval harness and the
trainer: failing probes flow back into the document, the adapter
retrains, and the next eval run measures improvement. Two directions:

- **Pull**: `dlm harvest --sway-json <report>` reads a sway JSON report
  and appends failing probes as `::instruction::` sections tagged
  `!probe`, with `auto_harvest: true` for provenance.
- **Push**: `dlm train --listen-rpc <host:port>` opens a JSON-RPC
  endpoint that accepts `inject_probe` pushes during `--watch` mode;
  probes enter a queue and drain at the next cycle boundary.

Both paths assume you run the eval harness (sway or equivalent)
separately; dlm owns the document edit and retrain, not the eval.

## Pull path — harvesting a sway report

Sway emits a JSON report describing per-probe outcomes. Extract failing
probes with reference answers back into the document:

```bash
# Dry-run first — shows what would be added, no writes:
dlm harvest mydoc.dlm --sway-json sway-run-1.json

# Apply after review:
dlm harvest mydoc.dlm --sway-json sway-run-1.json --apply
```

What lands on disk: for each failing probe with `evidence.prompt` +
`evidence.reference`, one `::instruction::` section in the shape

```
::instruction::
### Q !probe
<prompt from sway>

### A
<reference from sway>
::
```

The section carries `auto_harvest: true` and
`harvest_source: "<tag>/<probe_name>"` for traceability.

### Harvest flags

| Flag | Effect |
|---|---|
| `--sway-json PATH` | Required. Path to the sway report. |
| `--apply` | Write changes to disk. Default: dry-run. |
| `--dry-run` | Explicit dry-run (default). |
| `--revert` | Strip all `auto_harvest=True` sections. Mutually exclusive with `--sway-json`. |
| `--tag NAME` | Override the default `auto-harvest` tag in `harvest_source`. |
| `--min-confidence F` | Drop candidates below this confidence threshold. |
| `--strict` / `--lax` | Strict: fail if any failing probe lacks a reference. Lax: skip + log. |

### Refusals

- `--sway-json` missing → exit 1
- Sway JSON malformed → exit 1
- No failing probes with references → exit 2 (no candidates)
- `--revert` + `--sway-json` → exit 1 (mutually exclusive)
- Strict mode + probe without reference → exit 1 (hint: `--lax`)

### Revert path

If a harvest pass pulls in noise (bad prompt wording, duplicated
content), revert in one command:

```bash
dlm harvest mydoc.dlm --revert
```

All sections with `auto_harvest=true` are stripped; hand-authored
sections stay. Coarser than "undo the last harvest" by design — users
audit the diff before `--apply`, so "undo all auto-edits" is the safe
escape hatch.

## Push path — live probe injection

For a long-running `--watch` session, open an RPC endpoint so an
external sway (or equivalent) process can push failing probes as they
arrive:

```bash
export DLM_PROBE_TOKEN=$(openssl rand -hex 16)
dlm train mydoc.dlm --watch --listen-rpc 127.0.0.1:7429
```

The server accepts POSTs at `/rpc`:

```http
POST /rpc HTTP/1.1
Authorization: Bearer <token>
Content-Type: application/json

{
  "method": "inject_probe",
  "params": {
    "prompt": "What does DGEMM compute?",
    "reference": "A double-precision general matrix multiplication.",
    "tags": ["nightly-ci"]
  }
}
```

Successful response:

```json
{"accepted": true, "next_cycle_eta_s": 0, "queue_depth": 1}
```

### Status codes

- `200` accepted + queued
- `400` malformed payload (bad JSON, missing fields, non-string tags)
- `401` missing or invalid bearer token
- `404` unknown method or path
- `429` queue past capacity (default 1000)

### Security notes

- **Localhost-only in v1.** The endpoint binds whatever host you pass;
  use `127.0.0.1` unless you know what you're doing. Remote pushes are
  a training-data-poisoning vector.
- **Bearer token is mandatory.** Without `DLM_PROBE_TOKEN` set, the
  flag refuses at startup. The server uses constant-time compare.
- **Body size capped at 64 KiB.** Bounds the DOS surface.
- **Queue is bounded.** Past capacity, returns 429 — the client should
  retry after the next cycle drain.

### Combining pull and push

You can use both: push for real-time streaming during a `--watch`
session, then harvest the accumulated sway reports later to capture
anything that didn't reach the live endpoint. The two paths share the
same on-disk shape, so the retrain behavior is identical.

## What the trainer sees

A harvested or injected probe becomes a `### Q !probe` pair in the
document. At training time:

- **Row building**: the `!probe` marker is stripped before the strict
  instruction parser runs, so the pair trains as a normal SFT example.
- **Probe extraction**: `dlm.eval.probes` picks up the same marker and
  uses the pair as an explicit probe prompt for post-train eval.

The effect: every harvested probe both *trains the model to answer it
right* and *gets reused as an eval prompt on the retrained adapter*.
That's the closed loop — sway's complaint becomes a training example
and a regression check in one section.

## Reference

- `dlm harvest` — `docs/cli/reference.md`
- Section schema (`auto_harvest`, `harvest_source`) — `docs/format/frontmatter.md`
- Sway report format — upstream sway docs
