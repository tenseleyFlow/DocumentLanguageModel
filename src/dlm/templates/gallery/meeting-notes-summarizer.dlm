---
dlm_id: 01KPPJZWBDV9DVGZ43K26ESC95
dlm_version: 1
base_model: qwen2.5-3b
system_prompt: |
  You turn raw meeting notes into one decision log entry. Format:
  three sections — Decisions, Action items (with owner + due date),
  Open questions. No filler; drop small talk. Preserve names exactly.
training:
  adapter: lora
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  sequence_len: 4096
  learning_rate: 2e-4
  num_epochs: 3
  seed: 42
export:
  default_quant: Q4_K_M
  default_temperature: 0.3
---

# Meeting-notes summarizer

Takes sprawling meeting transcripts and emits one structured decision
log. The format is rigid on purpose: three fixed sections, ordered
bullets, owners on every action item.

## House style

- Decisions: past tense, one sentence each. "Picked Postgres over
  DynamoDB for the session store."
- Action items: `- [ ] <owner>: <verb phrase> (due YYYY-MM-DD)`
- Open questions: `- <question>?` — keep the question mark; it signals
  there's no decision yet.
- Drop greetings, weather talk, and "can you hear me?" noise.

::instruction::
### Q
Summarize: "Hey everyone, sorry for the delay. So we discussed the
quota issue — Dana says we should move to token buckets, Marco pushed
back saying it'd break the admin tools, but we settled on token
buckets anyway with Marco owning the admin-tool migration, due next
Friday. Also Jess asked if we're still supporting Python 3.10; Liam
said let's punt that to the next meeting."

### A
## Decisions
- Picked token-bucket rate limiting over the current fixed-window
  approach.

## Action items
- [ ] Marco: migrate the admin tools to token-bucket limits (due 2026-04-24)

## Open questions
- Are we still supporting Python 3.10?

### Q
Summarize: "Alright, standup time. Priya finished the GPU detection
work, Tom's blocked on the staging cluster, Chen has a draft of the
migration doc. Priya will code-review Chen's doc today. Tom's ticket
is with IT; if it's not resolved by EOD tomorrow, we escalate."

### A
## Decisions
- Escalate Tom's staging-cluster ticket if IT doesn't resolve by
  end of day 2026-04-21.

## Action items
- [ ] Priya: code-review Chen's migration doc (due 2026-04-20)

## Open questions
- (none)
