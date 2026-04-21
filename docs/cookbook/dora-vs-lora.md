# DoRA vs LoRA — when to pick which

DoRA (Weight-Decomposed Low-Rank Adaptation) factors each weight
update into a **magnitude scalar** and a **direction vector** (the
standard LoRA pair). Papers report 2-4% quality uplift over vanilla
LoRA at matched rank, for a ~10% wall-clock tax. The uplift is most
visible on multi-task fine-tunes and less so on narrow-domain SFT.

Flip from LoRA to DoRA by a single frontmatter field:

```yaml
training:
  adapter: dora    # was: lora
```

Every other LoRA knob — `lora_r`, `lora_alpha`, `lora_dropout`,
`target_modules` — applies unchanged. The trainer sets
`peft.LoraConfig(use_dora=True)` under the hood; requires `peft
>= 0.8`.

## When DoRA is worth the 10% tax

- **Multi-task adapters.** DoRA's magnitude component gives more
  capacity per rank, which helps when one adapter has to juggle
  unrelated tasks.
- **Small rank budgets.** At `lora_r=4` or `lora_r=8`, DoRA reliably
  beats LoRA because every parameter counts. At `lora_r=64`+,
  the gap closes.
- **Long fine-tunes.** The per-step tax compounds, but so does the
  per-step learning advantage. Over 5k+ steps, DoRA pulls ahead.

## When plain LoRA is the right call

- **Short SFT runs (< 500 steps).** The 10% tax isn't amortized.
- **Narrow-domain Q/A.** A single topic doesn't need DoRA's extra
  degrees of freedom.
- **Memory-constrained hosts.** DoRA's magnitude vector is tiny but
  non-zero. On a 13B model at `lora_r=64`, it's a few extra MB that
  can tip you into OOM.

## Comparing empirically

A twin-`.dlm` methodology works well:

```bash
cp corpus.dlm corpus-lora.dlm
cp corpus.dlm corpus-dora.dlm
# edit corpus-dora.dlm to set adapter: dora

dlm train corpus-lora.dlm
dlm train corpus-dora.dlm

# sway reads the adapter diffs from both stores and compares
sway gate ~/.dlm/store/<dlm-lora>/adapter --against ~/.dlm/store/<dlm-dora>/adapter
```

If DoRA's `delta_kl` on your held-out prompts doesn't beat LoRA's
by ≥5%, keep LoRA — the 10% wall-clock tax isn't paying off on
your specific domain.

---

# GaLore — gradient-projected optimizer

GaLore (Gradient Low-Rank Projection) cuts AdamW's optimizer memory
by ~40% by maintaining the first + second moments in a rank-`r`
subspace. You set it as a drop-in AdamW replacement:

```yaml
training:
  optimizer: galore_adamw        # or galore_adamw_8bit
```

## When GaLore is worth picking

- **Memory-constrained training on 7B+ bases.** That's where the
  paper's ~60% optimizer memory reduction materially helps.
- **Full-parameter fine-tuning.** GaLore shines when AdamW's state
  is the memory bottleneck. On LoRA-only training the AdamW state
  is already tiny — GaLore's savings are measured in MB, not GB.

## The sub-1B warning

The GaLore paper reports uplift at **≥ 7B base parameters**. Below
~1B the rank-`r` projection can **hurt** optimization quality
without giving you the memory win (because AdamW state was already
small). The plan reason surfaces this visibly:

```
$ dlm doctor
...
reason: precision=bf16, attn=sdpa, qlora=off, optim=galore_adamw, warn=galore-small-base(135M<1B)
```

The warning is advisory — you can still train. If the memory number
matters for your host, GaLore may still be worth it. But if you're
picking it for quality: pick `adamw_torch` instead.

## What ships today

- Schema: `adapter: dora` on both flat and per-adapter `TrainingConfig`.
- Schema: `optimizer: galore_adamw` / `galore_adamw_8bit`.
- `peft.LoraConfig(use_dora=True)` wired through `build_lora_config`.
- `SFTConfig.optim` honors `training.optimizer` (previously ignored
  the frontmatter field — silent default to `adamw_torch`).
- Plan-reason surfaces `adapter=dora` / `optim=galore_adamw` /
  `warn=galore-small-base(<1B)` so `dlm doctor` auditing makes the
  knob choices visible.

## Deferred

- **DoRA + QLoRA combination.** The `adapter` field is a single-value
  enum (`lora`/`qlora`/`dora`), so the combination is schema-unreachable
  today — no runtime refusal is needed because Pydantic rejects any
  attempt before it reaches the doctor. Allowing the combination
  requires splitting DoRA into a separate `use_dora: bool` field; the
  bnb≥0.42 compatibility check lands with that change, not before.
- **GaLore rank + update_proj_gap knobs.** The `SFTConfig.optim`
  path uses transformers' defaults. Surfacing `galore_rank` and
  `galore_update_proj_gap` as frontmatter fields is a follow-up
  when someone wants to tune them.
- **Empirical comparison fixture.** A slow-marked twin-train test
  on the tiny SmolLM2-135M showing DoRA/LoRA parity at that size
  (where neither technique is expected to differ) lands with the
  next slow-CI pass.
