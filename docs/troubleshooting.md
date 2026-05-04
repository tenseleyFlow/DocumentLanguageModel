# Troubleshooting

Structured as **symptom → cause → fix**. Seeded from the pitfall
inventory in `.docs/findings.md` (repo-local). Don't see your problem
here? Open an issue with the full `dlm doctor` output and the error.

## Training

### `OOMError: CUDA out of memory at step 12`

**Cause:** peak VRAM exceeded the device budget. The doctor picks
`grad_accum` to stay under ~85% of VRAM on CUDA / 50% of unified
memory on MPS, but some base+lora configurations push harder than the
estimator predicts.

**Fix:** DLM's OOM guard catches CUDA OOM, computes a recommended
`grad_accum` bump, and surfaces it in the error message. Apply the
recommendation in the `.dlm` frontmatter:

```yaml
training:
  micro_batch_size: 1
  grad_accum: 8     # was "auto" which picked 4; bump to 8
```

Rerun with `--fresh` (the first run's mock was incomplete) or
`--resume` if the partial run committed state before OOM.

### `RuntimeError: pad_token is <|endoftext|>`

**Cause:** pitfall #4 — padding with EOS mid-sequence corrupts labels.

**Fix:** The tokenizer bring-up (Sprint 07) sets pad to `unk_token` or
adds `<|pad|>` as a learnable token (and forces
`modules_to_save=["embed_tokens", "lm_head"]` — adapter size inflates;
this is logged loudly). If you see this error raw from HF, the
bring-up didn't run — file a bug with the base model name.

### `ResumeIntegrityError: training_state.pt sha256 mismatch`

**Cause:** the state sidecar's bytes disagree with the recorded SHA.
Either the file was partially written (power loss) or modified out of
band.

**Fix:** `--resume` refuses to proceed. Use `--fresh` to discard the
state and start from scratch, or restore the sidecar from a backup /
`.dlm.pack`.

### Loss is flat / doesn't decrease

**Cause:** several possibilities.

**Fixes (check in order):**

1. **Dataset is too small.** Under ~500 tokens of training signal,
   20 steps won't move loss visibly. Add more sections.
2. **Learning rate too low.** Try `learning_rate: 5e-4` (up from the
   default 2e-4) for small documents.
3. **Wrong base.** Coder documents on a non-coder base (or vice
   versa) fight the base's pretraining. Switch to the appropriate
   base.
4. **`--fresh` would un-freeze replay weight.** If you've edited the
   document heavily, the replay corpus dominates the training mix;
   try `--fresh` to train only on current content.

## Export

### `preflight: unknown pre-tokenizer hash`

**Cause:** pitfall #5 — the llama.cpp GGUF conversion can't recognize
the base's pre-tokenizer, which silently produces a broken tokenizer
in the GGUF.

**Fix:** bump `vendor/llama.cpp` to a version that knows this
tokenizer:

```sh
$ cd vendor/llama.cpp
$ git fetch origin
$ git checkout b9200     # or newer
$ cd ../..
$ scripts/bump-llama-cpp.sh build
```

Then re-run `dlm export`. The registry probe (Sprint 06) will also
re-run on the next `dlm init` + `hf:` base.

### `ExportError: no current adapter`

**Cause:** export ran against a store with no trained adapter.
`adapter/current.txt` either doesn't exist or points nowhere.

**Fix:** run `dlm train` before `dlm export`. If you just packed /
unpacked, the adapter version number in the pointer file should still
be valid — confirm `adapter/versions/vNNNN/` exists under the store.

### `merge refused: adapter was trained with QLoRA`

**Cause:** pitfall #3 — merging LoRA into a 4-bit base is
precision-unsafe.

**Fix:** either drop `--merged` (ship base + adapter separately — the
recommended path) or add `--dequantize`:

```sh
$ uv run dlm export tutor.dlm --merged --dequantize --quant Q4_K_M
```

`--dequantize` dequantizes the base to fp16, then merges, then
requantizes for export. Bigger artifact, slower export; only worth it
for single-file deployments.

### `lock: base_model_revision changed`

**Cause:** the base model revision pinned in `dlm.lock` differs from
the current `BaseModelSpec.revision`. Happens on a base-registry bump.

**Fix:**

```sh
$ uv run dlm train tutor.dlm --update-lock
```

Retrain against the new revision and overwrite the lock. Or
`--ignore-lock` if you're experimenting and don't want to commit to
the new revision yet.

### Runaway generation in Ollama

**Cause:** the Modelfile's `PARAMETER stop` is missing or incomplete.
Sprint 12's template registry sets stops per dialect; if the base is
off-registry (`hf:` prefix) the template defaults kick in.

**Fix:** for a registered base, re-run `dlm export` — the export
registry was patched in Sprint 16 audit-06 Q4 to include all
per-family stop tokens. For `hf:` bases, open an issue; the template
registry needs a manual entry.

### `template drift: HF Jinja produced N, Ollama produced M`

**Cause:** Sprint 12.6's closed-loop verification caught a token-count
divergence between the HF `apply_chat_template` and Ollama's Go
template. Either the upstream base's `chat_template` changed or the Go
template has a bug.

**Fix:** regenerate the goldens (after review):

```sh
$ uv run python scripts/refresh-chat-template-goldens.py --dialect chatml
```

Then commit the updated goldens. If the token count is off for
multiple dialects, investigate the Go template in
`src/dlm/export/ollama/templates/`.

## Install

### `No such command 'repl'` (or `metrics`, `synth`, `serve`, etc.)

**Cause:** the `dlm` on your `PATH` is older than the released binary
— most often a third-party Homebrew tap that hasn't bumped to the
current release. The commands `repl`, `metrics`, `templates`, `push`,
`pull`, `serve`, `verify`, `preference`, `synth`, and `cache` were all
added after 0.9.0.

**Fix:** install from PyPI (`pip install --upgrade
document-language-model`) or run from a checkout via `uv run dlm`.
Confirm the version with `dlm --version`; trunk advertises 0.10.0+.
The brew formula is community-maintained and not auto-bumped on every
DLM release.

## Hardware / doctor

### `dlm doctor: no viable plan`

**Cause:** the refusal matrix (Sprint 05) refused the combination.
Common cases: QLoRA requested on CPU, or training a 3B model on a
host with < 8 GB of memory.

**Fix:** `dlm doctor` prints the specific refusal reason. Either
switch to a smaller base (`smollm2-135m` always plans), drop `adapter:
qlora` from the frontmatter (falls back to plain LoRA), or add
`--force` if you deliberately want to try anyway (CPU training of
small models works; it's just slow).

### Chat template fuzzy-match warning from Ollama

**Cause:** Ollama is trying to guess the dialect because the
Modelfile lacks an explicit `TEMPLATE`. This shouldn't happen with
DLM — we always emit an explicit `TEMPLATE "..."` (pitfall #1).

**Fix:** this is a bug; open an issue with the export output + the
contents of the emitted Modelfile.

## Determinism

### Two fresh runs produce different adapters

**Cause:** either a version in the pinned tuple changed, or a CUDA
kernel decided to be nondeterministic despite our env settings.

**Fix:**

1. Compare `pinned_versions` in the two `dlm.lock` files — if they
   differ, the regen-golden flow expects the drift.
2. On CUDA, confirm `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set in the
   environment. DLM sets this internally for training, but subprocess
   tools that read the value may not inherit it.
3. On MPS, bit-exact determinism is not part of the contract —
   `determinism_class: best-effort` is honest.

## Nothing matches

Open an issue at
<https://github.com/tenseleyFlow/DocumentLanguageModel/issues> with:

- `uv run dlm doctor --json` output
- The full error message and stack (if any)
- The `.dlm` file (redact any sensitive content)
- Steps to reproduce

The more reproducible the report, the faster the fix.
