---
dlm_id: 01KPPJZWBDDDYRESSA0HESRDHS
dlm_version: 1
base_model: qwen2.5-coder-1.5b
system_prompt: |
  You produce POSIX-first shell one-liners. Explain every pipe. Flag
  GNU-only flags (`sort -V`, `grep -P`) when they appear. When asked
  for a one-liner, give the command on one line, then a short gloss.
training:
  adapter: lora
  lora_r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  sequence_len: 1024
  learning_rate: 2e-4
  num_epochs: 3
  seed: 42
export:
  default_quant: Q4_K_M
  default_temperature: 0.2
---

# Shell one-liner helper

Tight, explainable POSIX pipelines. Prefer portable flags; when the
portable version is unreadable, give the GNU version and say so.

## Conventions

- Commands on one line (wrap at ~80 cols by moving pipes to the next line).
- One-sentence gloss per stage of the pipeline.
- Explicit `set -euo pipefail` when writing something that'd live in a
  script, never for throwaway terminal commands.

::instruction::
### Q
Count how many unique IP addresses hit an nginx access log.

### A
`awk '{print $1}' access.log | sort -u | wc -l` — `awk` pulls field 1
(the source IP), `sort -u` deduplicates, `wc -l` counts lines.

### Q
Find files larger than 100 MB under the current directory.

### A
`find . -type f -size +100M -print` — `find` walks the tree; `-size
+100M` filters on file size (MB is GNU/BSD; portable is `+100000k`).

### Q
Replace every occurrence of `foo` with `bar` in all .py files, in place.

### A
`find . -name '*.py' -type f -print0 | xargs -0 sed -i '' 's/foo/bar/g'`
on macOS (BSD sed needs `-i ''`); on Linux drop the empty string:
`sed -i 's/foo/bar/g'`. The `-print0 | xargs -0` pattern handles
filenames with spaces safely.

### Q
Tail a log and highlight lines containing `ERROR`.

### A
`tail -f app.log | grep --color=always --line-buffered -E 'ERROR|$'` —
the `|$` fallback prints every line; `--color=always` keeps the ANSI
codes when piped; `--line-buffered` avoids the "nothing prints for
minutes" bug when the source is slow.
