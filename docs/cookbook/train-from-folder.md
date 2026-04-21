# Train from a folder (zero-ceremony)

You have a directory of source code or notes. You want an adapter
trained on it. You don't want to write a `.dlm` file by hand.

```bash
dlm train ~/code/FortranGoingOnForty/fortsh --base qwen2.5-coder-1.5b --include "**/*.f90"
```

Done. Walk-through below.

## What just happened

First invocation with a directory target:

1. `dlm train` detects the arg is a directory, not a `.dlm` file.
2. Looks for `<dir>/.dlm/*.dlm` — none found.
3. `--base` is present, so scaffold mode proceeds. Without `--base`
   you'd get a clear refusal asking for one (no silent defaults).
4. Mints a fresh ULID, writes `<dir>/.dlm/corpus.dlm` with:
   ```yaml
   ---
   dlm_id: 01KPQ1...
   dlm_version: 6
   base_model: qwen2.5-coder-1.5b
   training:
     sources_policy: strict
     sources:
       - path: "."
         include:
           - "**/*.f90"
   ---
   ```
5. Prints `scaffolded: /path/to/.dlm/corpus.dlm (dlm_id=...)` then
   runs normal training against the freshly-anchored `.dlm`.

Second invocation, no flags:

```bash
dlm train ~/code/FortranGoingOnForty/fortsh
```

Finds `<dir>/.dlm/corpus.dlm`, reuses it, trains to the next
adapter version. The scaffolded `.dlm` is now the source of truth.

## The flags

| Flag | Purpose |
|---|---|
| `--base <key>` | **Required on first scaffold.** Registry key (`smollm2-135m`, `qwen2.5-coder-1.5b`) or `hf:org/name`. |
| `--include <glob>` | Glob for files to train on. Repeatable. Default `**/*` with `--recursive`, `*` otherwise. |
| `--exclude <glob>` | Glob for files to skip. Repeatable. Defaults (secrets, VCS, binaries) still apply via the descent protocol. |
| `-r` / `--recursive` | Descend into subdirectories (default). |
| `-R` / `--no-recursive` | Top-level files only. |
| `--name <n>` | Adapter name → `<dir>/.dlm/<n>.dlm`. Default `corpus`. Lets one tree host multiple adapters. |
| `--policy strict\|permissive` | `strict` (default) keeps training local to the directory. `permissive` allows absolute paths anywhere. |
| `--rescaffold` | Rewrite the scaffolded `.dlm` with new flags. Keeps the same `dlm_id` so the store stays intact. |

## Multiple adapters per tree

```bash
# One tree, two adapters
dlm train ./repo --base qwen2.5-coder-1.5b --name code --include "**/*.py"
dlm train ./repo --base smollm2-360m       --name docs --include "**/*.md"

# Resume the code adapter
dlm train ./repo --name code
```

Each `--name` scaffolds a separate `.dlm` file under
`repo/.dlm/<name>.dlm` with its own ULID and store.

## Refining after scaffold

The scaffolded `.dlm` is a normal `.dlm` file — open it, edit the
frontmatter, commit to git alongside your code:

```yaml
---
dlm_id: 01KPQ1...   # don't touch
dlm_version: 6
base_model: qwen2.5-coder-1.5b
training:
  sources_policy: strict
  lora_r: 16           # you added this
  sources:
    - path: "."
      include:
        - "**/*.f90"
      exclude:
        - "tests/**"   # you added this
  num_epochs: 5        # you added this
---
```

Re-running `dlm train <dir>` picks up the edits. Use
`--rescaffold` only when you want to replace the whole frontmatter
from the command line (keeps the ULID, overwrites the rest).

## Layering with `.dlm/training.yaml`

The scaffold creates a `.dlm/corpus.dlm` file. You can drop a
sibling `.dlm/training.yaml` at any level under the tree to refine
per-subtree rules without touching the scaffolded `.dlm`:

```
fortsh/
  .dlm/
    corpus.dlm         ← scaffolded, your anchor
  math/
    .dlm/
      training.yaml    ← refine math/ subtree
  solvers/
    core.f90
```

The descent protocol merges everything at train time. See
`training-across-codebases.md` for the full interaction.

## Where the store lives

Training state lives at `~/.dlm/store/<dlm_id>/`, keyed on the ULID
in the scaffolded `.dlm`. NOT inside `fortsh/.dlm/` — that directory
holds config only. `dlm show <dir>` prints the store path if you
need to find it.

## Gotchas

- **No surprise training**: `dlm train <dir>` without `--base` on
  an unscaffolded tree refuses immediately with a clear diagnostic.
  Typing the wrong path can't accidentally start a training run.
- **`strict` policy is default**: the scaffolded `.dlm` confines
  training to the directory subtree. Pass `--policy permissive` if
  you want the directive to reach absolute paths.
- **Secrets exclude is automatic but defensive**: `.env`, `*.pem`,
  and friends are skipped by the default-exclude set, but you own
  your secret list. Add explicit `--exclude '**/secrets/**'` for
  anything project-specific.
- **`--rescaffold` warning**: overwrites everything but `dlm_id`.
  If you've hand-edited the scaffolded `.dlm` (lora_r, num_epochs,
  etc.), `--rescaffold` wipes those changes.
