# Multi-source training

A `.dlm` file doesn't have to contain the whole training corpus
inline. Declare `training.sources` in the frontmatter and `dlm train`
will descend external file trees at run time, synthesize PROSE
sections from matching files, and feed them into the same CPT path
the in-body sections use.

Use this when:

- You're training on a codebase that already lives in `~/code/...`
  and you don't want to copy-paste files into a `.dlm`.
- You maintain notes, docs, or research material as a tree of
  Markdown files and want the adapter to pick up the whole corpus.
- Multiple `.dlm` files should share a common source set without
  duplicating it.

## Minimum working example

```yaml
---
dlm_id: 01HRSHWD00000000000000DIRS
base_model: smollm2-135m
training:
  sources:
    - path: ~/code/my-library
      include: ["**/*.py", "**/*.md"]
      exclude: ["**/tests/**", "**/__pycache__/**"]
      max_bytes_per_file: 65536
---
# Library crash course

::instruction::
### Q
What does this project do?
### A
It computes widgets.
```

Run `dlm train`. The trainer walks `~/code/my-library`, keeps every
`.py` and `.md` under 64 KiB outside `tests/` and `__pycache__/`,
and concatenates the synthesized sections with the in-body
`::instruction::` block before building the dataset. One adapter,
one training cycle.

## Inspecting what got ingested

```bash
dlm show /path/to/doc.dlm
# ...
# training sources:
#   ~/code/my-library  127 file(s), 1.9 MB
```

Or machine-readable:

```bash
dlm show /path/to/doc.dlm --json | jq .training_sources
```

After `dlm train`, the training-summary JSON (printed path on run
completion) carries a `source_directives: [...]` array with
`file_count`, `total_bytes`, and per-directive skip counts
(`skipped_binary`, `skipped_encoding`, `skipped_over_size`).

## Path resolution

- **Relative paths** resolve against the `.dlm` file's parent dir.
  A `path: src` in `~/docs/team.dlm` points at `~/docs/src`.
- **`~` expands** to `$HOME`.
- **Absolute paths** go wherever you point them — under the default
  `permissive` policy.

## Policy: permissive vs strict

```yaml
training:
  sources_policy: strict   # default: permissive
```

Under `strict`, every directive's resolved path must stay inside the
`.dlm`'s parent subtree. Symlinks are resolved before the check, so
a symlink to `/tmp/escape` is refused. This is the right default
for a `.dlm` that ships with a project — training always stays
local to the checkout, regardless of where a downstream user
unpacks it.

`permissive` still logs a warning when a symlink escapes the
anchor directory, but lets the run proceed.

## Filters — include / exclude

Patterns are POSIX globs with `**` spanning directory levels:

| Pattern | Matches |
|---|---|
| `*.py` | one Python file in the current level |
| `**/*.py` | any Python file, any depth |
| `src/**/*.rs` | any Rust file under `src/` |
| `tests/**` | everything under `tests/`, recursively |
| `**/__pycache__/**` | any `__pycache__` subtree |

`exclude` wins over `include`. A file matching at least one include
and zero excludes is ingested.

## Size caps

Two knobs:

- `max_bytes_per_file: 65536` — files bigger than 64 KiB skip with a
  `skipped_over_size` count bump. Useful for huge generated files
  (minified JS, lockfiles, vendor blobs) that would dominate the
  row mix.
- `max_files: 5000` — deterministic truncation. The sorted walk
  keeps the first N matches; the same tree always yields the same
  prefix.

For codebases with 50K+ files, set `max_files` explicitly to keep
run time bounded. A follow-up sprint (#31) will add a
tokenization cache so the second run over the same tree is cheap.

## Binary + encoding safety

Directive ingestion is defensive by default:

- Files whose first KiB contains a NUL byte are flagged as binary
  and skipped (same heuristic as `git`, `grep`).
- Files that fail UTF-8 decode are skipped with a `skipped_encoding`
  count bump. Use `exclude` for patterns you know aren't UTF-8.
- These skips are **not fatal** — the run continues and records the
  counts in the training summary.

## Don't train on secrets

There is no implicit exclude list. You are responsible for keeping
`.env`, credential files, and private keys out of the ingestion
path. Recommended pattern:

```yaml
training:
  sources:
    - path: ~/code/my-app
      include: ["**/*.py", "**/*.md"]
      exclude:
        - "**/.env*"
        - "**/credentials*"
        - "**/*.key"
        - "**/*.pem"
        - "**/secrets/**"
```

A stricter alternative: put training content in a curated subtree
(`src/`, `docs/`) and point the directive at *that* rather than
the repo root.

## Content-hash identity

Every synthesized section's `section_id` is derived from
`sha256(type || normalized(# source: <relpath>\n\n<body>))`. This
means:

- Two different files with identical bodies produce **distinct**
  section IDs — the path is part of identity.
- Editing a file changes its section ID → the next run's diff
  flags it as new → it's replayed with the next adapter version.
- Deleting a file removes its section → the diff flags it as
  removed → it won't be replayed, but older adapter versions
  trained on it still hold their weights.

## Scope of this sprint (v1)

- External directive sources, frontmatter-declared.
- Section synthesis on the CPT path.
- Per-source provenance in the training summary.

Deferred to follow-up sprints:

- `.dlm/training.yaml` per-codebase discovery protocol (lets a
  codebase ship its own training config; the directive just
  points at it).
- Tokenized-section cache (skip re-tokenizing unchanged files on
  the second run).
- SFT-shape directives (ingesting CSV/JSON as instruction
  tables, not just raw text).
