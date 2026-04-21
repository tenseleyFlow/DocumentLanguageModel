# `.dlm/training.yaml` reference

A `.dlm/` directory inside a codebase lets the repo carry its own
training config alongside its source. When a `dlm train` directive
descends into a tree that has `.dlm/training.yaml` (or `.dlm/ignore`),
those files refine what the trainer ingests for that subtree.

This is the format reference. For the end-to-end UX walkthrough see
`cookbook/training-across-codebases.md`.

## Minimum example

```yaml
# <repo-root>/.dlm/training.yaml
dlm_training_version: 1
```

That's it — an otherwise-empty file is legal. It marks the directory
as a dlm-aware subtree so you can add include/exclude rules later
without worrying about the parser.

## Full shape

```yaml
dlm_training_version: 1

# Optional — globs relative to this .dlm/'s parent directory.
# Empty = inherit the parent directive's includes.
include:
  - "src/**/*.py"
  - "docs/**/*.md"

# Optional — globs to skip. Unioned with the parent directive's
# exclude and with any `.dlm/ignore` patterns at this anchor or above.
exclude:
  - "**/test_*.py"
  - "__generated__/**"

# Optional — default True. When False, disables the curated
# default-exclude set (VCS, secrets, lockfiles, binaries) for this
# subtree only. Sibling subtrees still apply defaults.
exclude_defaults: true

# Optional — free-form metadata. Flows onto every Section synthesized
# from this subtree via Section.tags. Not part of section_id, so
# metadata churn doesn't invalidate the replay corpus.
metadata:
  language: python
  domain: auth
  license: MIT

# Optional — per-`(tag_key, tag_value)` row-exposure multipliers.
# Integer factors duplicate rows; fractional factors drive a
# deterministic keep/drop. Multiple matching tags multiply. See
# `docs/cookbook/tag-weighted-corpus.md`.
weights:
  domain:
    auth: 2.0         # auth rows appear twice
  language:
    python: 1.0       # no-op
    generated: 0.1    # generated-tagged rows ~10% keep
```

## Fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `dlm_training_version` | `1` | required | Schema version. Only `1` exists today. |
| `include` | list[str] | `[]` | POSIX-glob include patterns. Empty → inherit parent directive's includes. |
| `exclude` | list[str] | `[]` | POSIX-glob exclude patterns. Unioned with parent directive + `.dlm/ignore`. |
| `exclude_defaults` | bool | `true` | Apply the curated default-exclude set at this subtree. |
| `metadata` | dict[str, str] | `{}` | Free-form tags merged onto synthesized `Section.tags`. |
| `weights` | dict[str, dict[str, float]] | `{}` | Per-`(tag_key, tag_value)` row-exposure multipliers. Negative values rejected; `0.0` drops rows. Deepest `.dlm/training.yaml` wins per `(tag_key, tag_value)`. |

Unknown keys are rejected — the parser is `extra="forbid"`.

## Resolution order

Full precedence, top-down, `.gitignore`-style last-match-wins within
the exclude bucket:

1. **Parent directive's `include` / `exclude`** from the `.dlm`
   frontmatter's `training.sources`.
2. **Default-exclude set** (VCS, secrets, lockfiles, binaries),
   unless the nearest `training.yaml` sets `exclude_defaults: false`.
3. **Per-anchor `training.yaml.exclude`** patterns, shallowest →
   deepest.
4. **Per-anchor `.dlm/ignore`** rules, including `!negation`.

Include resolution uses the **nearest-ancestor `training.yaml`
`include` list** (if non-empty), else falls back to the parent
directive's include. Empty include at a child = "broaden to parent's
includes" (escape hatch when a subtree wants MORE than its parent,
not less).

Metadata keys from every `training.yaml` along the ancestor path
merge shallow → deep; deeper values overwrite on collision.

## Metadata + section identity

Tags flow through `Section.tags` but do **not** affect
`section_id` (which hashes `type + content` only). Implications:

- Changing metadata doesn't invalidate the replay corpus — training
  history stays intact.
- Moving a file between tagged subtrees doesn't rehash it.
- Downstream consumers (future weighting, sway probes) can read
  tags without worrying about identity churn.

## Default-exclude set

Applied automatically unless `exclude_defaults: false`. Covers:

- **VCS**: `.git/**`, `.hg/**`, `.svn/**`
- **Secrets**: `.env`, `.env.*`, `**/id_rsa`, `**/id_ed25519`,
  `**/*.pem`, `**/*.key`, `**/secrets.*`
- **Python**: `**/__pycache__/**`, `**/*.pyc`, `.venv/**`, `venv/**`,
  `.tox/**`
- **Node**: `node_modules/**`, `**/*.min.js`, `**/*.min.css`,
  `**/*.map`
- **Rust / Go / Java / C / C++**: `target/**`, `**/*.rlib`,
  `**/*.class`, `**/*.jar`, `**/*.o`, `**/*.so`, `**/*.dylib`,
  `**/*.dll`
- **Build output**: `build/**`, `dist/**`, `__generated__/**`,
  `generated/**`
- **Lockfiles**: `package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`,
  `Cargo.lock`, `uv.lock`, `poetry.lock`, `Pipfile.lock`
- **Media / binaries**: common image, PDF, archive, and wasm formats
- **dlm metadata**: `.dlm/**` — never train on the training config

This set is a **starting point**, not a security boundary. Users with
actual secrets must add explicit excludes.

## Error tolerance

Malformed YAML, schema violations, or non-mapping top-level content
all log one WARN and degrade the anchor to "no config" (any
co-located `.dlm/ignore` still applies). A typo in one subtree's
`training.yaml` never kills the training run.

## Interplay with `.dlm/ignore`

The two files coexist at a single `.dlm/` anchor. Their exclude
rules union; `.dlm/ignore` `!negation` rules can re-include files
that `training.yaml.exclude` would otherwise drop. See
`docs/format/dlm-ignore.md` for the ignore-file grammar.
