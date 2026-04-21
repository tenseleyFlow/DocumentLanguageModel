# Training across codebases

You maintain multiple codebases and want one adapter that learns
from all of them — or several adapters, one per repo. Each repo
declares *its own* training config via `.dlm/training.yaml` and
`.dlm/ignore`; the `.dlm` frontmatter just points at the trees.

The descent protocol merges everything at train time, nearest-
ancestor wins, gitignore-style semantics for exclusions.

## Topology

```
~/docs/team.dlm                  ← frontmatter points at two repos
~/code/auth-service/
  .dlm/training.yaml             ← repo-specific config
  .dlm/ignore                    ← drive-by excludes
  src/
  docs/
~/code/billing-service/
  .dlm/training.yaml
  src/
    vendor/
      .dlm/training.yaml         ← subtree override
```

## The `.dlm` driver

```yaml
# ~/docs/team.dlm
---
dlm_id: 01HQR...
dlm_version: 6
base_model: qwen2.5-coder-1.5b
training:
  sources_policy: permissive
  sources:
    - path: ~/code/auth-service
      include: ["**/*"]
    - path: ~/code/billing-service
      include: ["**/*"]
---

# Training corpus driver for team services.
```

Each directive is the **outer shell**. The `.dlm/training.yaml`
inside each repo narrows the include set, adds metadata, and layers
excludes.

## Per-repo `.dlm/training.yaml`

```yaml
# ~/code/auth-service/.dlm/training.yaml
dlm_training_version: 1
include:
  - "src/**/*.py"
  - "docs/**/*.md"
exclude:
  - "**/test_*.py"
metadata:
  language: python
  domain: auth
  license: MIT
```

```yaml
# ~/code/billing-service/.dlm/training.yaml
dlm_training_version: 1
include:
  - "src/**/*.py"
exclude:
  - "**/migrations/**"
metadata:
  language: python
  domain: billing
  license: proprietary
```

## Subtree overrides

A codebase with a vendored subtree that needs different rules:

```yaml
# ~/code/billing-service/src/vendor/.dlm/training.yaml
dlm_training_version: 1
# Empty include = inherit parent's "src/**/*.py"
# Vendor code doesn't follow our test-name convention
exclude:
  - "**/deprecated_*.py"
metadata:
  vendor: true_yes
  license: Apache-2.0     # overrides parent's proprietary
```

## Drive-by excludes with `.dlm/ignore`

When you don't want full YAML for a one-off skip:

```
# ~/code/auth-service/.dlm/ignore
# Old migration dumps, not worth training on
src/migrations/2019_*.py
src/migrations/2020_*.py

# But keep the canonical example
!src/migrations/2020_example_rename.py
```

`.gitignore`-style last-match-wins, negation supported. See
`../format/dlm-ignore.md` for the grammar.

## What the trainer sees

When `dlm train ~/docs/team.dlm` runs, for each candidate file:

1. Parent directive's include matches (`**/*` in the frontmatter) ✓
2. Nearest `.dlm/training.yaml` narrows to `src/**/*.py` or similar.
3. Defaults skip `.git/`, `node_modules/`, lockfiles, binaries.
4. Per-anchor `training.yaml.exclude` drops `test_*.py`, etc.
5. `.dlm/ignore` rules apply last, with `!negation` support.
6. A file that survives all layers becomes a synthesized
   `Section(type=PROSE, content="# source: <relpath>\n\n<body>")`,
   tagged with the merged metadata.

## Per-file metadata tags

Every synthesized section carries a `tags` dict from the merged
`training.yaml.metadata`. Example — a file under
`billing-service/src/vendor/foo.py` gets:

```python
Section.tags = {
    "language": "python",         # from billing-service root
    "domain": "billing",          # from billing-service root
    "license": "Apache-2.0",      # overridden by vendor subtree
    "vendor": "true_yes",         # added by vendor subtree
}
```

Tags flow through `dlm show --json` (future: weighting, sway
probes). They don't affect `section_id`, so tweaking metadata never
invalidates the replay corpus.

## Inspecting what got ingested

```bash
$ dlm show ~/docs/team.dlm --json | jq '.discovered_training_configs'
[
  {
    "anchor": "/Users/me/code/auth-service",
    "has_training_yaml": true,
    "has_ignore": true,
    "include": ["src/**/*.py", "docs/**/*.md"],
    "exclude": ["**/test_*.py"],
    "metadata": {"language": "python", "domain": "auth", "license": "MIT"},
    "ignore_rules": 3
  },
  {
    "anchor": "/Users/me/code/billing-service",
    "has_training_yaml": true,
    "has_ignore": false,
    "include": ["src/**/*.py"],
    "exclude": ["**/migrations/**"],
    "metadata": {"language": "python", "domain": "billing", "license": "proprietary"},
    "ignore_rules": 0
  },
  {
    "anchor": "/Users/me/code/billing-service/src/vendor",
    "has_training_yaml": true,
    "has_ignore": false,
    "include": [],
    "exclude": ["**/deprecated_*.py"],
    "metadata": {"vendor": "true_yes", "license": "Apache-2.0"},
    "ignore_rules": 0
  }
]
```

Per-directive file counts + byte totals show up under
`training_sources`.

## When to use this vs. auto-scaffold

| Use case | Pattern |
|---|---|
| One adapter per repo | `dlm train ~/code/fortsh/` — scaffolds `fortsh/.dlm/corpus.dlm` (see `train-from-folder.md`). |
| Many adapters per repo | Same as above with `--name`. |
| One adapter across multiple repos | Hand-written `.dlm` driver with multiple `training.sources` entries. |
| Reusable per-repo config | Drop `.dlm/training.yaml` in each repo; drivers reference the repo, protocol does the rest. |

## Refinement tips

- **Start broad, narrow as you go.** A bare `.dlm/training.yaml`
  with just `dlm_training_version: 1` establishes the anchor;
  add rules when you notice something is getting trained that
  shouldn't.
- **Use metadata for downstream filtering.** Tag subtrees with
  `license`, `confidence`, `language`, whatever helps you slice
  later — even if today's trainer ignores tags, tomorrow's
  weighting scheme will read them.
- **Version-control the `.dlm/` dir.** `training.yaml` and `ignore`
  belong in git. The scaffolded `.dlm` (when present) is
  project-local config; commit or gitignore based on your team's
  norms.
- **Secrets are your job.** The default-exclude set catches the
  obvious foot-guns (`.env`, `*.pem`) but isn't a security
  boundary. Add explicit excludes for anything project-specific.
