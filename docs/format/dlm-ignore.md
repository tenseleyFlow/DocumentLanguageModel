# `.dlm/ignore` reference

A plain-text file that lives beside `.dlm/training.yaml` and carries
one ignore pattern per line. Same anchor, different file — a repo
can have either, both, or neither.

Use `.dlm/ignore` when you want drive-by exclusions without writing
YAML. A three-line file is legal and common.

!!! info "`.dlm` (file) vs `.dlm/` (directory) — two different things"
    Both `.dlm` shapes live in this project and it's worth naming the
    distinction once:

    - `mydoc.dlm` — **a file** with a `.dlm` extension. The document
      itself, with YAML frontmatter + markdown body. Plural: users own
      many `mydoc.dlm` files, one per trainable corpus.
    - `.dlm/` — **a directory** named literally `.dlm`. Lives inside a
      codebase or data tree and holds `training.yaml` + `ignore` +
      (when auto-scaffolded by `dlm train <dir>`) a `corpus.dlm` file.

    The two share a name but not a namespace. When `dlm train <dir>`
    auto-scaffolds, it creates a `.dlm/` directory inside `<dir>` and
    places `corpus.dlm` inside it — so a single tree can end up with
    `<dir>/.dlm/corpus.dlm`.

## Example

```
# <repo-root>/.dlm/ignore
# Comments start with #. Blank lines are skipped.

*.min.js
**/fixtures/huge_*.json
docs/generated/

# Negate to re-include a file an earlier pattern excluded:
!docs/generated/README.md

# Trailing / = directory-only
build/

# Leading / = anchored to this .dlm/'s parent (not any ancestor)
/scripts/local-dev.sh
```

## Grammar — supported

| Syntax | Meaning |
|---|---|
| `# comment` | Line comment. Leading whitespace allowed before `#`. |
| blank line | Skipped. |
| `pattern` | Exclude match. |
| `!pattern` | Re-include (negation) what an earlier pattern excluded. |
| `pattern/` | Directory-only — match only dirs, or paths under a dir that matches. |
| `/pattern` | Anchored — match only at the `.dlm/`'s parent (not any ancestor). |
| `**` | Globstar — matches zero or more path segments. |
| `*` | Matches one path segment (non-`/`). |
| `?` | Matches one character (non-`/`). |

## Grammar — NOT supported

The parser is a **strict subset** of `.gitignore`. Users coming from
git may expect these and be surprised:

| Missing feature | Workaround |
|---|---|
| Character classes `[abc]` | List patterns separately: `a.py`, `b.py`, `c.py`. |
| Backslash escapes | Avoid literal `*`, `?`, `#`, `!`, `/` at the start of a pattern; rename files if you must. |
| Whitespace-escape with backslash | File names with trailing whitespace aren't supportable. |

If you need any of the above, fall back to `training.yaml`'s
`exclude:` list — same glob grammar as `include:`, same fnmatch-style
semantics as elsewhere in dlm.

## Semantics — last-match-wins

Rules are evaluated top-to-bottom within a single file, and outer
anchors first across the ancestor chain. For each candidate file,
the **last** rule to match determines the verdict:

- Match a non-negation rule → excluded.
- Match a `!negation` rule → re-included.
- No match → pass through (exclude/include resolution falls to the
  outer layers — parent directive, defaults, `training.yaml`).

This mirrors `.gitignore` exactly, so intuition transfers.

## Worked example

Tree:

```
repo/
  .dlm/
    ignore           *.log\n!special.log\n
  debug.log
  special.log
  docs/
    .dlm/
      ignore         !*.log\n
    docs-only.log
```

Verdict (parent directive: `include: ['**/*']`):

| File | Outcome | Why |
|---|---|---|
| `repo/debug.log` | excluded | Matches `*.log`, no later rule reopens it. |
| `repo/special.log` | included | `*.log` excludes, `!special.log` re-includes. |
| `repo/docs/docs-only.log` | included | Parent `*.log` excludes; `docs/.dlm/ignore`'s `!*.log` re-includes for this subtree. |

## Error tolerance

Malformed lines log one WARN and are dropped — a typo in one line
never fails the walk. Bare `!` or bare `/` (no pattern after the
sigil) are treated as malformed.

## Relationship with `.dlm/training.yaml`

At the same `.dlm/` anchor, both files coexist. Their exclude rules
union. Because `.dlm/ignore` is evaluated **after**
`training.yaml.exclude` at the same anchor, a `.dlm/ignore`
`!negation` can re-include a file that `training.yaml` would
otherwise drop. (Reason: `.gitignore` users expect ignore-file rules
to be the final word at their anchor.)
