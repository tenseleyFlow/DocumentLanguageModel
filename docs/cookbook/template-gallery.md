# Template gallery

The fastest path from "I want to try `dlm`" to a trained adapter is
`dlm init --template <name>`. Each template is a curated `.dlm` file
with a sensible base model, body content to train on, and sample
prompts to exercise after training.

## Listing the gallery

```bash
$ dlm templates list
changelog                 Changelog entry generator              (smollm2-360m)
coding-tutor              Coding tutor (Python, curated)         (qwen2.5-coder-1.5b)
domain-kb                 Domain knowledge base                  (qwen2.5-3b)
meeting-notes-summarizer  Meeting notes → decision log           (qwen2.5-3b)
personal-assistant        Personal assistant                     (qwen2.5-1.5b)
regex-buddy               Regex explainer                        (qwen2.5-coder-1.5b)
shell-one-liner           Shell one-liner helper                 (qwen2.5-coder-1.5b)
writing-partner           Writing partner (stylistic continuation) (llama-3.2-3b)
```

Pass `--json` for machine-readable output (every field of the template's
`meta.yaml` including `domain_tags`, `expected_steps`,
`expected_duration`, and `sample_prompts`).

## Creating a new `.dlm` from a template

```bash
$ dlm init mytutor.dlm --template coding-tutor
init: wrote mytutor.dlm from template coding-tutor (Coding tutor
(Python, curated)) — base qwen2.5-coder-1.5b.

$ dlm train mytutor.dlm
$ dlm prompt mytutor.dlm "What are Python decorators?"
```

The template's `recommended_base` is adopted automatically. If you pass
`--base` alongside `--template`, the template wins (and you get a
note); the bundled body was authored against its recommended base, so
swapping bases mid-body is an advanced move — edit the frontmatter
yourself after `init` if that's what you want.

## What's in a template

Each template is a pair of files:

```
coding-tutor.dlm        # the `.dlm` body: frontmatter + sections
coding-tutor.meta.yaml  # metadata — name, title, tags, recommended_base, summary
```

The `meta.yaml` schema:

```yaml
name: coding-tutor               # filename stem, required to match
title: Coding tutor (Python, curated)
domain_tags: [code, python, tutor]
recommended_base: qwen2.5-coder-1.5b
expected_steps: 800              # rough step count at defaults
expected_duration:               # hardware-tier → wall-clock estimate
  cuda-sm80+: "~5 min"
  mps: "~15 min"
  cpu: "~2 hr"
summary: |
  A compact Python-focused Q&A tutor.
sample_prompts:
  - "What are Python decorators?"
```

Invalid metadata (unknown keys, missing required fields, sidecar
absent) drops the template from the listing with a logged warning —
never silently served.

## Where the gallery lives

The bundled gallery ships inside the `dlm` package at
`src/dlm/templates/gallery/`. `importlib.resources.files` resolves the
path regardless of whether you're running from a source checkout or a
`pip install`-ed wheel.

## Remote gallery — deferred

The sprint spec calls for `dlm templates list --refresh` to pull from
`github.com/<org>/dlm-templates` with signed-tag verification against a
pinned public key. That upstream repo and signing key are pending;
until they land, `--refresh` emits a clear warning and falls back to
the bundled gallery. All usage here works offline today.
