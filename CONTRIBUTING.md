# Contributing

Hey — glad you're here. DLM is a small, opinionated project, and patches
that fit the project's shape are very welcome. No sign-up, no CLA, just
open an issue if the change is non-trivial and a PR when you're ready.

## Getting set up

You'll need Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```sh
git clone https://github.com/tenseleyFlow/DocumentLanguageModel.git
cd DocumentLanguageModel
uv sync --all-extras --dev
uv run dlm --help
```

The four checks CI runs — run them locally before pushing:

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy src/dlm
uv run pytest
```

If you've only touched one module, you can run that module's tests
directly (`uv run pytest tests/unit/pack -q`, etc.). The full suite
takes around eight seconds.

`mypy --strict` is non-negotiable — if you need to loosen a type,
please fix the type at its source instead.

### Pre-commit hooks

Install once per clone to catch ruff / mypy / non-slow-pytest
failures before they reach CI:

```sh
uv run pre-commit install
```

The config lives at `.pre-commit-config.yaml`. The local hook runs
`pytest -m "not slow and not gpu and not online"`, so it's ~8 seconds
on a warm cache.

Testing conventions (markers, fixtures, the tiny-model fixture,
golden outputs) are documented separately at
[docs-internal/README-testing.md](./docs-internal/README-testing.md).

## Coverage

We keep each shipped module above 95% line coverage; CI enforces it.
When you add a new module, add a matching gate in `.github/workflows/ci.yml`
next to the existing ones. When you add a branch that's hard to exercise
in unit tests — a real-GPU path, a subprocess that needs a full HF
model — mark it `# pragma: no cover` with a short reason, and write a
slow-marked integration test for it.

## Commits

I care about commits being readable later — not because there's a
style police, but because `git log` is the one place future-you reads
history when something breaks.

- One commit per logical change. A new source file plus its tests is
  usually one commit. Unrelated fixes go in separate commits.
- Imperative subject line, under ~72 chars:
  `feat(export): emit explicit Go template in Modelfile`, not
  `Updated the export module to emit the template`.
- If the *why* needs a paragraph, put it in the commit body. If it
  doesn't, one line is fine.
- Stage files by name (`git add path/to/file.py`). `git add -A` picks
  up stray files we don't want in the repo.
- No coauthor trailers. No `--no-verify`. If a pre-commit hook fails,
  fix what it's telling you about.

## Pull requests

- Link the issue or discussion your change resolves, if there is one.
- If your change could alter training output under a fixed seed, call
  it out: "this breaks the determinism golden, here's the regeneration
  path." Better to know up front than discover it in a retrain.
- For larger cross-cutting changes (a new dependency, a new on-disk
  format, anything that touches the manifest schema), open an issue
  first so we can nail down the design before you write the code.

## Scope

DLM has a clear story — edit a document, train a LoRA, export to
Ollama, do it locally, don't forget on retrain. Contributions that
fit that story are the easy ones to land. Things that add scope
(new training paradigms, cloud integrations, alternate inference
backends) are worth discussing in an issue first; sometimes the
answer is "yes, but later," and it saves you from writing code that
won't merge.

A few things we actively don't want:

- Silent-failure surfaces. If a preflight can't verify something,
  it refuses rather than warns.
- Backwards-compat shims for code that hasn't shipped yet. If a v1
  hasn't gone out the door, you can rename a function without a
  deprecation wrapper.
- Telemetry or network calls outside of model download. Ever.

## Releasing

Tag-driven: pushing a `v*` tag triggers `.github/workflows/release.yml`,
which runs the full CI gate, builds wheel + sdist via `uv build`, and
publishes to PyPI via trusted-publisher OIDC.

### One-time PyPI trusted-publisher setup

Before the first real release:

1. Create a PyPI account for the `dlm` project (someone with publish
   rights has to own this).
2. Under project settings → **Publishing** → **Add a new pending
   publisher**, fill in:
   - Owner: `tenseleyFlow`
   - Repository name: `DocumentLanguageModel`
   - Workflow filename: `release.yml`
   - Environment name: `pypi`
3. Repeat on test.pypi.org with environment name `test-pypi`.
4. In the GitHub repo settings → **Environments**, create both
   `pypi` and `test-pypi` environments. Neither needs secrets; the
   OIDC token is minted per run.

### Pre-flight

The CI gate runs the full check suite (ruff, mypy, non-slow pytest,
`mkdocs build --strict`). Before tagging, eyeball these locally:

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy src/dlm
uv run pytest
uv sync --group docs
uv run mkdocs build --strict
```

Then bump the version in `pyproject.toml`, update `CHANGELOG.md`
(move the `## [Unreleased]` entries under a new `## [X.Y.Z]` heading),
and land both in the same commit.

### Tagging

`release.yml` classifies tags via `packaging.version.Version.is_prerelease`:

- **Prerelease** (routes to `test.pypi.org`): any PEP 440 prerelease.
  Canonical: `v1.0.0rc1`, `v1.0.0a2`, `v1.0.0b3`. Hyphenated also
  works: `v1.0.0-rc1`.
- **Release** (routes to `pypi.org`): clean `vMAJOR.MINOR.PATCH`.

```sh
# Dry-run via test.pypi.org first
git tag v1.0.0rc1
git push origin v1.0.0rc1

# Verify on https://test.pypi.org/project/dlm/, then:
git tag v1.0.0
git push origin v1.0.0
```

The release workflow publishes, then the `deploy-docs` job builds the
MkDocs site and pushes it to `gh-pages`.

### Rollback

There's no unpublish on PyPI (trusted-publisher or otherwise). If a
release is bad, bump the patch version and cut a fixed release rather
than trying to yank the old one.

Thanks again — reach out in issues if anything's unclear.

-mfw
