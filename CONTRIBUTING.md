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

Tag-driven. Pushing `v*` triggers `.github/workflows/release.yml`,
which runs the full CI gate, builds a "fat" source tarball (includes
`vendor/llama.cpp/` so the Homebrew formula can drop the convert
scripts into libexec without cloning submodules), and creates a
GitHub release with the tarball + computed sha256.

Docs are built strict in CI but not hosted — read them in-repo or in
the brew-installed tarball. Hosting (GitHub Pages or a custom
subdomain) is a separate change, deferred post-v0.9.0.

We publish via our Homebrew tap —
[tenseleyFlow/homebrew-tap](https://github.com/tenseleyFlow/homebrew-tap).
**We do not publish to PyPI.** Rationale lives in the audit-05 /
release-mode discussion; the short version is: PyPI makes versions
permanent, requires us to maintain a ~5 GB transitive dep surface,
and signals "this is battle-tested" in a way we're not ready to back
yet.

### Conservative versioning

Stay below `v1.0.0` until a human has trained + exported +
`ollama run`'d an adapter end-to-end. That's the only contract v1.0
actually owes users. Current target: `v0.9.0` for the first tagged
release.

### Pre-flight (run locally before tagging)

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy src/dlm
uv run pytest
uv sync --group docs
uv run mkdocs build --strict
```

Bump the version in `pyproject.toml`, move `## [Unreleased]` entries
under a new `## [X.Y.Z]` heading in `CHANGELOG.md`, and land both in
one commit.

### Tagging

```sh
git tag v0.9.0
git push origin v0.9.0
```

`release.yml` classifies the tag via
`packaging.version.Version.is_prerelease`:

- **Prerelease** (`v0.9.0rc1`, `v0.9.0a1`, `v0.9.0-rc1`): GitHub
  release gets the `prerelease` flag so it doesn't show as "latest."
- **Release** (`v0.9.0`, `v0.9.1`): standard GitHub release.

### Bumping the Homebrew formula

After the release workflow finishes, it prints the fat-tarball sha256
in the release notes. Bump `Formula/dlm.rb` in the tap:

```ruby
url "https://github.com/tenseleyFlow/DocumentLanguageModel/releases/download/v0.9.0/dlm-v0.9.0.tar.gz"
sha256 "<copy from release notes>"
```

Then:

```sh
cd ~/path/to/homebrew-tap
brew install --build-from-source ./Formula/dlm.rb   # local smoke
brew test ./Formula/dlm.rb                          # runs the `test do` block
git commit -am "dlm: bump to v0.9.0"
git push
```

### Rollback

Homebrew rollback is straightforward: delete the bad GitHub release
(or mark it draft), revert the formula bump in the tap. Users who
already installed the bad version can `brew uninstall dlm && brew
install dlm` to pick up the revert.

Thanks again — reach out in issues if anything's unclear.

-mfw
