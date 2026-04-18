# Contributing

Thanks for your interest. DocumentLanguageModel is being built to a specific
bar — "a real LLM, not a toy" — and contributions are welcome where they
respect that bar.

## Dev loop

```sh
uv sync --all-extras --dev
uv run dlm --help
uv run ruff check .
uv run ruff format .
uv run mypy src/dlm
uv run pytest
```

`mypy --strict` is required; no loosening accepted.

Testing details (markers, fixtures, golden outputs, CI matrix) live in
[docs-internal/README-testing.md](./docs-internal/README-testing.md).

## Coverage gates

CI enforces ≥ 95% branch coverage on each shipped module:

```sh
uv run pytest tests/unit/doc      --cov=src/dlm/doc      --cov-fail-under=95
uv run pytest tests/unit/store    --cov=src/dlm/store    --cov-fail-under=95
uv run pytest tests/unit/hardware --cov=src/dlm/hardware --cov-fail-under=95
```

New modules ship with their own gate.

## Commits

- **Imperative, terse, one line** unless a technical choice requires
  elaboration. `feat(export): emit explicit Go template in Modelfile`, not
  `Updated the export module to emit...`.
- **One commit per logical unit.** A new file + its test are usually one
  commit; unrelated changes never share a commit.
- **Never `git add -A`.** Stage specific files by name.
- **No coauthor trailers.** No `Co-Authored-By` lines.
- **No `--no-verify`.** Fix the hook failure rather than bypassing it.

## Pull requests

- Link the issue or discussion your change resolves.
- Any change that could alter training output under a fixed seed must flag
  "breaks determinism golden" and propose the regeneration path.
- New cross-cutting concepts need an issue / design note first.

## Scope discipline

- Features beyond the current release's scope need a proposal in an issue
  before the PR lands.
- Do not add backwards-compat shims for code that hasn't shipped yet.
- Reject shortcuts that create silent-failure surfaces.
