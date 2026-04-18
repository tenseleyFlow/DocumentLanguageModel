# Contributing

Thanks for your interest. DocumentLanguageModel is being built to a specific
bar ("a real LLM, not a toy"); contributions are welcome where they respect
that bar.

## Read these first

1. `.docs/overview.md` — canonical project description, stack, hardware tiers
2. `.docs/findings.md` — Stage 1 exploration digest (especially §9 pitfalls)
3. `.docs/sprints/00-index.md` — sprint roadmap
4. The sprint file that owns what you're changing

## Dev loop

```
uv sync --all-extras --dev
uv run dlm --help
uv run ruff check .
uv run ruff format .
uv run mypy src/dlm
uv run pytest
```

`mypy --strict` is required; no loosening is accepted.

## Commits

- **Imperative, terse, one line** unless a technical choice requires
  elaboration. `feat(export): emit explicit Go template in Modelfile`, not
  `Updated the export module to emit...`.
- **One commit per logical unit.** A new file + its test are usually one
  commit; unrelated changes never share a commit.
- **Never `git add -A`.** Stage specific files by name.
- **No coauthor trailers.** No `Co-Authored-By` lines.
- **No `--no-verify`.** Fix the hook failure rather than bypassing it.

## Testing

Tests live in `tests/unit/`, `tests/integration/`, and `tests/e2e/`.
Markers:

- `slow` — expensive; deselected by default
- `gpu` — requires CUDA; skipped on CPU/MPS runners
- `online` — touches the network; skipped in offline CI

Default `pytest` runs the fast local subset. Bigger gates run in CI.

## Pull requests

- Link to the sprint your change relates to
- New cross-cutting concepts need a sprint file before the PR
- Any change that could alter training output under a fixed seed must flag
  "breaks determinism golden" and propose the regeneration path

## Scope discipline

- Features beyond the sprint you're working in need to be proposed in
  `.docs/sprints/` first
- Do not add backwards-compat shims for code that hasn't shipped yet
- Per findings §9, respect the pitfall inventory: reject shortcuts that
  create silent-failure surfaces
