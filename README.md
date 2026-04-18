# DocumentLanguageModel

> A text file with a `.dlm` extension becomes a local, reproducible, trainable LLM.

Edit the document, retrain, share. Not a toy — LoRA/QLoRA on a real pretrained
base, exportable to Ollama.

**Status:** pre-alpha (Sprint 01 of 29). See `.docs/sprints/00-index.md` for the
development plan.

## Installation

```
uv sync
uv run dlm --help
```

## Project layout

```
src/dlm/            Package sources
tests/              Test suite
.docs/              Planning documents (local-only; gitignored)
.refs/              Cloned reference projects (local-only; gitignored)
```

## License

MIT
