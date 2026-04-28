# Use DLM in Zed

The `dlm-lsp` language server is editor-agnostic. Zed picks it up through a
custom language definition.

## Install

```bash
pip install dlm-lsp
which dlm-lsp   # confirm it's on PATH
```

## Configure

Zed configures custom languages through `~/.config/zed/settings.json`:

```json
{
  "languages": {
    "DLM": {
      "lsp": ["dlm-lsp"]
    }
  },
  "lsp": {
    "dlm-lsp": {
      "binary": {
        "path": "dlm-lsp",
        "arguments": []
      }
    }
  },
  "file_types": {
    "DLM": ["dlm"]
  }
}
```

If `dlm-lsp` isn't on your PATH (e.g. installed inside a venv), set
`binary.path` to its absolute location:

```json
"binary": {
  "path": "/Users/you/.venvs/dlm/bin/dlm-lsp",
  "arguments": []
}
```

## Verify

Open any `.dlm` file in Zed. You should see:

- Diagnostics on schema errors (red squiggles in frontmatter on bad keys)
- Hover info on `base_model:` keys
- Completions on the base-model registry

The LSP log is reachable through the command palette → `zed: open log`.

## Limitations vs. the VSCode extension

Zed clients consume the LSP only — they don't render the side panel or
quick-insert UI. For those, use the VSCode extension. Everything that's
diagnostic, hover, completion, or code-action is fully available in Zed.
