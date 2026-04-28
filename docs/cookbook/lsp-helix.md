# Use DLM in Helix

The `dlm-lsp` language server attaches to Helix through `languages.toml`.

## Install

```bash
pip install dlm-lsp
which dlm-lsp   # confirm it's on PATH
```

## Configure

Add to `~/.config/helix/languages.toml`:

```toml
[language-server.dlm-lsp]
command = "dlm-lsp"

[[language]]
name = "dlm"
scope = "source.dlm"
file-types = ["dlm"]
roots = []
comment-token = "#"
language-servers = ["dlm-lsp"]
indent = { tab-width = 2, unit = "  " }
```

If `dlm-lsp` isn't on your PATH (e.g. installed inside a venv), pass an
absolute path:

```toml
[language-server.dlm-lsp]
command = "/Users/you/.venvs/dlm/bin/dlm-lsp"
```

Reload Helix or restart it for the language definition to register.

## Verify

Open a `.dlm` file. Run `:lsp-restart` if completions don't appear right
away, then check `:log-open` to confirm `dlm-lsp` started without errors.

You'll get:

- Diagnostics on schema errors
- Hover (`K`) on `base_model:` and section fences
- Completions (`Ctrl-x`) for the base-model registry
- Code actions (`<space>a`) where applicable

## Optional: syntax highlighting

Helix uses Tree-sitter for highlighting, not TextMate, so the VSCode
extension's grammar isn't reusable directly. The LSP semantic-token
interface is the cleanest path; if you want richer highlighting, write a
small Tree-sitter grammar that injects YAML for the frontmatter and
Markdown for the body, then references the section fence as a custom
node.
