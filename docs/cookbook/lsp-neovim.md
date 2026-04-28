# Use DLM in Neovim

The `dlm-lsp` language server attaches to Neovim through
[`nvim-lspconfig`](https://github.com/neovim/nvim-lspconfig).

## Install

```bash
pip install dlm-lsp
which dlm-lsp   # confirm it's on PATH
```

## Configure

Add to your Neovim config (Lua, e.g. `~/.config/nvim/lua/plugins/dlm.lua`
or anywhere in your `init.lua`):

```lua
-- Recognize .dlm files
vim.filetype.add({ extension = { dlm = "dlm" } })

-- Register the LSP
local lspconfig = require("lspconfig")
local configs = require("lspconfig.configs")

if not configs.dlm_lsp then
  configs.dlm_lsp = {
    default_config = {
      cmd = { "dlm-lsp" },
      filetypes = { "dlm" },
      root_dir = lspconfig.util.find_git_ancestor,
      single_file_support = true,
      settings = {},
    },
  }
end

lspconfig.dlm_lsp.setup({})
```

If `dlm-lsp` isn't on your PATH (e.g. installed inside a venv), pass the
absolute path:

```lua
cmd = { "/Users/you/.venvs/dlm/bin/dlm-lsp" },
```

## Verify

Open a `.dlm` file. Run `:LspInfo` and confirm `dlm_lsp` is attached. If
not, `:LspLog` shows the spawn error.

You'll get:

- Diagnostics through the standard Neovim diagnostic interface
- Hover (`K`) on `base_model:` keys and section fences
- Completion through your usual completion plugin
  (`nvim-cmp`, `coq_nvim`, the built-in `Ctrl-x Ctrl-o`, etc.)
- Code actions through `vim.lsp.buf.code_action()`

## Optional: Tree-sitter highlighting

Neovim uses Tree-sitter for highlighting. There is no `tree-sitter-dlm`
parser yet; if you want richer highlighting before that lands, set the
filetype to `markdown` for the body region and rely on LSP semantic
tokens for the frontmatter and fence accents.
