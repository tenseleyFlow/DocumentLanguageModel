# Interactive sessions

`dlm repl <path>` gives you a conversational prompt against a trained
`.dlm` without reloading the model between turns. It's the
human-facing counterpart to `dlm prompt`: same backend plumbing,
multi-turn context, readline-style editing, history that persists
across sessions.

## When to use it

- You're iterating on how the adapter responds to a series of
  related prompts.
- You want to tune generation knobs (`temperature`, `top_p`) on the
  fly without restarting.
- You're demoing the trained document to someone who expects a
  chat-style interface.

Single-shot `dlm prompt <path> "…"` is still the right call for
scripts, one-liners, or piped input.

## Basic usage

```bash
$ dlm repl mydoc.dlm
dlm repl — /help for commands, /exit to quit (history: ~/.dlm/history)
> Hello.
Hi! How can I help?
[1] > What does the document cover?
…
```

The prompt `[N] > ` reports how many turn pairs have already
happened. Full chat template is applied every turn, so the model
sees the whole conversation in context.

## Slash commands

| Command | Effect |
|---|---|
| `/help` | Print the command list. |
| `/exit` or `/quit` | End the session (Ctrl-D does the same). |
| `/clear` | Reset conversation history; model stays loaded. |
| `/save <path>` | Write history as JSON for later review or replay. |
| `/history` | Print the current conversation. |
| `/adapter <name>` | Switch active adapter (multi-adapter docs only). |
| `/params key=value` | Update a generation knob in place. |
| `/params` | Print current generation knobs. |
| `/model` | Print the active backend + adapter. |

### `/params`

Accepts these keys: `temperature`, `top_p`, `top_k`, `max_new_tokens`,
`repetition_penalty`. Multiple updates in one line are allowed:

```
[2] > /params temperature=0.3 top_p=0.9
temperature=0.3 top_p=0.9 top_k=None max_new_tokens=256 repetition_penalty=None
```

Bad values reject without partial updates — your previous knobs
stay intact.

## Ctrl-C semantics

- **During input**: cancels the line you're editing. The REPL
  redraws the prompt; session keeps running. (Use `/exit` or
  Ctrl-D to leave.)
- **During generation**: stops the model mid-stream. Tokens
  emitted so far stay on screen, and the partial response is
  appended to history with a `[cancelled]` marker so the model
  sees it in future turns.

## History persistence

Readline history (your past prompts) is stored at
`~/.dlm/history`. Arrow-up / Ctrl-R work across sessions. This is
*not* the conversation transcript — use `/save <path>` to export
that.

## Non-interactive output

`dlm repl mydoc.dlm > transcript.txt` detects the non-TTY stdout
and disables streaming; each response is printed once, complete.
Useful for capturing a scripted session (feed prompts on stdin).

## Backend selection

`--backend auto` (default) picks MLX on Apple Silicon when the
`mlx` extra is installed, PyTorch otherwise. Force either with
`--backend pytorch|mlx`. MLX drops `top_p`/`top_k`/
`repetition_penalty` today — see the [CLI reference](../cli/reference.md)
for the full matrix.
