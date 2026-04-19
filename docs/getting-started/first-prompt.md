# First prompt

`dlm prompt` runs inference against the current adapter using the base
model. It's the fastest way to check "did the training actually stick?"
without involving Ollama or GGUF conversion.

## The happy path

```sh
$ uv run dlm prompt tutor.dlm "What is a Python decorator?"
A decorator is a function that takes another function as input…
```

Behind the scenes:

1. `dlm prompt` parses the `.dlm`, resolves the base model, and
   checks the hardware doctor's capability report.
2. It loads the base model + `adapter/current.txt`-pointed LoRA
   weights via PEFT.
3. It calls `generate()` with your prompt, `--max-tokens 256`,
   `--temp 0.7` by default.
4. The response is streamed to stdout; the Rich reporter writes
   progress / plan info to stderr so you can pipe stdout cleanly.

## Deterministic generation

For reproducible output (useful for comparing adapters), pin
temperature to 0:

```sh
$ uv run dlm prompt tutor.dlm --temp 0 --max-tokens 32 "Say hi"
```

Greedy decoding is deterministic when the weights are byte-identical —
which is the whole point of the [determinism contract](../determinism.md).

## Verbose plan

Pass `--verbose` to surface the inference plan before generation:

```sh
$ uv run dlm prompt tutor.dlm --verbose "Hello"
plan: {'device': 'mps', 'dtype': 'fp16', 'adapter_path': '...', 'quantization': 'none'}
adapter: ~/.dlm/store/01KC…/adapter/versions/v0001
Hello! How can I help you today?
```

The `plan` dict is the same object written into `manifest.json` on
training, so you can cross-reference what the model was doing the
last time it trained.

## Piping and stdin

Prompt via stdin for long inputs:

```sh
$ cat long-prompt.txt | uv run dlm prompt tutor.dlm
```

An empty stdin (no query argument either) exits with a non-zero code
and a clear error, rather than hanging.

## Next

Happy with inference? [Export to Ollama](first-export.md) for a real
standalone model.
