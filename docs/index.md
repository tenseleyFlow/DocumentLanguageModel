# DocumentLanguageModel

> `.dlm` is a trainable local AI document format: typed sections, directives,
> replay-backed retraining, and export.

DocumentLanguageModel (DLM) is a local-first training and inference toolkit
built around authored documents instead of hosted dashboards.

A `.dlm` can be a hand-authored training doc, a directive-driven entrypoint
into a codebase, a multi-adapter project with learned routing, or a selected
multimodal / audio-language document. DLM trains LoRA / QLoRA / DoRA adapters
on real pretrained bases, keeps replay history, and exports local runtimes such
as Ollama, `llama-server`, `vllm`, and `mlx-serve`.

## What DLM Ships Today

- **Structured `.dlm` authoring** with frontmatter plus typed body sections
  like prose, `::instruction::`, `::preference::`, `::image::`, and
  `::audio::`
- **Directive-driven corpus building** via `training.sources`, plus nested
  `.dlm/training.yaml` / `.dlm/ignore` for repo-local curation
- **Modern base-model registry** across text, reasoning, sparse-MoE,
  vision-language, and audio-language rows
- **Replay-backed retraining** so edits accumulate instead of silently wiping
  prior state
- **Synthetic data loops** through `dlm synth instructions` and
  `dlm synth preferences`
- **Multi-adapter docs + learned gating** for separating knowledge, tone, or
  persona lanes inside one project
- **Local iteration UX** with `prompt`, `repl`, `train --watch`, `metrics`,
  and `doctor`
- **Runtime export** to `ollama`, `llama-server`, `vllm`, and `mlx-serve`
- **Probe-driven improvement** through `sway`-style harvest flows

## 30-Second Demo

```sh
$ uv run dlm init tutor.dlm --base smollm2-135m
$ $EDITOR tutor.dlm
$ uv run dlm train tutor.dlm
$ uv run dlm prompt tutor.dlm "Explain Python decorators"
$ uv run dlm export tutor.dlm --target ollama --name my-tutor
```

## Where To Start

| If you want to… | Start here |
|---|---|
| Install DLM and run the first cycle | [Getting started → Install](getting-started/install.md) |
| Understand the `.dlm` file format | [Frontmatter](format/frontmatter.md) and [Section grammar](format/sections.md) |
| Train across a real repo | [Training across codebases](cookbook/training-across-codebases.md) |
| Use named adapters and routing | [Multi-adapter](cookbook/multi-adapter.md) and [Learned adapter gate](cookbook/learned-adapter-gate.md) |
| Work with images or audio | [Multimodal training](cookbook/multimodal-training.md) and [Audio training](cookbook/audio-training.md) |
| Turn prose into instruction data | [Synthesize training data](cookbook/synthesize-training-data.md) and [Bootstrap self-improving](cookbook/bootstrap-self-improving.md) |
| Mine preference pairs from a live adapter | [Self-improving loop](cookbook/self-improving-loop.md) and [Reward-model integration](cookbook/reward-model-integration.md) |
| Export or ship a model | [Multi-target export](cookbook/multi-target-export.md), [CLI reference](cli/reference.md), and [Determinism](determinism.md) |
| Pull eval failures back into training | [Probe-driven training](cookbook/probe-driven-training.md) |

## Status

DLM is pre-v1.0 but substantially broader than the original MVP framing.
Core author/train/prompt/export/pack/share flows are in place, and current
runtime-target work is extending export beyond the original Ollama-only path.
