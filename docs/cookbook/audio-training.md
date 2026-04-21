# Audio training (audio + Qwen2-Audio)

Sprint 35.2 adds audio sections to `.dlm` files. This recipe walks a
spoken-corpus workflow end-to-end: scaffold → drop clips + transcripts
→ train → query the adapter against new audio.

## Prerequisites

- Apple Silicon with ≥ 32 GB unified memory, or CUDA ≥ SM 8.0 with ≥
  24 GB VRAM. Qwen2-Audio-7B-Instruct fp16 weighs ~15 GB; the 16 GB
  consumer GPUs don't fit this base without quantization (4-bit audio
  training is deferred).
- A Hugging Face account with the [Qwen2-Audio-7B-Instruct terms
  accepted](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) and
  `HF_TOKEN` exported.
- Qwen2-Audio cached locally (`huggingface-cli download
  Qwen/Qwen2-Audio-7B-Instruct`). First train without this triggers
  the download automatically.
- The `audio` extra installed: `uv sync --extra audio` (pulls
  `soundfile` for decoding `.wav` / `.flac` / `.ogg`).

## Step 1 — Scaffold an audio `.dlm`

```bash
dlm init my-audio.dlm --audio --i-accept-license
```

`--audio` pins the base to `qwen2-audio-7b-instruct` and emits a
schema-v11 scaffold with a sample `::audio::` fence. The initial
body references `clips/your-clip.wav` (non-existent by default —
drop a real clip at that path before the first train).

## Step 2 — Author audio sections

Two ways to supply audio. Inline each fence with the transcript:

```dlm
::audio path="clips/intro.wav" transcript="Welcome to the podcast."::

::instruction::
### Q
What did the speaker say?

### A
"Welcome to the podcast."
```

Or ingest a directory through a source directive. Audio files need
a matching `<stem>.txt` sidecar with the transcript:

```
corpus/
├── intro.wav
├── intro.txt         ← transcript for intro.wav
├── outro.flac
└── outro.txt
```

```dlm
---
dlm_id: 01JZ...
dlm_version: 11
base_model: qwen2-audio-7b-instruct
training:
  sources:
    - path: ./corpus
      include: ["**/*.wav", "**/*.flac"]
---
```

Each `.wav`/`.flac`/`.ogg` with a sibling `.txt` becomes an
`::audio::` section. Files without a sidecar are silently skipped +
counted in provenance (`dlm show --json` surfaces the skip count
under `source_directives[].skipped_audio_no_transcript`).

## Step 3 — Train

```bash
dlm train my-audio.dlm
```

The trainer:

1. Loads Qwen2-Audio via `Qwen2AudioForConditionalGeneration` + its
   matching `AutoProcessor` (feature extractor + tokenizer).
2. Walks `training.sources` directives, copies each audio file's
   bytes into the content-addressed blob store at
   `~/.dlm/store/<dlm_id>/blobs/`.
3. Emits training rows shaped
   `{audio_blob_sha, audio_path, text: "<|AUDIO|>\n<transcript>"}`.
4. Runs our `AudioLmCollator` (custom — TRL 1.2 has no audio
   auto-dispatch). The collator decodes each waveform via
   `soundfile`, truncates to 30 s, hands the batch to the processor,
   and emits `{input_ids, attention_mask, input_features, labels}`.
5. Commits the adapter under `adapter/versions/v0001/`.

**Sample-rate policy.** Sprint 35.2 v1 refuses audio whose native
rate doesn't match the base's pinned `sample_rate` (Qwen2-Audio:
16 kHz). Re-encode with ffmpeg:

```bash
ffmpeg -i in.mp3 -ar 16000 out.wav
```

Resampling as an automatic step lands in a 35.2 follow-up.

**Wall-clock expectations.** A 5-clip (30 s each) corpus + 3 epochs
on an RTX 4090 at `micro_batch_size=1` + `grad_accum=4` takes about
20 minutes. Apple Silicon is ~4× slower.

## Step 4 — Prompt the trained adapter

```bash
dlm prompt my-audio.dlm --audio clips/new-clip.wav \
  "What did the speaker say?"
```

`--audio` is required for audio bases. Repeat the flag for multi-clip
prompts; each occurrence expands to one `<|AUDIO|>` placeholder that
the processor replaces with 750 audio tokens (30 s × 25 tokens/s).

`--image` and `--audio` cannot be combined — each targets a different
modality.

## Step 5 — Export

Audio bases take the HF-snapshot path (audio architectures aren't on
`llama.cpp`'s roadmap, so GGUF isn't available):

```bash
dlm export my-audio.dlm
```

Writes to `~/.dlm/store/<dlm_id>/exports/hf-audio-snapshot/`:

```
hf-audio-snapshot/
  adapter/                  # PEFT LoRA weights
  processor/                # AutoProcessor config + feature extractor
  snapshot_manifest.json    # export_target=hf_snapshot + sha256s
  README.md                 # how to load downstream
```

Load on the other side:

```python
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import PeftModel

base = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
)
model = PeftModel.from_pretrained(base, "./adapter")
processor = AutoProcessor.from_pretrained("./processor")
```

The base isn't bundled — recipients download it on first use.

## Troubleshooting

### "audio not found: clips/your-clip.wav"

The `--audio` scaffold points at a placeholder; drop a real clip at
that path or edit the `::audio path="..."::` fence.

### "native sample_rate=44100 Hz does not match pinned 16000 Hz"

Your clip is at 44.1 kHz (CD rate) but Qwen2-Audio expects 16 kHz.
Re-encode:

```bash
ffmpeg -i in.wav -ar 16000 out.wav
```

### "audio-language base requires at least one --audio PATH"

You ran `dlm prompt` on an audio `.dlm` without attaching a clip.
Audio bases always expect a waveform — even a throwaway question
about transcript content needs an audio input to anchor the
placeholder token.

### "AUDIO section has empty transcript"

Both the inline `transcript="..."` form and the sibling `<stem>.txt`
form must produce a non-empty transcript. Whitespace-only transcripts
are refused (the trainer has no target text to predict).

### Disk / memory issues

Qwen2-Audio-7B is ~15 GB on disk and another ~15 GB in memory at
fp16. Close other GPU consumers, use `--max-steps 1` to dry-run, or
wait for the audio-QLoRA path (deferred).

## What's not yet in Sprint 35.2

- **Resampling.** v1 refuses sample-rate mismatches. Automatic
  resampling (probably via `soxr` or the HF feature extractor's own
  support) lands in a follow-up.
- **MP3 support.** `soundfile` needs libsndfile ≥ 1.1 for MP3;
  we lock to `.wav` / `.flac` / `.ogg` in v1 to avoid shipping a
  libsndfile hard-pin.
- **Audio feature caching in training.** `AudioCache` is wired for
  the standalone inference path and the slow integration test;
  the training hot path doesn't re-use the cache yet (each epoch
  re-extracts features). Meaningful speed-up lands alongside
  multi-epoch audio corpora where re-extraction dominates.
- **QLoRA for audio.** 4-bit audio training needs extra safety
  testing for the audio encoder weights; deferred.
- **Multiple audio clips per section.** Each `::audio::` fence carries
  one clip; prompts can stack multiple `<|AUDIO|>` tokens by repeating
  `--audio` on the CLI.
