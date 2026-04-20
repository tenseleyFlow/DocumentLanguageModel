# Quantization tradeoffs

`dlm export --quant <Q>` picks how aggressively the base model gets
compressed on the way out. Smaller files run faster; more aggressive
quantization costs quality. Here's the cheat sheet.

## The quant levels

| Quant | Bits/weight (avg) | Size vs F16 | Notes |
|---|---|---|---|
| `F16` | 16 | 100% | No quantization. Baseline for quality comparisons. |
| `Q8_0` | 8.5 | ~55% | Near-lossless. Still noticeably smaller. |
| `Q6_K` | 6.6 | ~42% | Strong quality, middle-ground size. |
| `Q5_K_M` | 5.7 | ~37% | The "willing to spend disk for quality" default. |
| `Q4_K_M` | 4.8 | ~31% | The recommended starting point. Great quality/size. |

For a 1.5B-parameter base:

- `F16` → ~3.0 GB
- `Q8_0` → ~1.6 GB
- `Q5_K_M` → ~1.1 GB
- `Q4_K_M` → ~0.95 GB

## When to pick which

**`Q4_K_M` (default)**
: Production recommendation for v1.0. Good quality, fits in a
  "normal" amount of RAM/VRAM, fast inference. Start here.

**`Q5_K_M`**
: You have disk to spare and want slightly better generations. The
  size bump is modest; the quality bump is audible.

**`Q6_K`**
: Willing to trade another ~10% disk for near-`Q8_0` quality. Useful
  when A/B testing against full-precision behavior.

**`Q8_0`**
: Baseline for "is the quant regression real?" investigations. If
  `Q8_0` also regresses, the bug isn't the quant.

**`F16`**
: Debugging a quant-caused regression, or running on a platform where
  the kernels for quantized inference are slower than f16 for some
  reason (rare on modern CPUs/GPUs).

## Imatrix-calibrated quantization

Sprint 11.6 added automatic **importance-matrix** calibration when
your store has enough replay-corpus text. The imatrix tells
`llama-quantize` which weight directions matter most for the model's
behavior on YOUR content — so the low-bit quants preserve the
directions that matter and compress the rest more aggressively.

```sh
$ uv run dlm export tutor.dlm --quant Q4_K_M
# imatrix built from replay/corpus.zst, cached, applied automatically
```

Empirically, imatrix-calibrated `Q4_K_M` is close to static `Q5_K_M`
quality at `Q4_K_M` size. The imatrix is cached per-document so
subsequent `dlm export` runs at the same quant reuse it.

Opt out with `--no-imatrix` if you want a static quant for a
regression comparison.

## QLoRA + `--merged` is a safety gate

```sh
$ uv run dlm export tutor.dlm --merged --quant Q4_K_M
export: merge refused: adapter was trained with QLoRA (4-bit base);
        merging into a quantized base is precision-unsafe. Re-run
        with --dequantize to dequantize to fp16 before merge, or drop
        --merged to ship base + adapter separately.
```

The default (base + adapter separate) is fine for almost every use
case — Ollama loads them with `FROM` + `ADAPTER` directives and
merges at inference time. Use `--merged --dequantize` only if you
need a single-file deployment and accept the bigger artifact.

## See also

- [First export walkthrough](../getting-started/first-export.md) for
  the full flow
- [Determinism](../determinism.md) — the quant tuple participates in
  the `dlm.lock` reproducibility record
