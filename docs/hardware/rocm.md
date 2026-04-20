# AMD ROCm support

DLM supports AMD GPUs via ROCm as a **Tier 2** backend: LoRA training
and inference work, but QLoRA is refused and the CI coverage is
weaker than the CUDA path.

## What works

- **Training**: LoRA on bf16-capable AMD GPUs.
- **Inference**: `dlm prompt` uses the standard PyTorch path; MLX and
  bitsandbytes are not involved.
- **Export**: llama.cpp produces GGUF quantized weights. For
  ROCm-accelerated quantization, rebuild llama.cpp with HIP (see
  below); otherwise the default CPU build is used.

## Supported GPUs

| Generation | Arch codes       | Example SKUs                | bf16 | FA2 (via `flash_attn`) |
|------------|------------------|-----------------------------|------|------------------------|
| CDNA2      | `gfx90a`         | Instinct MI200/MI210/MI250  | yes  | yes                    |
| CDNA3      | `gfx942`         | Instinct MI300              | yes  | yes                    |
| RDNA3      | `gfx1100`/`1101`/`1102` | RX 7900 XTX/XT/7800/7700 | yes  | yes (experimental)     |
| RDNA4      | `gfx1200`/`1201` | RX 9000-series              | yes  | varies                 |
| RDNA2      | `gfx1030`/`1031` | RX 6000-series              | no   | no                     |
| CDNA1      | `gfx908`         | MI100                       | no   | no                     |
| Vega20     | `gfx906`         | Radeon VII / MI50           | no   | no                     |

The bf16-capable allowlist is enforced in `dlm.hardware.capabilities`
based on `torch.cuda.get_device_properties(0).gcnArchName`.
Unsupported arches fall back to fp16 (still functional, just slower
per token on weight-heavy layers).

## What doesn't work

**QLoRA is refused on ROCm.** `bitsandbytes` ROCm builds are
upstream-unstable — 4-bit quantized matmuls silently return wrong
values on several arch/driver combinations. We refuse the
combination rather than risk corrupt gradients. Use `adapter: lora`
in your `.dlm` frontmatter.

**Multi-GPU ROCm** is out of scope for this sprint. Sprint 23's
multi-GPU work targets CUDA first; ROCm multi-GPU is a follow-on.

## Software prerequisites

- **ROCm** ≥ 5.7; 6.0+ preferred. We test against 6.0 and 6.2.
- **PyTorch** with HIP build — install via the ROCm wheels from
  pytorch.org. The `torch.version.hip` attribute must be non-None.
- **FlashAttention 2 (optional)**: AMD's `flash_attn` fork is the
  package name on ROCm. Install for CDNA (MI200/MI300); RDNA3
  support is experimental. If `flash_attn` is not importable or the
  arch is not on the allowlist, SDPA is used instead.

## Determinism posture

The doctor reports `determinism_class: best-effort` on ROCm. ROCm's
deterministic kernels exist but are not as thorough as CUDA's; fp
match may drift across PyTorch/ROCm upgrades even with a pinned
seed.

## Rebuilding llama.cpp with ROCm

The default vendored llama.cpp binary is CPU-only. Build a ROCm
version once for faster quantization:

```bash
# Set your GPU arch
export AMDGPU_TARGETS="gfx1100"   # RDNA3
# export AMDGPU_TARGETS="gfx90a"  # MI200
# export AMDGPU_TARGETS="gfx942"  # MI300

scripts/build-llama-cpp-rocm.sh
```

The script writes to `vendor/llama.cpp/build-rocm/`. To make
`dlm export` prefer this build, point the runner at it:

```bash
export DLM_LLAMA_CPP_BUILD=vendor/llama.cpp/build-rocm
```

(Environment-variable plumbing in `dlm.export.vendoring` lands as
part of the next ROCm polish pass — for now, manually invoke the
ROCm binaries if you need them.)

## CI / testing

No default ROCm CI runner exists. Contributors with ROCm hardware can
run the gated smoke:

```bash
DLM_ENABLE_ROCM_SMOKE=1 uv run pytest tests/integration/hardware/test_rocm_train_smoke.py -v
```

A scheduled self-hosted runner is the intended deployment; contact
the maintainers if you'd like to host one.
