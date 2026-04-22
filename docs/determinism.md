# Determinism & reproducibility

DLM treats determinism as a contract: same input → same adapter SHA.
The contract is enforced by `src/dlm/lock/` (Sprint 15), backed by a
golden integration test, and surfaced to users via three CLI flags.

## The contract

Given:

- the same `.dlm` source text (SHA-256 match),
- the same base model revision,
- the same pinned versions (torch, transformers, peft, trl,
  bitsandbytes, accelerate, llama.cpp tag),
- the same hardware tier,
- the same seed and determinism flags,

training produces a byte-identical `adapter_model.safetensors`.

Proved by `tests/integration/lock/test_determinism_golden.py`, which
runs two fresh training cycles on the tiny model and asserts the
adapter SHAs match. Approved tuple goldens are tracked at the repo
level in `.determinism/lock.json`.

## What's in `dlm.lock`

Each store has a `dlm.lock` next to `manifest.json`:

```json
{
  "lock_version": 1,
  "created_at": "2026-04-19T17:30:00",
  "dlm_id": "01HRZYQ2X0MB5K4VN7E9DNT5GH",
  "dlm_sha256": "0123…ef",
  "base_model_revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
  "base_model_sha256": null,
  "pinned_versions": {
    "torch": "2.5.1",
    "transformers": "4.46.2",
    "peft": "0.14.0",
    "trl": "0.12.2",
    "bitsandbytes": "0.45.0"
  },
  "cuda_version": null,
  "rocm_version": null,
  "hardware_tier": "mps",
  "seed": 42,
  "determinism_flags": {},
  "determinism_class": "best-effort",
  "license_acceptance": null,
  "last_run_id": 3
}
```

Validated on every `dlm train`; written on success.

## Mismatch severity table

When the live runtime diverges from the recorded lock, each field is
classified:

| Field | Severity | Policy |
|---|---|---|
| `dlm_sha256` | ALLOW | Editing the doc is the point of DLM. |
| `base_model_revision` | ERROR | Breaks reproducibility; requires `--update-lock` to accept. |
| `torch` major version | ERROR | |
| `torch` minor/patch | WARN | |
| `transformers` / `peft` / `trl` / `accelerate` / `llama_cpp` | WARN | |
| `bitsandbytes` any | WARN | QLoRA kernels are version-sensitive. |
| `hardware_tier` | WARN | Re-plan recommended. |
| `determinism_class` | WARN | |
| `determinism_flags` | WARN | |

WARN mismatches print to stderr but don't block the run. ERROR
mismatches raise `LockValidationError` → exit code 1 with runbook
hints.

## CLI flags

| Flag | Behavior |
|---|---|
| *(default)* | Validate; abort on ERROR, warn on WARN, proceed + write. |
| `--strict-lock` | Upgrade every WARN to ERROR. |
| `--update-lock` | Skip validation, always write. For intentional drift acceptance. |
| `--ignore-lock` | Skip validation, don't write. For experimentation; the lock on disk stays stale. |

The three flags are mutually exclusive. See [CLI reference](cli/reference.md).

## Determinism tiers

The `determinism_class` field records what tier the host supports:

- **`strong`** — CUDA with all deterministic kernels available. Bit-exact
  reproduction expected across runs.
- **`best-effort`** — MPS, ROCm, or CUDA without the full deterministic
  kernel set. Loss curves are close but not bit-identical.
- **`advisory`** — CPU-only or a configuration where DLM refuses to
  promise determinism (some MPS ops fall here).

The golden integration test runs on CPU (tier `advisory`) and still
passes because SmolLM2-135M doesn't exercise the nondeterministic
kernels. On larger bases the CPU tier stops being bit-exact; that's
honest and documented.

## Regenerating the golden

When a pinned version changes deliberately (dep bump, llama.cpp tag
move), the recorded adapter SHA must be refreshed:

```sh
# Dry run — report the old vs new SHA without writing.
$ uv run python scripts/regen-determinism-golden.py

# Review the diff; then approve:
$ uv run python scripts/regen-determinism-golden.py --approve
```

The script:

1. Samples `capture_runtime_versions()` to produce the current tuple.
2. Runs the tiny-model training twice; confirms the two SHAs match.
3. Writes `tests/golden/determinism/tuple-<hash>.json` keyed by a
   SHA-256 of the sorted version tuple + platform.
4. Upserts `.determinism/lock.json` with the tuple path, adapter SHA,
   platform, and pinned versions.

Each tuple gets its own golden; the tuple file is keyed by content so
running on a new platform simply writes a new golden file. The repo-level
index keeps the checked-in set explicit and avoids overloading the
per-store `dlm.lock` name with a second meaning. The reviewer checks in
the tuple file and the index update alongside the dep bump.

## Non-goals

- **Byte-exact reproducibility from pure source.** DLM's replay corpus
  carries prior-run signal. Reconstructing a specific adapter without
  its replay history isn't possible — use `dlm pack` to archive.
- **Airgapped reproducibility.** The first `dlm train` against a new
  base pulls from HuggingFace. Subsequent runs use the local cache.
  We don't currently ship a fully-offline path; `--include-base` on
  `dlm pack` is the workaround.
- **MPS bit-exactness for large bases.** Apple's Metal kernels aren't
  deterministic for every op we use; the `best-effort` tier is an
  honest label, not a TODO.
