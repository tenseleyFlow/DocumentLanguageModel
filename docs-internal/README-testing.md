# Testing guide (contributor-facing)

Everything you need to run the test suite locally and understand what each
layer does.

## Layers

```
tests/
  test_smoke.py           package + CLI boot
  unit/                   fast, in-process, no network
  integration/            crosses 2+ modules (e.g. parser + store)
  e2e/                    full CLI against tmp stores
  fixtures/               factories + mocks (see below)
  golden/                 checked-in JSON goldens per (name, torch_version)
```

## Markers

| marker | meaning | default |
|---|---|---|
| (none) | fast unit, <1s each | run |
| `slow` | expensive; may load the tiny model | **skipped** |
| `gpu` | requires CUDA | skipped on CPU/MPS |
| `online` | touches the network (HF Hub) | skipped offline |

`pyproject.toml` sets `addopts = ["-m", "not slow and not gpu and not online"]`
so the default `uv run pytest` is always the fast, local subset.

## Running

```
uv run pytest                         # fast subset, default
uv run pytest -m slow                 # tiny-model and long-running paths
uv run pytest -m "slow and online"    # tiny-model download + inference
uv run pytest --update-goldens        # regenerate goldens (see below)
uv run pytest -v path/to/test_file.py # single-file verbose
```

## Fixtures

### `tests/fixtures/dlm_factory.py`

Builds synthetic `.dlm` text. Stable shape matching Sprint 03's parser.

```python
from tests.fixtures.dlm_factory import make_dlm, prose, instruction, preference

text = make_dlm(
    sections=[
        prose("# intro\n\nbody\n"),
        instruction(("Q1?", "A1."), ("Q2?", "A2.")),
        preference(("prompt", "good", "bad")),
    ],
    base_model="smollm2-135m",
    dlm_id="01HZ...",                # omit for a fresh ULID
    training_overrides={"lora_r": 16},
)
```

### `tests/fixtures/hardware_mocks.py`

Context managers for backend simulation without real hardware.

```python
from tests.fixtures.hardware_mocks import force_cuda, force_mps, force_cpu

with force_cuda(sm=(8, 9), vram_gb=24.0):
    # torch.cuda.is_available() is True, capability (8, 9), mem 24GB
    ...

with force_mps():
    # MPS is available; CUDA is not
    ...
```

Nesting works — the inner context is restored on exit.

### `tests/fixtures/tiny_model.py`

SmolLM2-135M-Instruct as a session-scoped fixture. Download is gated behind
`@pytest.mark.online`; the session-scoped `tiny_model_dir` fixture returns the
cached path.

```python
import pytest

@pytest.mark.online
@pytest.mark.slow
def test_something(tiny_model_dir):
    # tiny_model_dir is a pathlib.Path to the cached model
    ...
```

The revision is pinned via `DLM_TINY_MODEL_REVISION` (defaulting to `main`
until Sprint 06's base-model registry owns the SHA).

### `tests/fixtures/golden.py`

```python
from tests.fixtures.golden import assert_golden

def test_loss_curve():
    values = compute_loss_curve()
    assert_golden({"loss": values}, name="loss-curve-v1")
```

Goldens live at `tests/golden/<name>.torch-<version>.json`. Bumping torch
creates a new key; the old one stays until deliberately removed.

## Regenerating goldens

```
uv run pytest --update-goldens
```

This flips `assert_golden` into write mode. Review the diff before
committing:

```
git diff tests/golden/
```

A two-person review is mandatory for golden changes — they're determinism
contracts. See Sprint 15's `scripts/regen-determinism-golden.py` for the
heavier regeneration workflow once that lands.

## CI layout

Three GitHub Actions jobs:

1. **lint / typecheck / test** — ubuntu-latest + macos-latest matrix.
   Runs ruff, ruff format --check, mypy, default pytest selection.
2. **no-network sandbox** — ubuntu-latest. Blocks egress via iptables,
   then runs the local-only CLI surfaces (`dlm --version`, `--help`,
   and later `init`/`doctor`/`show`). Asserts the "no telemetry, ever"
   promise.
3. **slow tests (hf-cache)** — ubuntu-latest. Restores HF cache keyed
   on `(pyproject.toml hash, TINY_MODEL_REVISION)`, pre-warms the tiny
   model, then runs `pytest -m slow`.

## Offline-first autouse

`tests/conftest.py` sets `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` +
`HF_DATASETS_OFFLINE=1` via an autouse fixture. The `tiny_model_dir`
fixture temporarily clears these for its scope when an online test opts
in. This means a test that *accidentally* touches HF without the fixture
will fail fast instead of downloading silently.

## Common pitfalls

- **Importing torch in test collection is slow** (~5s). Fixtures that
  need it import lazily inside functions.
- **Hardware mocks don't simulate actual CUDA computation.** They only
  toggle `is_available`-shaped attributes. Tests that need a real GPU use
  the `gpu` marker.
- **Golden drift on torch bumps is expected.** Regeneration is the fix;
  review the old vs new checksum side-by-side before approval.
