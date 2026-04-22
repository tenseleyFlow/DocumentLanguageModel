"""`--gpus` CLI flag parsing.

Three accepted shapes:

- `all` — use every visible GPU (requires `torch.cuda.device_count()`
  at resolve time; parsing defers the probe so tests don't need torch).
- `N` — an integer string, use the first N device IDs.
- `0,1,2` — a comma-separated list of explicit device IDs.

Parsing is pure string logic. `GpuSpec.resolve(device_count)` turns the
spec into a concrete device-id list, which is what the launcher
passes through `CUDA_VISIBLE_DEVICES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


class UnsupportedGpuSpecError(ValueError):
    """Raised when `--gpus` can't be parsed or resolved."""


def strip_gpus_flag(args: list[str], *, skip_argv0: bool = False) -> list[str]:
    """Drop `--gpus <value>` / `--gpus=<value>` from an argv-like list.

    Shared helper so the launcher side (strips argv[0] because
    `accelerate launch -m <entry>` substitutes it) and the worker
    side (passes argv[1:] from `sys.argv`) don't drift. The
    `skip_argv0` flag controls which input convention is used.
    """
    start = 1 if skip_argv0 else 0
    out: list[str] = []
    skip_next = False
    for token in args[start:]:
        if skip_next:
            skip_next = False
            continue
        if token == "--gpus":
            skip_next = True
            continue
        if token.startswith("--gpus="):
            continue
        out.append(token)
    return out


GpuSpecKind = Literal["all", "count", "list"]


@dataclass(frozen=True)
class GpuSpec:
    """Parsed representation of the `--gpus` CLI flag.

    One of three shapes:

    - kind=`"all"`, value=None: use every visible GPU.
    - kind=`"count"`, value=int: use the first N (0..N-1).
    - kind=`"list"`, value=tuple[int, ...]: use exactly these device ids.
    """

    kind: GpuSpecKind
    value: int | tuple[int, ...] | None

    def resolve(self, device_count: int) -> tuple[int, ...]:
        """Turn the spec into an explicit, validated device-id tuple.

        `device_count` is the actual number of CUDA devices visible on
        the host (typically `torch.cuda.device_count()`). We raise
        rather than silently clip if the spec over-reaches — a user
        asking for `--gpus 4` on a 2-GPU box should get a clear error,
        not a 2-GPU run.
        """
        if device_count < 1:
            raise UnsupportedGpuSpecError(
                f"--gpus requires at least 1 CUDA device; `device_count={device_count}`"
            )
        if self.kind == "all":
            return tuple(range(device_count))
        if self.kind == "count":
            assert isinstance(self.value, int)
            n = self.value
            if n < 1:
                raise UnsupportedGpuSpecError(f"--gpus {n} is invalid; must be >= 1")
            if n > device_count:
                raise UnsupportedGpuSpecError(
                    f"--gpus {n} exceeds visible device count ({device_count})"
                )
            return tuple(range(n))
        # kind == "list"
        assert isinstance(self.value, tuple)
        out_of_range = [d for d in self.value if d < 0 or d >= device_count]
        if out_of_range:
            raise UnsupportedGpuSpecError(
                f"--gpus {list(self.value)} contains out-of-range device ids "
                f"(visible count={device_count})"
            )
        if len(set(self.value)) != len(self.value):
            raise UnsupportedGpuSpecError(
                f"--gpus {list(self.value)} contains duplicate device ids"
            )
        return tuple(self.value)


def parse_gpus(raw: str) -> GpuSpec:
    """Parse the raw `--gpus` string.

    Accepts:
    - `"all"` (case-insensitive)
    - `"N"` — integer ≥ 1
    - `"a,b,c"` — comma-separated device ids (each ≥ 0)

    Whitespace is tolerated. Empty string and malformed inputs raise
    `UnsupportedGpuSpecError`.
    """
    if raw is None:
        raise UnsupportedGpuSpecError("--gpus is empty")
    text = raw.strip()
    if not text:
        raise UnsupportedGpuSpecError("--gpus is empty")
    if text.lower() == "all":
        return GpuSpec(kind="all", value=None)
    if "," in text:
        parts = [p.strip() for p in text.split(",")]
        try:
            ids = tuple(int(p) for p in parts if p)
        except ValueError as exc:
            raise UnsupportedGpuSpecError(f"--gpus list {raw!r} has non-integer entries") from exc
        if not ids:
            raise UnsupportedGpuSpecError(f"--gpus list {raw!r} is empty")
        if any(i < 0 for i in ids):
            raise UnsupportedGpuSpecError(f"--gpus list {raw!r} has negative device ids")
        return GpuSpec(kind="list", value=ids)
    # Bare integer → count.
    try:
        n = int(text)
    except ValueError as exc:
        raise UnsupportedGpuSpecError(
            f"--gpus {raw!r} is not `all`, an integer, or a comma-separated list"
        ) from exc
    return GpuSpec(kind="count", value=n)
