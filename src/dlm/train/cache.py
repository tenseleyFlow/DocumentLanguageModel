"""Process-level disable flag for the tokenized-section cache.

The trainer's pre-tokenize pass normally routes through
`TokenizedCache`. Two operator switches can disable it for a given
run:

- `training.cache.enabled: false` in frontmatter (per-document).
- `dlm train --no-cache` at the CLI (per-invocation), which sets
  `DLM_DISABLE_TOKENIZED_CACHE=1` on the trainer's environment.

Both meet at `is_cache_disabled()` in the trainer's pretokenize
helper. This module centralizes the env-var name + the imperative
set/unset so the CLI stops poking `os.environ` directly and future
scope changes (say, a subprocess-only override) have one place to
land.

The env-var survives `accelerate launch` re-invocations because the
child inherits the parent's environment — that's the point; rolling
the flag into `TrainingPlan` is deferred until we actually need
per-worker overrides.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator

_LOG = logging.getLogger(__name__)

DISABLE_ENV_VAR = "DLM_DISABLE_TOKENIZED_CACHE"


def is_cache_disabled() -> bool:
    """True iff the tokenized cache is turned off for this process."""
    return os.environ.get(DISABLE_ENV_VAR, "0") == "1"


def set_disable_flag(reason: str) -> None:
    """Turn the tokenized cache off for the rest of this process.

    `reason` is logged once so downstream test failures or slow-run
    investigations can trace why the cache path was skipped.
    """
    _LOG.info("tokenized cache disabled (%s)", reason)
    os.environ[DISABLE_ENV_VAR] = "1"


@contextlib.contextmanager
def disabled_cache(reason: str) -> Iterator[None]:
    """Scope-cache-disable to a `with` block.

    Restores the prior env-var value on exit — relevant for tests +
    REPL use where the caller doesn't want the flag to leak to
    subsequent commands.
    """
    prior = os.environ.get(DISABLE_ENV_VAR)
    set_disable_flag(reason)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(DISABLE_ENV_VAR, None)
        else:
            os.environ[DISABLE_ENV_VAR] = prior
