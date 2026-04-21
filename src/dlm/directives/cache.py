"""Per-store tokenized-section cache.

Avoid re-tokenizing unchanged directive-sourced files on every
`dlm train` run. At 50K+ files this is the difference between an
hour of retokenization and seconds of cache warm-up.

Layout (per store):

    ~/.dlm/store/<dlm_id>/tokenized-cache/
        manifest.json          { version, tokenizer_sha256, total_bytes,
                                 entries: {key_str: {size, last_access_ts,
                                                      shard, filename}} }
        entries/
            <section_id[:2]>/  sharded to avoid 50K files in one dir
                <...>.npz      numpy save of input_ids + attention_mask

Entries are keyed by `(section_id, tokenizer_sha, sequence_len)`. A
change to any input produces a new key — stale entries are garbage-
collected by `prune` when their last-access age exceeds the cutoff.

Atomicity: `put` writes to a tmp file + `os.replace`; the manifest
update is the last step, so a mid-put SIGTERM leaves no torn entries
(the tmp file may orphan, `prune` sweeps it).

LRU eviction fires on `put` when `total_bytes + incoming_size >
max_bytes`. Oldest `last_access_ts` wins; **current-run entries are
protected** so a cold cache doesn't self-starve.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dlm.directives.cache_key import CacheKey

_LOG = logging.getLogger(__name__)

_CACHE_VERSION = 1
_MANIFEST_FILENAME = "manifest.json"
_ENTRIES_DIR = "entries"
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024 * 1024  # 10 GiB


@dataclass(frozen=True)
class CachedTokens:
    """Tokenizer output pulled from the cache.

    `input_ids` and `attention_mask` are 1D numpy int arrays matching
    what `tokenizer(text, truncation=True, padding=False, max_length=seq_len)`
    would return. Callers convert to torch tensors at the dataloader
    boundary.
    """

    input_ids: np.ndarray
    attention_mask: np.ndarray


@dataclass
class _Entry:
    """Manifest row for one cached tokenization.

    `key_str` is the canonical string form of the CacheKey. `size`
    is bytes on disk (best-effort stat). `last_access_ts` is Unix
    seconds as a float.
    """

    key_str: str
    size: int
    last_access_ts: float
    shard: str
    filename: str
    tokenizer_sha: str


def _key_str(key: CacheKey) -> str:
    """Canonical str form: used as manifest dict key."""
    return f"{key.section_id}|{key.tokenizer_sha}|{key.sequence_len}"


class TokenizedCache:
    """Per-store tokenized-section cache.

    Open via `TokenizedCache.open(store.tokenized_cache_dir)`. The
    constructor eagerly loads the manifest (cheap: one JSON file with
    N entries). `get` and `put` touch disk for the actual tensors.
    """

    def __init__(
        self,
        root: Path,
        *,
        manifest: dict[str, _Entry],
        tokenizer_sha_hint: str | None = None,
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        self._root = root
        self._manifest = manifest
        self._tokenizer_sha_hint = tokenizer_sha_hint
        self._max_bytes = max_bytes
        # Entries inserted or touched during this session — protected
        # from LRU eviction to avoid cold-cache self-starvation.
        self._touched_this_run: set[str] = set()
        # Counters for end-of-run metrics.
        self._hits = 0
        self._misses = 0

    # ---- Open / construct --------------------------------------------

    @classmethod
    def open(
        cls, root: Path, *, max_bytes: int = _DEFAULT_MAX_BYTES
    ) -> TokenizedCache:
        """Open (or create) a cache at `root`.

        Creates the directory layout idempotently. Missing manifest →
        fresh empty cache. Corrupt manifest → log a WARN and start
        fresh, leaving any orphaned entry files to `prune` later.
        """
        root.mkdir(parents=True, exist_ok=True)
        (root / _ENTRIES_DIR).mkdir(exist_ok=True)
        manifest_path = root / _MANIFEST_FILENAME

        if not manifest_path.is_file():
            return cls(root=root, manifest={}, max_bytes=max_bytes)

        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning(
                "cache: manifest at %s unreadable (%s); starting fresh",
                manifest_path,
                exc,
            )
            return cls(root=root, manifest={}, max_bytes=max_bytes)

        if not isinstance(raw, dict) or raw.get("version") != _CACHE_VERSION:
            _LOG.warning(
                "cache: manifest version mismatch at %s; starting fresh",
                manifest_path,
            )
            return cls(root=root, manifest={}, max_bytes=max_bytes)

        entries_raw = raw.get("entries", {})
        if not isinstance(entries_raw, dict):
            return cls(root=root, manifest={}, max_bytes=max_bytes)

        manifest: dict[str, _Entry] = {}
        for key_str, row in entries_raw.items():
            if not isinstance(row, dict):
                continue
            try:
                manifest[key_str] = _Entry(
                    key_str=key_str,
                    size=int(row["size"]),
                    last_access_ts=float(row["last_access_ts"]),
                    shard=str(row["shard"]),
                    filename=str(row["filename"]),
                    tokenizer_sha=str(row.get("tokenizer_sha", "")),
                )
            except (KeyError, TypeError, ValueError) as exc:
                _LOG.warning("cache: skipping malformed entry %s: %s", key_str, exc)

        return cls(
            root=root,
            manifest=manifest,
            tokenizer_sha_hint=raw.get("tokenizer_sha256"),
            max_bytes=max_bytes,
        )

    # ---- Properties --------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def total_bytes(self) -> int:
        return sum(e.size for e in self._manifest.values())

    @property
    def entry_count(self) -> int:
        return len(self._manifest)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    # ---- Get / Put ---------------------------------------------------

    def get(self, key: CacheKey) -> CachedTokens | None:
        """Return cached tokens for `key` or None on miss."""
        key_str = _key_str(key)
        entry = self._manifest.get(key_str)
        if entry is None:
            self._misses += 1
            return None

        path = self._entry_path(entry)
        if not path.is_file():
            # Manifest drift (file deleted under us) — treat as miss,
            # remove the stale manifest row so we don't re-hit it.
            _LOG.warning("cache: entry file missing for %s; re-tokenizing", key_str)
            del self._manifest[key_str]
            self._misses += 1
            return None

        try:
            with np.load(path) as data:
                tokens = CachedTokens(
                    input_ids=np.array(data["input_ids"], copy=True),
                    attention_mask=np.array(data["attention_mask"], copy=True),
                )
        except (OSError, ValueError, KeyError) as exc:
            _LOG.warning("cache: corrupt entry %s (%s); re-tokenizing", key_str, exc)
            del self._manifest[key_str]
            self._misses += 1
            return None

        entry.last_access_ts = time.time()
        self._touched_this_run.add(key_str)
        self._hits += 1
        return tokens

    def put(self, key: CacheKey, tokens: CachedTokens) -> None:
        """Write `tokens` to the cache under `key`. Evicts if needed."""
        key_str = _key_str(key)
        shard = key.shard()
        filename = key.as_filename()
        shard_dir = self._root / _ENTRIES_DIR / shard
        shard_dir.mkdir(parents=True, exist_ok=True)
        final_path = shard_dir / filename
        tmp_path = shard_dir / f"{filename}.tmp"

        # Save to tmp, then atomic rename. np.savez_compressed writes
        # directly to the file handle, so we can't stream-then-rename
        # with a single call — open the tmp file manually.
        try:
            with tmp_path.open("wb") as f:
                np.savez_compressed(
                    f,
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                )
        except OSError as exc:
            _LOG.warning("cache: write failed for %s: %s; dropping entry", key_str, exc)
            tmp_path.unlink(missing_ok=True)
            return

        try:
            size = tmp_path.stat().st_size
        except OSError:
            size = 0

        # Evict BEFORE replacing so we have the budget headroom.
        self._evict_if_needed(incoming_bytes=size)

        tmp_path.replace(final_path)

        self._manifest[key_str] = _Entry(
            key_str=key_str,
            size=size,
            last_access_ts=time.time(),
            shard=shard,
            filename=filename,
            tokenizer_sha=key.tokenizer_sha,
        )
        self._touched_this_run.add(key_str)

    # ---- Eviction / Prune / Clear ------------------------------------

    def _evict_if_needed(self, *, incoming_bytes: int) -> None:
        """Delete oldest entries until (total + incoming) ≤ max_bytes.

        Current-run entries are protected: a cold cache won't evict
        what it just put in to make room for the next put.
        """
        budget = self._max_bytes - incoming_bytes
        if self.total_bytes <= budget:
            return

        # Candidates: entries not touched this run, sorted by age.
        candidates = sorted(
            (e for e in self._manifest.values() if e.key_str not in self._touched_this_run),
            key=lambda e: e.last_access_ts,
        )
        evicted = 0
        freed = 0
        for entry in candidates:
            if self.total_bytes <= budget:
                break
            path = self._entry_path(entry)
            path.unlink(missing_ok=True)
            del self._manifest[entry.key_str]
            evicted += 1
            freed += entry.size
        if evicted:
            _LOG.info(
                "cache: evicted %d entries (%d bytes) to stay under %d",
                evicted,
                freed,
                self._max_bytes,
            )

    def prune(self, *, older_than_seconds: float) -> int:
        """Delete entries whose `last_access_ts` is older than the cutoff.

        Returns the number of entries removed. Protected-set doesn't
        apply — `prune` is an explicit operator action, not a
        mid-put fallback.
        """
        cutoff = time.time() - older_than_seconds
        stale_keys = [
            e.key_str
            for e in self._manifest.values()
            if e.last_access_ts < cutoff
        ]
        for key_str in stale_keys:
            entry = self._manifest[key_str]
            self._entry_path(entry).unlink(missing_ok=True)
            del self._manifest[key_str]
        if stale_keys:
            _LOG.info("cache: pruned %d entries older than %ds", len(stale_keys), older_than_seconds)
        return len(stale_keys)

    def clear(self) -> int:
        """Delete every entry. Returns count removed."""
        count = len(self._manifest)
        for entry in list(self._manifest.values()):
            self._entry_path(entry).unlink(missing_ok=True)
        self._manifest.clear()
        self._touched_this_run.clear()
        return count

    # ---- Manifest persistence ----------------------------------------

    def save_manifest(self, *, tokenizer_sha: str | None = None) -> None:
        """Persist the manifest atomically.

        Call at the end of a training run (or on explicit CLI
        commands). `tokenizer_sha` is stored at the top level so
        future opens can detect a tokenizer bump before reading
        entries.
        """
        manifest_path = self._root / _MANIFEST_FILENAME
        tmp_path = manifest_path.with_suffix(".json.tmp")
        payload: dict[str, Any] = {
            "version": _CACHE_VERSION,
            "tokenizer_sha256": tokenizer_sha or self._tokenizer_sha_hint or "",
            "total_bytes": self.total_bytes,
            "entries": {
                e.key_str: {
                    "size": e.size,
                    "last_access_ts": e.last_access_ts,
                    "shard": e.shard,
                    "filename": e.filename,
                    "tokenizer_sha": e.tokenizer_sha,
                }
                for e in self._manifest.values()
            },
        }
        tmp_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(manifest_path)

    # ---- Helpers -----------------------------------------------------

    def _entry_path(self, entry: _Entry) -> Path:
        return self._root / _ENTRIES_DIR / entry.shard / entry.filename
