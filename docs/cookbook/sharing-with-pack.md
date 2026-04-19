# Sharing with `dlm pack`

A `.dlm.pack` is a single file that bundles everything another person
needs to reproduce your training: the `.dlm` source, the store's
manifest + training state, the adapter weights, optionally the GGUF
exports, optionally the base model.

## The minimal pack

```sh
$ uv run dlm pack tutor.dlm
wrote: tutor.dlm.pack (3.2 MB)
```

What ships in the default pack:

- `tutor.dlm` (the source)
- `manifest.json`
- `adapter/versions/v000N/` (the current adapter only)
- `training_state.pt` + `.sha256` (for bit-exact resume)
- `dlm.lock` (determinism contract)
- Per-file SHA-256 checksums + a `content_sha256` rollup (Sprint 14)

Not included by default: exports (you can regenerate them), the base
model (you already have it or can download it), logs (bulky, optional).

## Receiving a pack

```sh
$ uv run dlm unpack tutor.dlm.pack
unpacked: 01HRTUTOR… under ~/.dlm/store/
           tutor.dlm placed at ./tutor.dlm

$ uv run dlm prompt tutor.dlm "Explain decorators"
A decorator is a function that…
```

`dlm unpack` verifies every file's SHA-256 against the pack manifest
before writing — a corrupted pack aborts with the specific file that
failed.

## Including exports

```sh
$ uv run dlm pack tutor.dlm --include-exports
```

Bundles every GGUF under `exports/`. Useful when the recipient doesn't
have llama.cpp built; they can `dlm unpack` and go straight to
`ollama create`.

## Including the base model

```sh
$ uv run dlm pack tutor.dlm --include-base \
    --i-am-the-licensee https://example.com/our-license-acknowledgement
```

Bundles the base model weights into the pack. Required flag for gated
bases (Llama 3.2, Qwen 2.5 licenses that require each downloader to
accept separately) — the `--i-am-the-licensee URL` asserts that the
person you're sending to has also accepted the license at that URL.

Without `--include-base`, the recipient needs network + license
acceptance to reproduce. With it, the pack is a single self-contained
artifact (but larger — a 1.5B base adds ~3 GB).

## Integrity guarantees

Two packs of the same store produce **byte-identical** pack files
(Sprint 14 audit-06 B5). Under the hood:

- `mtime`, `uid`, `gid` inside the tar are zeroed
- File ordering is sorted by path
- Zstd compression level + frame parameters are pinned
- The manifest's `content_sha256` is a sorted-path rollup, so two
  identical stores produce identical rollups

This means `dlm pack` is a reproducible build; you can diff two packs
and confirm nothing leaked through (timestamps, username, etc.).

## Security

`dlm unpack` defends against tar and zstd bombs (Sprint 14 audit-06
B7):

- Maximum expanded size per entry + total
- Symlinks inside the pack are refused
- Duplicate paths in the pack manifest fail validation

A pack from an untrusted source will either unpack cleanly or fail
fast with a clear reason; there's no in-between where it partially
extracts onto your filesystem.

## Version compatibility

The pack format itself is versioned. Sprint 14 ships v1. Future
pack format bumps register a migrator (Sprint 14 migrations registry),
so a pack written by v2 can be read by a v1 reader with a migration
pass.
