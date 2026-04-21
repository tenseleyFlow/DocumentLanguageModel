# Adapter provenance — signing and verification

A signed adapter answers a question no downloader can otherwise
answer: **"did this adapter actually come from the document and
base I think it did, signed by someone I trust?"** The provenance
chain records the tuple `(adapter_sha256, base_revision,
corpus_root_sha256, env_lock_digest, signer_public_key, signature)`
inside the pack. `dlm verify <pack>` recomputes the pieces and
validates the minisign signature against your local trusted-keys
allowlist.

## Trust model

**TOFU** (trust-on-first-use). You maintain a local allowlist at
`~/.dlm/trusted-keys/*.pub`. The first time you verify an unknown
signer, you explicitly opt in with `--trust-on-first-use`; from
then on the key is in the allowlist and subsequent verifies run in
strict mode automatically.

This is weaker than a PKI but stronger than nothing: an attacker
who compromises the signer's key can publish malicious updates,
but can't impersonate *without* already having the key. And the
failure mode on key rotation is loud (a new fingerprint appears),
not silent.

Out of scope for this sprint:

- Key rotation, revocation, or CA chains. If a signer's key is
  compromised, you delete their `.pub` file and add the replacement
  manually.
- Automatic trust propagation across machines. Each host's
  `~/.dlm/trusted-keys/` is independent.

## Anatomy of a provenance.json

```json
{
  "adapter_sha256": "a1b2c3...64 hex chars...",
  "base_revision":  "fffb8e5...40 hex git SHA...",
  "corpus_root_sha256": "e5f6...64 hex...",
  "env_lock_digest":    "d4e5...64 hex sha over dlm.lock...",
  "signed_at":       "2026-04-21T12:00:00Z",
  "signer_public_key": "untrusted comment: minisign public key ...\nRWS...==",
  "signature":         "untrusted comment: signature from minisign\n..."
}
```

The signature covers canonical JSON (sort-keys, compact
separators) over the six non-signature fields. `dlm.share.provenance.
chain_bytes()` returns exactly those bytes — use it when signing,
not `json.dumps` with default arguments.

## Workflow (verifier side)

```bash
# 1. First-time receive a pack from Alice.
dlm verify alice-formality.dlm.pack --trust-on-first-use
# > verified: alice-formality.dlm.pack
# >   signer:          a1b2c3d4e5f6
# >   trusted-key:     /Users/you/.dlm/trusted-keys/a1b2c3d4e5f6.pub
# >   adapter_sha256:  abcd123...
# > note: recorded new trust entry at /Users/you/.dlm/trusted-keys/a1b2c3d4e5f6.pub;
# >       subsequent verifies use strict mode.

# 2. Alice updates the pack, ships a new version.
dlm verify alice-formality-v2.dlm.pack
# > verified: alice-formality-v2.dlm.pack
# >   signer:          a1b2c3d4e5f6      <- same fingerprint, silently verified

# 3. Someone else re-packages Alice's work with a different
# adapter, but forgets to re-sign.
dlm verify mallory-tampered.dlm.pack
# > verify: chain broken: adapter_sha256 mismatch: provenance
# > claims <abcd...> but pack contents hash to <xyz9...>.
# > (exit 1)

# 4. Brand-new signer, no --trust-on-first-use flag.
dlm verify bob-styletuning.dlm.pack
# > verify: signer fingerprint b0b1c2d3e4f5 not in
# > /Users/you/.dlm/trusted-keys/; re-run with `--trust-on-first-use`
# > to record, or add the key manually and retry.
# > (exit 2)
```

## Exit codes

| Code | Meaning | When it fires |
|---|---|---|
| 0 | Verified | Chain recomputed, signature valid, signer in allowlist |
| 1 | Chain broken | Missing provenance.json, malformed JSON, adapter hash mismatch |
| 2 | Untrusted signer | Signer not in allowlist AND `--trust-on-first-use` not set |
| 3 | Signature rejected | Minisign verify failed (signature bytes don't match the chain) |

Map to CI with:

```bash
dlm verify "$pack" || case $? in
  1) echo "broken chain, refusing to install"; exit 1 ;;
  2) echo "unknown signer; run locally with --trust-on-first-use first"; exit 1 ;;
  3) echo "signature forged or corrupted"; exit 1 ;;
esac
```

## Workflow (signer side) — deferred

Signing isn't yet wired into `dlm push`. For v1, signers compose
the provenance.json manually:

```python
from dlm.share.provenance import Provenance, dump_provenance_json, iso_utc_now

prov = Provenance(
    adapter_sha256="...",           # sha256 over your adapter safetensors
    base_revision="...",            # git SHA of the base model commit you trained from
    corpus_root_sha256="...",       # sha256 over your .dlm file bytes
    env_lock_digest="...",          # sha256 over the dlm.lock file
    signed_at=iso_utc_now(),
    signer_public_key=Path("~/.minisign/my-pubkey.pub").read_text(),
    signature="",  # fill in step 2
)

# Step 1: write the chain bytes for signing.
chain_bytes = prov.chain_bytes()
Path("provenance.chain").write_bytes(chain_bytes)

# Step 2: minisign -Sm provenance.chain (produces provenance.chain.minisig).
#   Fill the signature into the record and re-dump.
prov = dataclasses.replace(prov, signature=Path("provenance.chain.minisig").read_text())
dump_provenance_json(prov, Path("provenance.json"))

# Step 3: include provenance.json in your pack — for now, add it by hand
#   to the tar before zstd-compressing, or wait for the dlm push wiring.
```

Streamlining this flow into `dlm push --sign` is the next follow-
up; the math primitives + CLI verify path ship here so the
verifier side can go live immediately.

## What ships today

- `dlm.share.provenance`: `Provenance` dataclass, `canonical_json_bytes`,
  `dump_provenance_json` / `load_provenance_json`, `iso_utc_now`.
- TOFU helpers: `pubkey_fingerprint`, `record_trusted_key`,
  `find_matching_trusted_key`.
- `verify_provenance(prov, ..., tofu=...)` — orchestrates chain
  lookup, signature verify via `minisign -V`, and TOFU recording.
- `recompute_chain_consistency(prov, adapter_sha256=...)` — helper
  for callers who've independently computed the adapter sha and
  want to cross-check the chain.
- Pack layout: `PROVENANCE_FILENAME = "provenance.json"` (optional
  top-level member).
- Pack reader: `read_pack_member_bytes(pack_path, member_name)` —
  lightweight single-member extraction without full unpack.
- `dlm verify <pack> [--trust-on-first-use]` CLI with 0/1/2/3 exit
  codes.

## Deferred

- **`dlm push --sign`** — interactive signing flow. Users currently
  hand-compose provenance.json.
- **Pack writer integration.** The packer (`dlm pack`) doesn't yet
  include provenance.json automatically. Hand-add it before pack
  assembly, or wait for the `--sign` follow-up.
- **Adapter-sha cross-check in verify.** `recompute_chain_consistency`
  is implemented but the CLI doesn't call it yet (would require
  reading the actual adapter bytes from the pack). Tracked as a
  T-future.
- **Provenance on unpacked store.** When `dlm pull` lands a pack
  locally, provenance.json should propagate to the store alongside
  the adapter for future re-verify. Today it lives only in the pack.
