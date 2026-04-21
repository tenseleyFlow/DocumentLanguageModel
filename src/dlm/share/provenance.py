"""Provenance chain for shared adapters.

A provenance record captures *what* an adapter was trained against
(base revision, corpus content hash, environment lock digest) and
*who* signed it (minisign public key + signature over the chain
bytes). `dlm verify <pack>` recomputes the chain, validates the
signature, and enforces a TOFU allowlist against
`~/.dlm/trusted-keys/`.

The signature covers the **canonical JSON** of the five
non-signature fields — callers that compute the digest locally and
sign externally must use `chain_bytes()` to get the exact bytes the
verifier will check, not whatever `json.dumps()` defaulted to.

Storage layout:

- In a pack: `provenance.json` at the top level (optional).
- On a store: deferred — the v1 scope lets users ship provenance
  by dropping the file next to their `.dlm.pack` and letting the
  packer include it on the next build.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SignatureVerifier = Callable[[bytes, str, Path], None]

from dlm.share.errors import ShareError


class ProvenanceError(ShareError):
    """Base for provenance-chain failures that aren't signature-verify."""


class ProvenanceChainBroken(ProvenanceError):  # noqa: N818
    """A field in the chain doesn't match the recomputed value.

    Typical causes: adapter bytes were edited after signing (the
    recorded `adapter_sha256` no longer matches); the pack was
    reassembled from a different base or corpus than the one the
    signer compiled against.
    """


class ProvenanceSchemaError(ProvenanceError):
    """The provenance.json file is malformed or missing required fields."""


@dataclass(frozen=True)
class Provenance:
    """Signed provenance record for a shared adapter.

    `signer_public_key` is the raw minisign public-key line
    (base64, prefixed with `untrusted comment: minisign public key:`).
    `signature` is the matching minisign signature block as a string;
    callers write it out to a `.minisig` file at verify time.

    `chain_bytes()` returns the exact bytes the signer should sign —
    canonical JSON (sort_keys, compact separators) over the five
    non-signature fields. `compute_chain_digest()` wraps that in
    sha256 for the fingerprint field.
    """

    adapter_sha256: str
    base_revision: str
    corpus_root_sha256: str
    env_lock_digest: str
    signed_at: str  # ISO-8601 UTC, second-precision
    signer_public_key: str
    signature: str

    def chain_fields(self) -> dict[str, str]:
        """Return the subset of fields covered by the signature."""
        return {
            "adapter_sha256": self.adapter_sha256,
            "base_revision": self.base_revision,
            "corpus_root_sha256": self.corpus_root_sha256,
            "env_lock_digest": self.env_lock_digest,
            "signed_at": self.signed_at,
            "signer_public_key": self.signer_public_key,
        }

    def chain_bytes(self) -> bytes:
        """Canonical JSON over the signature-covered fields."""
        return canonical_json_bytes(self.chain_fields())

    def compute_chain_digest(self) -> str:
        """sha256 hex of `chain_bytes()` — the fingerprint a verifier recomputes."""
        return hashlib.sha256(self.chain_bytes()).hexdigest()


def canonical_json_bytes(data: dict[str, str]) -> bytes:
    """Deterministic JSON encoding for signing.

    Uses `sort_keys=True` and `separators=(",", ":")` — matches what
    `minisign -Sm` will sign. Any downstream consumer recomputing
    the digest MUST use this function (not `json.dumps` with
    defaults) or the signature check fails.
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def dump_provenance_json(prov: Provenance, path: Path) -> None:
    """Write `prov` to `path` in the on-disk JSON shape.

    The output is pretty-printed for human readability — this is
    the *storage* format, not the signing format. The signer
    cryptographically covers `chain_bytes()`, which is the compact
    form. Pretty-printing the stored file doesn't affect verify.
    """
    payload = {
        **prov.chain_fields(),
        "signature": prov.signature,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_provenance_json(path: Path) -> Provenance:
    """Parse a provenance.json file into a `Provenance` record.

    Missing or unreadable file → `ProvenanceSchemaError`. Missing
    required fields or wrong types → same error with the offending
    key name.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProvenanceSchemaError(f"provenance file not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceSchemaError(f"provenance file at {path} is unreadable: {exc}") from exc

    if not isinstance(raw, dict):
        raise ProvenanceSchemaError(
            f"provenance.json must be a JSON object, got {type(raw).__name__}"
        )

    required = (
        "adapter_sha256",
        "base_revision",
        "corpus_root_sha256",
        "env_lock_digest",
        "signed_at",
        "signer_public_key",
        "signature",
    )
    missing = [k for k in required if k not in raw]
    if missing:
        raise ProvenanceSchemaError(
            f"provenance.json missing required fields: {', '.join(sorted(missing))}"
        )
    for key in required:
        if not isinstance(raw[key], str):
            raise ProvenanceSchemaError(
                f"provenance.json[{key!r}] must be a string, got {type(raw[key]).__name__}"
            )

    return Provenance(
        adapter_sha256=raw["adapter_sha256"],
        base_revision=raw["base_revision"],
        corpus_root_sha256=raw["corpus_root_sha256"],
        env_lock_digest=raw["env_lock_digest"],
        signed_at=raw["signed_at"],
        signer_public_key=raw["signer_public_key"],
        signature=raw["signature"],
    )


def iso_utc_now() -> str:
    """ISO-8601 UTC second-precision string — matches the `signed_at` format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- TOFU key registry -------------------------------------------------------


class UnknownSignerError(ProvenanceError):
    """Provenance is signed by a key not in the trusted-keys directory.

    Raised when verify runs in strict mode and the signer's public
    key doesn't match any `*.pub` file under `~/.dlm/trusted-keys/`.
    TOFU mode catches this error and records the key instead.
    """


def pubkey_fingerprint(public_key: str) -> str:
    """Short fingerprint (first 12 hex chars of sha256) for a pubkey string.

    Useful for CLI output — full key blocks are long and the
    fingerprint is what users compare against out-of-band trust
    sources.
    """
    return hashlib.sha256(public_key.encode("utf-8")).hexdigest()[:12]


def record_trusted_key(
    public_key: str,
    *,
    trusted_keys_dir: Path,
    label: str | None = None,
) -> Path:
    """Write `public_key` to the trusted-keys directory.

    Filename is `<fingerprint>.pub` (or `<label>-<fingerprint>.pub`
    when `label` is set). Idempotent: if a file with the same
    contents already exists, returns that path without rewriting.
    Raises `ProvenanceError` if a different key already occupies the
    target filename — refusing to clobber an existing trust entry.
    """
    trusted_keys_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = pubkey_fingerprint(public_key)
    filename = f"{label}-{fingerprint}.pub" if label else f"{fingerprint}.pub"
    target = trusted_keys_dir / filename

    canonical = public_key.strip() + "\n"
    if target.is_file():
        existing = target.read_text(encoding="utf-8").strip()
        if existing == public_key.strip():
            return target
        raise ProvenanceError(
            f"trust-key filename {target} exists with different contents; "
            "refusing to overwrite. Remove the file manually or use a different label."
        )
    target.write_text(canonical, encoding="utf-8")
    return target


def find_matching_trusted_key(
    public_key: str,
    *,
    trusted_keys_dir: Path,
) -> Path | None:
    """Return the `*.pub` file in `trusted_keys_dir` whose contents match.

    Compared against the stripped key body (whitespace-insensitive).
    `None` means the key isn't in the allowlist yet. Missing
    directory is treated as an empty allowlist.
    """
    if not trusted_keys_dir.is_dir():
        return None
    needle = public_key.strip()
    for candidate in sorted(trusted_keys_dir.glob("*.pub")):
        try:
            body = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if body == needle:
            return candidate
    return None


# --- verify -----------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceVerifyResult:
    """Outcome of `verify_provenance`.

    `trusted_key_path` is the `*.pub` file that matched the signer —
    populated when `verified=True`. `tofu_recorded` is True when the
    verifier ran in TOFU mode AND wrote a new trust entry on this
    call; callers surface it in CLI output so operators notice.
    `signer_fingerprint` is always set for pretty output, matched or
    not.
    """

    verified: bool
    signer_fingerprint: str
    trusted_key_path: Path | None = None
    tofu_recorded: bool = False
    detail: str = ""


def verify_provenance(
    prov: Provenance,
    *,
    trusted_keys_dir: Path,
    tofu: bool = False,
    signature_verifier: SignatureVerifier | None = None,
) -> ProvenanceVerifyResult:
    """Verify `prov` against the trusted-keys allowlist + its signature.

    Pipeline:

    1. Compute chain bytes from `prov.chain_fields()`.
    2. Resolve the signer: match `prov.signer_public_key` against
       `trusted_keys_dir/*.pub`. No match → record under TOFU, else
       raise `UnknownSignerError`.
    3. Invoke `signature_verifier(chain_bytes, signature, pubkey_path)`
       — defaults to `dlm.share.signing._minisign_verify` via a temp
       payload file. Tests inject a stub to avoid requiring minisign.

    Callers map exceptions to CLI exit codes: `ProvenanceChainBroken`
    → 1, `UnknownSignerError` → 2, any other `ProvenanceError` → 1.
    """
    chain = prov.chain_bytes()
    fingerprint = pubkey_fingerprint(prov.signer_public_key)

    trusted_path = find_matching_trusted_key(
        prov.signer_public_key, trusted_keys_dir=trusted_keys_dir
    )
    tofu_recorded = False
    if trusted_path is None:
        if not tofu:
            raise UnknownSignerError(
                f"signer fingerprint {fingerprint} not in {trusted_keys_dir}; "
                "re-run with `--trust-on-first-use` to record, or add the key "
                "manually and retry."
            )
        trusted_path = record_trusted_key(
            prov.signer_public_key, trusted_keys_dir=trusted_keys_dir
        )
        tofu_recorded = True

    verifier = signature_verifier or _default_signature_verifier
    verifier(chain, prov.signature, trusted_path)

    return ProvenanceVerifyResult(
        verified=True,
        signer_fingerprint=fingerprint,
        trusted_key_path=trusted_path,
        tofu_recorded=tofu_recorded,
        detail="",
    )


def _default_signature_verifier(
    chain_bytes: bytes, signature: str, pubkey_path: Path
) -> None:
    """Default signature verifier: writes chain + signature to temp files and
    calls `minisign -V`.

    Factored out so tests can inject a fake that skips the real
    minisign CLI — keeps unit tests offline + deterministic.
    """
    import tempfile

    from dlm.share.signing import _minisign_verify

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        payload = tmp / "provenance.chain"
        payload.write_bytes(chain_bytes)
        sig = tmp / "provenance.chain.minisig"
        sig.write_text(signature, encoding="utf-8")
        _minisign_verify(payload, sig, pubkey_path)


def recompute_chain_consistency(prov: Provenance, *, adapter_sha256: str) -> None:
    """Raise `ProvenanceChainBroken` if `prov.adapter_sha256` disagrees
    with the sha256 computed over the actual adapter bytes at verify
    time. Callers supply the fresh sha — we don't open files here,
    keeping the module side-effect-free for easy testing.
    """
    if prov.adapter_sha256 != adapter_sha256:
        raise ProvenanceChainBroken(
            f"adapter_sha256 mismatch: provenance claims {prov.adapter_sha256} "
            f"but pack contents hash to {adapter_sha256}. The adapter was "
            "modified after signing, or the pack was reassembled from a "
            "different source."
        )
