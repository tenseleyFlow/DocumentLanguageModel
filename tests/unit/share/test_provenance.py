"""Provenance chain: dataclass + digest + TOFU + verify."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dlm.share.provenance import (
    Provenance,
    ProvenanceChainBroken,
    ProvenanceSchemaError,
    ProvenanceVerifyResult,
    UnknownSignerError,
    canonical_json_bytes,
    dump_provenance_json,
    find_matching_trusted_key,
    iso_utc_now,
    load_provenance_json,
    pubkey_fingerprint,
    recompute_chain_consistency,
    record_trusted_key,
    verify_provenance,
)

_SAMPLE_PUBKEY = (
    "untrusted comment: minisign public key ABCDEF1234567890\n"
    "RWSABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmno+/=\n"
).strip()


def _sample_provenance(**overrides: str) -> Provenance:
    base = {
        "adapter_sha256": "a" * 64,
        "base_revision": "b" * 40,
        "corpus_root_sha256": "c" * 64,
        "env_lock_digest": "d" * 64,
        "signed_at": "2026-04-21T12:00:00Z",
        "signer_public_key": _SAMPLE_PUBKEY,
        "signature": "untrusted comment: signature\nabcxyz=\ntrusted comment: signed",
    }
    base.update(overrides)
    return Provenance(**base)  # type: ignore[arg-type]


class TestCanonicalJsonBytes:
    def test_sorts_keys(self) -> None:
        out_a = canonical_json_bytes({"b": "2", "a": "1"})
        out_b = canonical_json_bytes({"a": "1", "b": "2"})
        assert out_a == out_b

    def test_compact_separators(self) -> None:
        # No whitespace — signature determinism depends on this.
        out = canonical_json_bytes({"a": "1", "b": "2"})
        assert b" " not in out
        assert b"\n" not in out

    def test_utf8_preserved_roundtrip(self) -> None:
        out = canonical_json_bytes({"a": "1"})
        assert json.loads(out.decode("utf-8")) == {"a": "1"}


class TestProvenanceDigest:
    def test_chain_bytes_excludes_signature(self) -> None:
        prov = _sample_provenance()
        fields = json.loads(prov.chain_bytes().decode("utf-8"))
        assert "signature" not in fields
        # Sanity — the non-signature fields ARE present.
        assert fields["adapter_sha256"] == prov.adapter_sha256

    def test_chain_digest_is_deterministic(self) -> None:
        prov_a = _sample_provenance()
        prov_b = _sample_provenance()
        assert prov_a.compute_chain_digest() == prov_b.compute_chain_digest()

    def test_chain_digest_changes_on_any_field(self) -> None:
        base = _sample_provenance().compute_chain_digest()
        for field, new in (
            ("adapter_sha256", "z" * 64),
            ("base_revision", "z" * 40),
            ("corpus_root_sha256", "z" * 64),
            ("env_lock_digest", "z" * 64),
            ("signed_at", "2025-01-01T00:00:00Z"),
            ("signer_public_key", "different-key"),
        ):
            alt = _sample_provenance(**{field: new}).compute_chain_digest()
            assert alt != base, f"{field} change didn't affect digest"

    def test_signature_change_does_not_change_digest(self) -> None:
        """The digest is over the SIGNED fields — the signature itself
        is not part of what gets hashed, or verify would be circular."""
        base = _sample_provenance().compute_chain_digest()
        alt = _sample_provenance(signature="different-sig").compute_chain_digest()
        assert base == alt


class TestJsonIO:
    def test_roundtrip(self, tmp_path: Path) -> None:
        prov = _sample_provenance()
        path = tmp_path / "provenance.json"
        dump_provenance_json(prov, path)
        loaded = load_provenance_json(path)
        assert loaded == prov

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ProvenanceSchemaError, match="not found"):
            load_provenance_json(tmp_path / "does-not-exist.json")

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{ not json", encoding="utf-8")
        with pytest.raises(ProvenanceSchemaError, match="unreadable"):
            load_provenance_json(path)

    def test_missing_field_raises_with_names(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.json"
        path.write_text(
            json.dumps({"adapter_sha256": "x" * 64, "signature": "sig"}),
            encoding="utf-8",
        )
        with pytest.raises(ProvenanceSchemaError, match="missing required fields"):
            load_provenance_json(path)

    def test_non_string_field_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "typed.json"
        payload = {
            "adapter_sha256": 12345,  # int, not str
            "base_revision": "b" * 40,
            "corpus_root_sha256": "c" * 64,
            "env_lock_digest": "d" * 64,
            "signed_at": "2026-04-21T12:00:00Z",
            "signer_public_key": "key",
            "signature": "sig",
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ProvenanceSchemaError, match="adapter_sha256"):
            load_provenance_json(path)

    def test_non_object_root_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "array.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ProvenanceSchemaError, match="JSON object"):
            load_provenance_json(path)


class TestIsoUtcNow:
    def test_format_matches_signed_at(self) -> None:
        s = iso_utc_now()
        # Pattern: `YYYY-MM-DDTHH:MM:SSZ`
        assert s.endswith("Z")
        assert s[4] == "-"
        assert s[7] == "-"
        assert s[10] == "T"
        assert s[13] == ":"
        assert s[16] == ":"


class TestTrustedKeyRegistry:
    def test_fingerprint_is_stable(self) -> None:
        assert pubkey_fingerprint(_SAMPLE_PUBKEY) == pubkey_fingerprint(_SAMPLE_PUBKEY)
        # sha256 first 12 hex chars.
        expected = hashlib.sha256(_SAMPLE_PUBKEY.encode("utf-8")).hexdigest()[:12]
        assert pubkey_fingerprint(_SAMPLE_PUBKEY) == expected

    def test_record_creates_file_with_fingerprint_name(self, tmp_path: Path) -> None:
        target = record_trusted_key(_SAMPLE_PUBKEY, trusted_keys_dir=tmp_path)
        assert target.name.endswith(".pub")
        assert pubkey_fingerprint(_SAMPLE_PUBKEY) in target.name

    def test_record_with_label(self, tmp_path: Path) -> None:
        target = record_trusted_key(_SAMPLE_PUBKEY, trusted_keys_dir=tmp_path, label="alice")
        assert target.name.startswith("alice-")

    def test_record_is_idempotent(self, tmp_path: Path) -> None:
        first = record_trusted_key(_SAMPLE_PUBKEY, trusted_keys_dir=tmp_path)
        second = record_trusted_key(_SAMPLE_PUBKEY, trusted_keys_dir=tmp_path)
        assert first == second

    def test_find_matching_returns_path(self, tmp_path: Path) -> None:
        record_trusted_key(_SAMPLE_PUBKEY, trusted_keys_dir=tmp_path)
        found = find_matching_trusted_key(_SAMPLE_PUBKEY, trusted_keys_dir=tmp_path)
        assert found is not None

    def test_find_matching_returns_none_on_miss(self, tmp_path: Path) -> None:
        found = find_matching_trusted_key(_SAMPLE_PUBKEY, trusted_keys_dir=tmp_path)
        assert found is None

    def test_find_matching_handles_missing_dir(self, tmp_path: Path) -> None:
        found = find_matching_trusted_key(
            _SAMPLE_PUBKEY, trusted_keys_dir=tmp_path / "does-not-exist"
        )
        assert found is None


class TestVerifyProvenance:
    def _stub_verifier_accepts(self, chain: bytes, signature: str, pubkey_path: Path) -> None:
        """Pretend-verifier that always succeeds."""

    def _stub_verifier_rejects(self, chain: bytes, signature: str, pubkey_path: Path) -> None:
        """Pretend-verifier that always refuses."""
        from dlm.share.errors import ShareError

        raise ShareError("stub: refusing")

    def test_verified_happy_path(self, tmp_path: Path) -> None:
        record_trusted_key(_SAMPLE_PUBKEY, trusted_keys_dir=tmp_path)
        prov = _sample_provenance()
        result = verify_provenance(
            prov,
            trusted_keys_dir=tmp_path,
            signature_verifier=self._stub_verifier_accepts,
        )
        assert isinstance(result, ProvenanceVerifyResult)
        assert result.verified is True
        assert result.tofu_recorded is False
        assert result.signer_fingerprint == pubkey_fingerprint(_SAMPLE_PUBKEY)

    def test_unknown_signer_strict_raises(self, tmp_path: Path) -> None:
        prov = _sample_provenance()
        with pytest.raises(UnknownSignerError, match=pubkey_fingerprint(_SAMPLE_PUBKEY)):
            verify_provenance(
                prov,
                trusted_keys_dir=tmp_path,
                signature_verifier=self._stub_verifier_accepts,
            )

    def test_unknown_signer_tofu_records(self, tmp_path: Path) -> None:
        prov = _sample_provenance()
        result = verify_provenance(
            prov,
            trusted_keys_dir=tmp_path,
            tofu=True,
            signature_verifier=self._stub_verifier_accepts,
        )
        assert result.verified is True
        assert result.tofu_recorded is True
        # Second verify under TOFU is now just a regular match.
        second = verify_provenance(
            prov,
            trusted_keys_dir=tmp_path,
            tofu=True,
            signature_verifier=self._stub_verifier_accepts,
        )
        assert second.tofu_recorded is False

    def test_bad_signature_raises(self, tmp_path: Path) -> None:
        from dlm.share.errors import ShareError

        record_trusted_key(_SAMPLE_PUBKEY, trusted_keys_dir=tmp_path)
        prov = _sample_provenance()

        with pytest.raises(ShareError):
            verify_provenance(
                prov,
                trusted_keys_dir=tmp_path,
                signature_verifier=self._stub_verifier_rejects,
            )


class TestChainConsistency:
    def test_matching_sha_passes(self) -> None:
        prov = _sample_provenance(adapter_sha256="a" * 64)
        recompute_chain_consistency(prov, adapter_sha256="a" * 64)

    def test_mismatched_sha_raises(self) -> None:
        prov = _sample_provenance(adapter_sha256="a" * 64)
        with pytest.raises(ProvenanceChainBroken, match="mismatch"):
            recompute_chain_consistency(prov, adapter_sha256="b" * 64)
