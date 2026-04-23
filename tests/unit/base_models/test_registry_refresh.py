"""Live-drift helper coverage for registry refresh."""

from __future__ import annotations

from types import SimpleNamespace

from dlm.base_models.registry_refresh import Drift, check_entry
from dlm.base_models.schema import BaseModelSpec


def _spec(**overrides: object) -> BaseModelSpec:
    defaults: dict[str, object] = {
        "key": "demo-1b",
        "hf_id": "org/demo-1b",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "architecture": "DemoForCausalLM",
        "params": 1_000_000_000,
        "target_modules": ["q_proj", "v_proj"],
        "template": "chatml",
        "gguf_arch": "demo",
        "tokenizer_pre": "demo",
        "license_spdx": "Apache-2.0",
        "redistributable": True,
        "size_gb_fp16": 2.0,
        "context_length": 4096,
        "recommended_seq_len": 2048,
    }
    defaults.update(overrides)
    return BaseModelSpec.model_validate(defaults)


class _Api:
    def __init__(self, *, sha: str, gated: object = False) -> None:
        self._info = SimpleNamespace(sha=sha, gated=gated)

    def model_info(self, _hf_id: str) -> SimpleNamespace:
        return self._info


class TestCheckEntry:
    def test_no_drift_when_revision_and_gating_match(self) -> None:
        spec = _spec()
        drift = check_entry(_Api(sha=spec.revision), spec)
        assert drift is None

    def test_revision_drift_is_reported(self) -> None:
        spec = _spec()
        drift = check_entry(_Api(sha="a" * 40), spec)
        assert isinstance(drift, Drift)
        assert ("revision", spec.revision, "a" * 40) in drift.fields

    def test_gating_drift_is_skipped_when_entry_opts_out(self) -> None:
        spec = _spec(
            requires_acceptance=True,
            refresh_check_hf_gating=False,
            provenance_url="https://example.com/provenance",
            provenance_match_text="official marker",
        )
        drift = check_entry(
            _Api(sha=spec.revision, gated=False),
            spec,
            fetch_url_text=lambda _url: "official marker",
        )
        assert drift is None

    def test_provenance_marker_missing_is_reported(self) -> None:
        spec = _spec(
            refresh_check_hf_gating=False,
            provenance_url="https://example.com/provenance",
            provenance_match_text="official marker",
        )
        drift = check_entry(
            _Api(sha=spec.revision),
            spec,
            fetch_url_text=lambda _url: "different text",
        )
        assert isinstance(drift, Drift)
        assert (
            "provenance_marker",
            "official marker",
            "missing from https://example.com/provenance",
        ) in drift.fields
