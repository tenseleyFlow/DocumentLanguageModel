"""Live-drift helper coverage for registry refresh."""

from __future__ import annotations

from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from dlm.base_models.registry_refresh import Drift, check_entry, check_registry, fetch_text
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


class _RaisingApi:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def model_info(self, _hf_id: str) -> SimpleNamespace:
        raise self._exc


class _FakeResponse:
    def __init__(self, body: bytes, charset: str = "utf-8") -> None:
        self._body = body
        self.headers = SimpleNamespace(get_content_charset=lambda: charset)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


class TestDrift:
    def test_render_formats_each_field_on_its_own_line(self) -> None:
        drift = Drift(
            key="demo-1b",
            hf_id="org/demo-1b",
            fields=(("revision", "old", "new"), ("gating", "False", "True")),
        )
        assert drift.render() == (
            "  demo-1b (org/demo-1b)\n"
            "    revision               'old' → 'new'\n"
            "    gating                 'False' → 'True'"
        )


class TestFetchText:
    def test_fetch_text_decodes_response_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "dlm.base_models.registry_refresh.urlopen",
            lambda req, timeout: _FakeResponse("olá".encode()),
        )
        assert fetch_text("https://example.com") == "olá"


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

    def test_gated_repo_is_reported_as_drift(self) -> None:
        from unittest.mock import Mock

        from huggingface_hub.errors import GatedRepoError

        drift = check_entry(
            _RaisingApi(GatedRepoError("gated", response=Mock())),
            _spec(),
        )
        assert isinstance(drift, Drift)
        assert ("gating", "readable", "now fully gated") in drift.fields

    def test_missing_repository_is_reported_as_drift(self) -> None:
        from unittest.mock import Mock

        from huggingface_hub.errors import RepositoryNotFoundError

        drift = check_entry(
            _RaisingApi(RepositoryNotFoundError("missing", response=Mock())),
            _spec(),
        )
        assert isinstance(drift, Drift)
        assert ("repository", "present", "missing (renamed or deleted)") in drift.fields

    def test_gating_mismatch_is_reported_when_enabled(self) -> None:
        spec = _spec(requires_acceptance=False)
        drift = check_entry(_Api(sha=spec.revision, gated=True), spec)
        assert isinstance(drift, Drift)
        assert ("requires_acceptance", "False", "True") in drift.fields

    def test_unreachable_provenance_url_is_reported(self) -> None:
        spec = _spec(
            refresh_check_hf_gating=False,
            provenance_url="https://example.com/provenance",
            provenance_match_text="official marker",
        )
        drift = check_entry(
            _Api(sha=spec.revision),
            spec,
            fetch_url_text=lambda _url: (_ for _ in ()).throw(
                HTTPError(_url, 404, "missing", hdrs=None, fp=None)
            ),
        )
        assert isinstance(drift, Drift)
        assert drift.fields[0][0] == "provenance_url"
        assert "unreachable" in drift.fields[0][2]


class TestCheckRegistry:
    def test_check_registry_collects_non_null_drifts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entries = {
            "one": _spec(key="one", hf_id="org/one"),
            "two": _spec(key="two", hf_id="org/two"),
        }
        monkeypatch.setattr("dlm.base_models.registry_refresh.BASE_MODELS", entries)
        monkeypatch.setattr("dlm.base_models.registry_refresh.HfApi", lambda: object())

        def _fake_check_entry(
            api: object, entry: BaseModelSpec, *, fetch_url_text: object
        ) -> Drift | None:
            if entry.key == "one":
                return Drift(key="one", hf_id=entry.hf_id, fields=(("revision", "old", "new"),))
            return None

        monkeypatch.setattr("dlm.base_models.registry_refresh.check_entry", _fake_check_entry)
        drifts = check_registry()
        assert len(drifts) == 1
        assert drifts[0].key == "one"
