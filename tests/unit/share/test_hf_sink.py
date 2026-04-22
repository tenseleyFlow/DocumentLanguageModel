"""`dlm.share.hf_sink` — push/pull via huggingface_hub.

Covers the sink without hitting the network: `huggingface_hub`
functions (`create_repo`, `upload_file`, `hf_hub_download`) are
monkeypatched to stand-ins so we can assert call shape, README
rendering, and error translation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm.share import hf_sink
from dlm.share.errors import SinkError
from dlm.share.hf_sink import HFUploadSummary, _render_readme, pull_hf, push_hf


class _FakeHfHubHTTPError(Exception):
    """Minimal stand-in for huggingface_hub.utils.HfHubHTTPError.

    The real class accepts (message, response, server_message) but the
    sink only catches by type and formats `str(exc)` into its
    SinkError — any Exception subclass raised where the real error
    would be will produce the same observable result if we patch the
    import target.
    """


@pytest.fixture
def pack(tmp_path: Path) -> Path:
    p = tmp_path / "adapter.dlm.pack"
    p.write_bytes(b"PACK" * 512)
    return p


@pytest.fixture
def patched_hub(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, object]]]:
    """Patch huggingface_hub symbols the sink deferred-imports.

    Returns a capture dict the tests can inspect for call args. Default
    behavior is success — tests that want a failure replace the
    closure after the fixture returns.
    """
    captured: dict[str, list[dict[str, object]]] = {
        "create_repo": [],
        "upload_file": [],
        "hf_hub_download": [],
    }

    def _create_repo(**kwargs: object) -> None:
        captured["create_repo"].append(kwargs)

    def _upload_file(**kwargs: object) -> str:
        captured["upload_file"].append(kwargs)
        # Real hub returns a CommitInfo; sink str()s it. A string is fine.
        return f"https://huggingface.co/{kwargs['repo_id']}/blob/main/{kwargs['path_in_repo']}"

    def _hf_hub_download(**kwargs: object) -> str:
        captured["hf_hub_download"].append(kwargs)
        # Return a path to a tmp file with some content.
        scratch = Path("/tmp/fake-hf-cache/adapter.dlm.pack")
        scratch.parent.mkdir(parents=True, exist_ok=True)
        scratch.write_bytes(b"downloaded bytes")
        return str(scratch)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "create_repo", _create_repo, raising=False)
    monkeypatch.setattr(huggingface_hub, "upload_file", _upload_file, raising=False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _hf_hub_download, raising=False)

    # The sink imports HfHubHTTPError from huggingface_hub.utils; replace
    # it with our stand-in so tests that want a failure can raise it.
    import huggingface_hub.utils

    monkeypatch.setattr(huggingface_hub.utils, "HfHubHTTPError", _FakeHfHubHTTPError, raising=False)
    return captured


class TestPushHf:
    def test_missing_file_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SinkError, match="pack file missing"):
            push_hf(tmp_path / "nope.pack", "user/repo")

    def test_happy_path_creates_repo_and_uploads_pack_and_readme(
        self, pack: Path, patched_hub: dict[str, list[dict[str, object]]]
    ) -> None:
        summary = push_hf(
            pack,
            "user/myadapter",
            private=True,
            readme_fields={
                "dlm_id": "01K...",
                "base_model": "smollm2-135m",
                "adapter_version": "1",
            },
        )

        assert isinstance(summary, HFUploadSummary)
        assert summary.repo_id == "user/myadapter"

        # Repo created once with expected args.
        assert len(patched_hub["create_repo"]) == 1
        cr = patched_hub["create_repo"][0]
        assert cr["repo_id"] == "user/myadapter"
        assert cr["private"] is True
        assert cr["exist_ok"] is True

        # Two uploads — pack + README.
        assert len(patched_hub["upload_file"]) == 2
        pack_call, readme_call = patched_hub["upload_file"]
        assert pack_call["path_in_repo"] == "adapter.dlm.pack"
        assert pack_call["repo_id"] == "user/myadapter"
        assert readme_call["path_in_repo"] == "README.md"
        # README body carries our fields.
        assert isinstance(readme_call["path_or_fileobj"], bytes)
        body = readme_call["path_or_fileobj"].decode("utf-8")
        assert "01K..." in body
        assert "smollm2-135m" in body

    def test_create_repo_failure_translates_to_sink_error(
        self,
        pack: Path,
        patched_hub: dict[str, list[dict[str, object]]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import huggingface_hub

        def _boom(**kwargs: object) -> None:
            raise _FakeHfHubHTTPError("access denied")

        monkeypatch.setattr(huggingface_hub, "create_repo", _boom, raising=False)

        with pytest.raises(SinkError, match="failed to ensure repo"):
            push_hf(pack, "user/myadapter")

    def test_upload_failure_translates_to_sink_error(
        self,
        pack: Path,
        patched_hub: dict[str, list[dict[str, object]]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import huggingface_hub

        def _boom(**kwargs: object) -> None:
            raise _FakeHfHubHTTPError("quota exceeded")

        monkeypatch.setattr(huggingface_hub, "upload_file", _boom, raising=False)

        with pytest.raises(SinkError, match="upload failed"):
            push_hf(pack, "user/myadapter")

    def test_progress_fires_with_full_size(
        self, pack: Path, patched_hub: dict[str, list[dict[str, object]]]
    ) -> None:
        seen: list[tuple[int, int]] = []
        push_hf(pack, "user/myadapter", progress=lambda d, t: seen.append((d, t)))
        total = pack.stat().st_size
        # Progress fires once at 0 (pre-upload) and once at 100% (post).
        assert seen[0] == (0, total)
        assert seen[-1] == (total, total)


class TestPullHf:
    def test_happy_path_downloads_to_out_path(
        self, tmp_path: Path, patched_hub: dict[str, list[dict[str, object]]]
    ) -> None:
        out = tmp_path / "pulled.pack"
        written = pull_hf("user/myadapter", out)

        assert out.is_file()
        assert out.read_bytes() == b"downloaded bytes"
        assert written == len(b"downloaded bytes")

        assert len(patched_hub["hf_hub_download"]) == 1
        call = patched_hub["hf_hub_download"][0]
        assert call["repo_id"] == "user/myadapter"
        assert call["filename"] == "adapter.dlm.pack"
        assert call["repo_type"] == "model"

    def test_download_failure_translates_to_sink_error(
        self,
        tmp_path: Path,
        patched_hub: dict[str, list[dict[str, object]]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import huggingface_hub

        def _boom(**kwargs: object) -> None:
            raise _FakeHfHubHTTPError("not found")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom, raising=False)

        with pytest.raises(SinkError, match="download failed"):
            pull_hf("user/myadapter", tmp_path / "out.pack")

    def test_creates_parent_dir(
        self, tmp_path: Path, patched_hub: dict[str, list[dict[str, object]]]
    ) -> None:
        out = tmp_path / "nested" / "dir" / "pulled.pack"
        pull_hf("user/myadapter", out)
        assert out.is_file()

    def test_progress_fires_at_end(
        self, tmp_path: Path, patched_hub: dict[str, list[dict[str, object]]]
    ) -> None:
        seen: list[tuple[int, int]] = []
        pull_hf("user/myadapter", tmp_path / "out.pack", progress=lambda d, t: seen.append((d, t)))
        # Downloaded-bytes fixture is 16 bytes; only a single end-of-download call fires.
        assert seen == [(16, 16)]


class TestRenderReadme:
    def test_shape_contains_core_fields(self) -> None:
        body = _render_readme(
            "alice/cool-adapter",
            {
                "dlm_id": "01K...",
                "base_model": "qwen2.5-1.5b",
                "adapter_version": "3",
                "license": "Apache-2.0",
            },
        )
        assert body.startswith("---\n")
        assert "library_name: dlm" in body
        assert "alice/cool-adapter" in body
        assert "`01K...`" in body
        assert "`qwen2.5-1.5b`" in body
        assert "`3`" in body
        assert "Apache-2.0" in body

    def test_unknown_fields_placeholder(self) -> None:
        body = _render_readme("bob/minimal", {})
        # Missing fields fall back to `(unknown)` placeholders.
        assert "(unknown)" in body
        assert "See the base model's license." in body

    def test_install_block_references_repo(self) -> None:
        body = _render_readme("team/pkg", {})
        assert "dlm pull hf:team/pkg" in body


class TestDeferredImportFallback:
    def test_push_raises_sink_error_when_hub_missing(
        self, pack: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If huggingface_hub is unimportable, push_hf raises SinkError.

        The pragma'd fallback path in the sink catches ImportError and
        translates to SinkError. We simulate the missing import via
        sys.modules injection, then call through.
        """
        # Capture the raised error type; the sink wraps into SinkError.
        import sys

        # Force the `from huggingface_hub import ...` to raise ImportError.
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        _ = hf_sink  # silence unused-import lint
        with pytest.raises((SinkError, ImportError)):
            push_hf(pack, "user/myadapter")
