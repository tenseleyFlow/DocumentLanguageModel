"""Unit coverage for the share pull orchestrator."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from dlm.share.errors import ShareError, SinkError
from dlm.share.pull import (
    PullResult,
    _dispatch_pull,
    _log_verification,
    _try_hf_sidecar,
    _try_peer_sidecar,
    _try_url_sidecar,
    pull,
)
from dlm.share.signing import VerifyResult, VerifyStatus
from dlm.share.sinks import SinkKind, SinkSpec

pull_mod = importlib.import_module("dlm.share.pull")


class TestPull:
    def test_pull_dispatches_verifies_and_unpacks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = "https://example.test/adapter.dlm.pack"
        out_dir = tmp_path / "out"
        home = tmp_path / "home"
        progress = object()
        spec = SinkSpec(kind=SinkKind.URL, target=source)
        order: list[str] = []
        verification = VerifyResult(status=VerifyStatus.VERIFIED, key_path=tmp_path / "trusted.pub")

        monkeypatch.setattr(
            pull_mod, "parse_source", lambda value: spec if value == source else None
        )

        def _fake_dispatch(
            actual_spec: SinkSpec,
            pack_path: Path,
            sig_path: Path,
            *,
            progress: object | None,
        ) -> int:
            order.append("dispatch")
            assert actual_spec == spec
            assert pack_path.name == "incoming.dlm.pack"
            assert sig_path.name == "incoming.dlm.pack.minisig"
            assert progress is not None
            pack_path.write_bytes(b"pack-bytes")
            sig_path.write_text("signature", encoding="utf-8")
            return 123

        def _fake_verify(pack_path: Path, sig_path: Path) -> VerifyResult:
            order.append("verify")
            assert pack_path.read_bytes() == b"pack-bytes"
            assert sig_path.read_text(encoding="utf-8") == "signature"
            return verification

        def _fake_unpack(
            pack_path: Path,
            *,
            home: Path | None,
            force: bool,
            out_dir: Path,
        ) -> SimpleNamespace:
            order.append("unpack")
            assert pack_path.read_bytes() == b"pack-bytes"
            assert home == tmp_path / "home"
            assert force is True
            assert out_dir == tmp_path / "out"
            return SimpleNamespace(
                dlm_path=out_dir / "restored.dlm",
                store_path=home / "store" / "01HZPULL",
                dlm_id="01HZPULL",
            )

        monkeypatch.setattr(pull_mod, "_dispatch_pull", _fake_dispatch)
        monkeypatch.setattr(pull_mod, "verify_signature", _fake_verify)
        monkeypatch.setattr(pull_mod, "pack_unpack", _fake_unpack)

        result = pull(
            source,
            out_dir=out_dir,
            force=True,
            home=home,
            progress=cast("object", progress),
        )

        assert result == PullResult(
            dlm_path=out_dir / "restored.dlm",
            store_path=home / "store" / "01HZPULL",
            dlm_id="01HZPULL",
            source=source,
            bytes_received=123,
            verification=verification,
        )
        assert order == ["dispatch", "verify", "unpack"]


class TestDispatchPull:
    def test_dispatch_pull_hf_downloads_pack_and_sidecar(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.share.hf_sink as hf_sink

        pack_path = tmp_path / "pack.dlm.pack"
        sig_path = tmp_path / "pack.dlm.pack.minisig"
        progress = object()
        seen: dict[str, object] = {}

        def _fake_pull_hf(repo_id: str, out_path: Path, *, progress: object | None = None) -> int:
            seen["repo_id"] = repo_id
            seen["progress"] = progress
            out_path.write_bytes(b"hf-pack")
            return 7

        monkeypatch.setattr(hf_sink, "pull_hf", _fake_pull_hf)
        monkeypatch.setattr(
            pull_mod,
            "_try_hf_sidecar",
            lambda repo_id, sidecar_path: seen.update(
                {"sidecar_repo_id": repo_id, "sidecar_path": sidecar_path}
            ),
        )

        bytes_received = _dispatch_pull(
            SinkSpec(kind=SinkKind.HF, target="org/repo"),
            pack_path,
            sig_path,
            progress=cast("object", progress),
        )

        assert bytes_received == 7
        assert pack_path.read_bytes() == b"hf-pack"
        assert seen == {
            "repo_id": "org/repo",
            "progress": progress,
            "sidecar_repo_id": "org/repo",
            "sidecar_path": sig_path,
        }

    def test_dispatch_pull_url_downloads_pack_and_sidecar(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.share.url_sink as url_sink

        pack_path = tmp_path / "pack.dlm.pack"
        sig_path = tmp_path / "pack.dlm.pack.minisig"
        seen: dict[str, object] = {}

        def _fake_pull_url(url: str, out_path: Path, *, progress: object | None = None) -> int:
            seen["url"] = url
            seen["progress"] = progress
            out_path.write_bytes(b"url-pack")
            return 9

        monkeypatch.setattr(url_sink, "pull_url", _fake_pull_url)
        monkeypatch.setattr(
            pull_mod,
            "_try_url_sidecar",
            lambda url, sidecar_path: seen.update(
                {"sidecar_url": url, "sidecar_path": sidecar_path}
            ),
        )

        bytes_received = _dispatch_pull(
            SinkSpec(kind=SinkKind.URL, target="https://example.test/pack"),
            pack_path,
            sig_path,
            progress=None,
        )

        assert bytes_received == 9
        assert pack_path.read_bytes() == b"url-pack"
        assert seen == {
            "url": "https://example.test/pack",
            "progress": None,
            "sidecar_url": "https://example.test/pack",
            "sidecar_path": sig_path,
        }

    def test_dispatch_pull_peer_downloads_pack_and_sidecar(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.share.peer as peer

        pack_path = tmp_path / "pack.dlm.pack"
        sig_path = tmp_path / "pack.dlm.pack.minisig"
        seen: dict[str, object] = {}

        def _fake_pull_peer(target: str, out_path: Path, *, progress: object | None = None) -> int:
            seen["target"] = target
            seen["progress"] = progress
            out_path.write_bytes(b"peer-pack")
            return 11

        monkeypatch.setattr(peer, "pull_peer", _fake_pull_peer)
        monkeypatch.setattr(
            pull_mod,
            "_try_peer_sidecar",
            lambda target, sidecar_path: seen.update(
                {"sidecar_target": target, "sidecar_path": sidecar_path}
            ),
        )

        bytes_received = _dispatch_pull(
            SinkSpec(kind=SinkKind.PEER, target="host:7337/pack?token=abc"),
            pack_path,
            sig_path,
            progress=None,
        )

        assert bytes_received == 11
        assert pack_path.read_bytes() == b"peer-pack"
        assert seen == {
            "target": "host:7337/pack?token=abc",
            "progress": None,
            "sidecar_target": "host:7337/pack?token=abc",
            "sidecar_path": sig_path,
        }

    def test_dispatch_pull_local_copies_pack_and_signature(self, tmp_path: Path) -> None:
        src = tmp_path / "src.dlm.pack"
        sig = tmp_path / "src.dlm.pack.minisig"
        src.write_bytes(b"local-pack")
        sig.write_text("local-signature", encoding="utf-8")
        pack_path = tmp_path / "incoming.dlm.pack"
        sig_path = tmp_path / "incoming.dlm.pack.minisig"

        bytes_received = _dispatch_pull(
            SinkSpec(kind=SinkKind.LOCAL, target=str(src)),
            pack_path,
            sig_path,
            progress=None,
        )

        assert bytes_received == len(b"local-pack")
        assert pack_path.read_bytes() == b"local-pack"
        assert sig_path.read_text(encoding="utf-8") == "local-signature"

    def test_dispatch_pull_local_missing_source_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SinkError, match="source missing"):
            _dispatch_pull(
                SinkSpec(kind=SinkKind.LOCAL, target=str(tmp_path / "missing.dlm.pack")),
                tmp_path / "incoming.dlm.pack",
                tmp_path / "incoming.dlm.pack.minisig",
                progress=None,
            )

    def test_dispatch_pull_rejects_unsupported_kind(self, tmp_path: Path) -> None:
        weird = SinkSpec(kind=cast("SinkKind", "weird"), target="x")

        with pytest.raises(ShareError, match="unsupported sink kind"):
            _dispatch_pull(
                weird,
                tmp_path / "incoming.dlm.pack",
                tmp_path / "incoming.dlm.pack.minisig",
                progress=None,
            )


class TestPullSidecars:
    def test_try_hf_sidecar_copies_downloaded_signature(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_hub = ModuleType("huggingface_hub")
        fake_utils = ModuleType("huggingface_hub.utils")
        downloaded = tmp_path / "downloaded.minisig"
        downloaded.write_text("hf-signature", encoding="utf-8")

        class FakeHfHubHTTPError(Exception):
            pass

        def _fake_download(*, repo_id: str, filename: str, repo_type: str) -> str:
            assert repo_id == "org/repo"
            assert filename == "adapter.dlm.pack.minisig"
            assert repo_type == "model"
            return str(downloaded)

        fake_hub.hf_hub_download = _fake_download
        fake_utils.HfHubHTTPError = FakeHfHubHTTPError
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
        monkeypatch.setitem(sys.modules, "huggingface_hub.utils", fake_utils)

        sig_path = tmp_path / "incoming.minisig"
        _try_hf_sidecar("org/repo", sig_path)

        assert sig_path.read_text(encoding="utf-8") == "hf-signature"

    def test_try_hf_sidecar_suppresses_hub_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_hub = ModuleType("huggingface_hub")
        fake_utils = ModuleType("huggingface_hub.utils")

        class FakeHfHubHTTPError(Exception):
            pass

        def _fake_download(*, repo_id: str, filename: str, repo_type: str) -> str:
            raise FakeHfHubHTTPError("missing")

        fake_hub.hf_hub_download = _fake_download
        fake_utils.HfHubHTTPError = FakeHfHubHTTPError
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
        monkeypatch.setitem(sys.modules, "huggingface_hub.utils", fake_utils)

        sig_path = tmp_path / "incoming.minisig"
        _try_hf_sidecar("org/repo", sig_path)

        assert not sig_path.exists()

    def test_try_url_sidecar_suppresses_missing_sidecar(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.share.url_sink as url_sink

        def _fake_pull_url(url: str, out_path: Path, *, progress: object | None = None) -> int:
            raise SinkError(f"missing {url}")

        monkeypatch.setattr(url_sink, "pull_url", _fake_pull_url)

        _try_url_sidecar("https://example.test/pack", tmp_path / "incoming.minisig")

    def test_try_peer_sidecar_suppresses_missing_sidecar(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.share.peer as peer

        def _fake_pull_peer(target: str, out_path: Path, *, progress: object | None = None) -> int:
            raise SinkError(f"missing {target}")

        monkeypatch.setattr(peer, "pull_peer", _fake_pull_peer)

        _try_peer_sidecar("host:7337/pack?token=abc", tmp_path / "incoming.minisig")


class TestVerificationLogging:
    def test_log_verification_verified(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        caplog.set_level("INFO")

        _log_verification(
            "hf:org/repo",
            VerifyResult(status=VerifyStatus.VERIFIED, key_path=tmp_path / "trusted.pub"),
        )

        assert "verified signature" in caplog.text

    def test_log_verification_unverified(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("WARNING")

        _log_verification(
            "https://example.test/pack",
            VerifyResult(status=VerifyStatus.UNVERIFIED, detail="no trusted key matched"),
        )

        assert "signature present but could not verify" in caplog.text
        assert "no trusted key matched" in caplog.text

    def test_log_verification_unsigned(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("INFO")

        _log_verification(
            "./local.dlm.pack",
            VerifyResult(status=VerifyStatus.UNSIGNED),
        )

        assert "no signature" in caplog.text
