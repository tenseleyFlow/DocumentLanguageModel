"""Unit coverage for the share push orchestrator."""

from __future__ import annotations

import importlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import zstandard as zstd

from dlm.share.errors import ShareError, SinkError
from dlm.share.push import (
    PushResult,
    _collect_readme_fields,
    _dispatch_push,
    _ensure_pack,
    _noop,
    _sign_pack,
    push,
)
from dlm.share.sinks import SinkKind, SinkSpec

push_mod = importlib.import_module("dlm.share.push")


def _write_pack_with_header(tmp_path: Path, header: dict[str, str]) -> Path:
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w") as tar:
        payload = json.dumps(header).encode("utf-8")
        info = tarfile.TarInfo("pack/header.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    pack_path = tmp_path / "bundle.dlm.pack"
    with pack_path.open("wb") as dst, zstd.ZstdCompressor().stream_writer(dst) as writer:
        writer.write(tar_bytes.getvalue())
    return pack_path


class TestPush:
    def test_push_rejects_peer_destinations(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            push_mod,
            "parse_source",
            lambda destination: SinkSpec(kind=SinkKind.PEER, target=destination),
        )

        with pytest.raises(ShareError, match="push to peer:// is not supported"):
            push(tmp_path / "doc.dlm", "peer://host:7337/doc?token=abc")

    def test_push_signs_dispatches_and_cleans_up(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "doc.dlm"
        source.write_text("body", encoding="utf-8")
        pack_path = tmp_path / "doc.dlm.pack"
        cleanup_called = False
        order: list[str] = []
        progress = object()
        expected = PushResult(
            destination="https://example.test/upload",
            sink_kind=SinkKind.URL,
            bytes_sent=11,
        )

        monkeypatch.setattr(
            push_mod,
            "parse_source",
            lambda destination: SinkSpec(kind=SinkKind.URL, target=destination),
        )

        def _fake_ensure_pack(
            actual_source: Path,
            *,
            include_exports: bool,
            include_base: bool,
            include_logs: bool,
            licensee_acceptance_url: str | None,
        ) -> tuple[Path, object]:
            nonlocal cleanup_called
            assert actual_source == source
            assert include_exports is True
            assert include_base is True
            assert include_logs is True
            assert licensee_acceptance_url == "https://license.example/accept"
            pack_path.write_bytes(b"packed-bytes")

            def _cleanup() -> None:
                nonlocal cleanup_called
                cleanup_called = True

            return pack_path, _cleanup

        def _fake_sign_pack(actual_pack: Path) -> None:
            order.append("sign")
            assert actual_pack == pack_path

        def _fake_dispatch(
            actual_pack: Path,
            spec: SinkSpec,
            *,
            progress: object | None,
        ) -> PushResult:
            order.append("dispatch")
            assert actual_pack == pack_path
            assert spec == SinkSpec(
                kind=SinkKind.URL,
                target="https://example.test/upload",
            )
            assert progress is not None
            return expected

        monkeypatch.setattr(push_mod, "_ensure_pack", _fake_ensure_pack)
        monkeypatch.setattr(push_mod, "_sign_pack", _fake_sign_pack)
        monkeypatch.setattr(push_mod, "_dispatch_push", _fake_dispatch)

        result = push(
            source,
            "https://example.test/upload",
            sign=True,
            include_exports=True,
            include_base=True,
            include_logs=True,
            licensee_acceptance_url="https://license.example/accept",
            progress=cast("object", progress),
        )

        assert result == expected
        assert order == ["sign", "dispatch"]
        assert cleanup_called is True

    def test_push_cleans_up_when_dispatch_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "doc.dlm"
        source.write_text("body", encoding="utf-8")
        pack_path = tmp_path / "doc.dlm.pack"
        cleanup_called = False

        monkeypatch.setattr(
            push_mod,
            "parse_source",
            lambda destination: SinkSpec(kind=SinkKind.URL, target=destination),
        )
        monkeypatch.setattr(
            push_mod,
            "_ensure_pack",
            lambda *args, **kwargs: (
                pack_path,
                lambda: globals().__setitem__("_unused", None),
            ),
        )

        def _cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        monkeypatch.setattr(push_mod, "_ensure_pack", lambda *args, **kwargs: (pack_path, _cleanup))
        monkeypatch.setattr(
            push_mod,
            "_dispatch_push",
            lambda *args, **kwargs: (_ for _ in ()).throw(SinkError("boom")),
        )

        with pytest.raises(SinkError, match="boom"):
            push(source, "https://example.test/upload")

        assert cleanup_called is True


class TestEnsurePack:
    def test_ensure_pack_keeps_existing_pack(self, tmp_path: Path) -> None:
        pack_path = tmp_path / "doc.dlm.pack"
        pack_path.write_bytes(b"already-packed")

        actual_path, cleanup = _ensure_pack(
            pack_path,
            include_exports=False,
            include_base=False,
            include_logs=False,
            licensee_acceptance_url=None,
        )

        assert actual_path == pack_path
        assert cleanup is _noop

    def test_ensure_pack_packs_dlm_and_cleans_up(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "doc.dlm"
        source.write_text("body", encoding="utf-8")
        seen: dict[str, object] = {}

        def _fake_pack(
            actual_source: Path,
            *,
            out: Path,
            include_exports: bool,
            include_base: bool,
            include_logs: bool,
            licensee_acceptance_url: str | None,
        ) -> SimpleNamespace:
            seen["source"] = actual_source
            seen["out"] = out
            seen["include_exports"] = include_exports
            seen["include_base"] = include_base
            seen["include_logs"] = include_logs
            seen["license"] = licensee_acceptance_url
            out.write_bytes(b"packed")
            return SimpleNamespace(path=out)

        monkeypatch.setattr(push_mod, "pack", _fake_pack)

        actual_path, cleanup = _ensure_pack(
            source,
            include_exports=True,
            include_base=True,
            include_logs=True,
            licensee_acceptance_url="https://license.example/accept",
        )

        temp_dir = actual_path.parent
        assert actual_path.read_bytes() == b"packed"
        assert seen == {
            "source": source,
            "out": actual_path,
            "include_exports": True,
            "include_base": True,
            "include_logs": True,
            "license": "https://license.example/accept",
        }

        cleanup()
        assert not temp_dir.exists()


class TestSignPack:
    def test_sign_pack_calls_sign_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dlm.share.signing as signing

        pack_path = tmp_path / "bundle.dlm.pack"
        pack_path.write_bytes(b"packed")
        sig_path = pack_path.with_suffix(pack_path.suffix + ".minisig")
        seen: dict[str, object] = {}

        def _fake_sign_file(target: Path, *, comment: str | None = None) -> Path:
            seen["target"] = target
            seen["comment"] = comment
            sig_path.write_text("signature", encoding="utf-8")
            return sig_path

        monkeypatch.setattr(signing, "sign_file", _fake_sign_file)

        _sign_pack(pack_path)

        assert seen == {
            "target": pack_path,
            "comment": f"dlm push {pack_path.name}",
        }

    def test_sign_pack_propagates_missing_minisign(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlm.share.signing as signing

        pack_path = tmp_path / "bundle.dlm.pack"
        pack_path.write_bytes(b"packed")

        def _fake_sign_file(target: Path, *, comment: str | None = None) -> Path:
            raise signing.MinisignNotAvailableError("missing")

        monkeypatch.setattr(signing, "sign_file", _fake_sign_file)

        with pytest.raises(signing.MinisignNotAvailableError, match="missing"):
            _sign_pack(pack_path)


class TestDispatchPush:
    def test_dispatch_push_hf_uploads_pack(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dlm.share.hf_sink as hf_sink

        pack_path = tmp_path / "bundle.dlm.pack"
        pack_path.write_bytes(b"packed")
        progress = object()
        seen: dict[str, object] = {}

        def _fake_push_hf(
            actual_pack: Path,
            repo_id: str,
            *,
            private: bool = False,
            readme_fields: dict[str, str] | None = None,
            progress: object | None = None,
        ) -> SimpleNamespace:
            seen["pack"] = actual_pack
            seen["repo_id"] = repo_id
            seen["private"] = private
            seen["readme_fields"] = readme_fields
            seen["progress"] = progress
            return SimpleNamespace(
                pack_url="https://huggingface.co/org/repo/blob/main/adapter.dlm.pack"
            )

        monkeypatch.setattr(hf_sink, "push_hf", _fake_push_hf)
        monkeypatch.setattr(
            push_mod,
            "_collect_readme_fields",
            lambda path: {"dlm_id": "01HZPUSH", "base_model": "qwen3-4b"},
        )

        result = _dispatch_push(
            pack_path,
            SinkSpec(kind=SinkKind.HF, target="org/repo"),
            progress=cast("object", progress),
        )

        assert result == PushResult(
            destination="hf:org/repo",
            sink_kind=SinkKind.HF,
            bytes_sent=len(b"packed"),
            detail="pack: https://huggingface.co/org/repo/blob/main/adapter.dlm.pack",
        )
        assert seen == {
            "pack": pack_path,
            "repo_id": "org/repo",
            "private": False,
            "readme_fields": {"dlm_id": "01HZPUSH", "base_model": "qwen3-4b"},
            "progress": progress,
        }

    def test_dispatch_push_url_uploads_pack(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dlm.share.url_sink as url_sink

        pack_path = tmp_path / "bundle.dlm.pack"
        pack_path.write_bytes(b"packed")
        seen: dict[str, object] = {}

        def _fake_push_url(actual_pack: Path, url: str, *, progress: object | None = None) -> None:
            seen["pack"] = actual_pack
            seen["url"] = url
            seen["progress"] = progress

        monkeypatch.setattr(url_sink, "push_url", _fake_push_url)

        result = _dispatch_push(
            pack_path,
            SinkSpec(kind=SinkKind.URL, target="https://example.test/upload"),
            progress=None,
        )

        assert result == PushResult(
            destination="https://example.test/upload",
            sink_kind=SinkKind.URL,
            bytes_sent=len(b"packed"),
        )
        assert seen == {
            "pack": pack_path,
            "url": "https://example.test/upload",
            "progress": None,
        }

    def test_dispatch_push_local_copies_pack(self, tmp_path: Path) -> None:
        pack_path = tmp_path / "bundle.dlm.pack"
        pack_path.write_bytes(b"packed")
        dest = tmp_path / "nested" / "copy.dlm.pack"

        result = _dispatch_push(
            pack_path,
            SinkSpec(kind=SinkKind.LOCAL, target=str(dest)),
            progress=None,
        )

        assert result == PushResult(
            destination=str(dest),
            sink_kind=SinkKind.LOCAL,
            bytes_sent=len(b"packed"),
        )
        assert dest.read_bytes() == b"packed"

    def test_dispatch_push_rejects_unsupported_kind(self, tmp_path: Path) -> None:
        pack_path = tmp_path / "bundle.dlm.pack"
        pack_path.write_bytes(b"packed")

        with pytest.raises(SinkError, match="unsupported sink kind"):
            _dispatch_push(
                pack_path,
                SinkSpec(kind=cast("SinkKind", "weird"), target="x"),
                progress=None,
            )


class TestReadmeFields:
    def test_collect_readme_fields_from_pack(self, tmp_path: Path) -> None:
        pack_path = _write_pack_with_header(
            tmp_path,
            {
                "dlm_id": "01HZHEADER",
                "base_model": "qwen3-8b",
                "adapter_version": "v0007",
            },
        )

        assert _collect_readme_fields(pack_path) == {
            "dlm_id": "01HZHEADER",
            "base_model": "qwen3-8b",
            "adapter_version": "v0007",
        }

    def test_collect_readme_fields_returns_empty_on_bad_pack(self, tmp_path: Path) -> None:
        assert _collect_readme_fields(tmp_path / "missing.dlm.pack") == {}

    def test_noop_is_noop(self) -> None:
        assert _noop() is None
