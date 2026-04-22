"""`dlm.share.url_sink` — HTTPS POST push + GET pull with auth + streaming.

Covers the sink without hitting the network: `urllib.request.urlopen`
is monkeypatched to a fake response so we can assert header shape,
chunked write behavior, progress fanout, and error-translation.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from dlm.share.errors import SinkError
from dlm.share.url_sink import pull_url, push_url


class _FakeResponse:
    """Stand-in for the context-manager returned by urllib.request.urlopen."""

    def __init__(
        self,
        *,
        status: int = 200,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._body = io.BytesIO(body)

    def read(self, n: int = -1) -> bytes:
        return self._body.read(n)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


@pytest.fixture
def pack(tmp_path: Path) -> Path:
    """A minimal fake pack — the url sink treats it as opaque bytes."""
    p = tmp_path / "adapter.dlm.pack"
    p.write_bytes(b"PACK" * 1024)  # 4 KiB
    return p


class TestPushUrl:
    def test_missing_file_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SinkError, match="pack file missing"):
            push_url(tmp_path / "nope.pack", "https://example.com/upload")

    def test_happy_path_sends_pack_bytes(self, pack: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def _fake_urlopen(req: urllib.request.Request, data: object, timeout: int) -> _FakeResponse:
            captured["method"] = req.get_method()
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            # urllib passes our streaming read adapter's output as `data`;
            # materialize it fully so we can assert length.
            body = data.read() if hasattr(data, "read") else data
            assert isinstance(body, (bytes, bytearray))
            captured["body_len"] = len(body)
            return _FakeResponse(status=201)

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        monkeypatch.delenv("DLM_SHARE_AUTH", raising=False)

        push_url(pack, "https://example.com/upload")

        assert captured["method"] == "POST"
        assert captured["url"] == "https://example.com/upload"
        headers = captured["headers"]
        assert isinstance(headers, dict)
        # header keys are title-cased by urllib
        assert headers.get("Content-type") == "application/octet-stream"
        assert headers.get("Content-length") == str(pack.stat().st_size)
        # No Authorization header when env var unset.
        assert "Authorization" not in headers
        assert captured["body_len"] == pack.stat().st_size

    def test_auth_header_from_env(self, pack: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_auth: list[str | None] = []

        def _fake_urlopen(req: urllib.request.Request, data: object, timeout: int) -> _FakeResponse:
            captured_auth.append(req.get_header("Authorization"))
            return _FakeResponse(status=200)

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        monkeypatch.setenv("DLM_SHARE_AUTH", "Bearer secret-token")

        push_url(pack, "https://example.com/upload")
        assert captured_auth == ["Bearer secret-token"]

    def test_non_2xx_raises_sink_error(self, pack: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fake_urlopen(req: urllib.request.Request, data: object, timeout: int) -> _FakeResponse:
            raise urllib.error.HTTPError(
                url=req.full_url,
                code=403,
                msg="Forbidden",
                hdrs=None,  # type: ignore[arg-type]
                fp=None,
            )

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

        with pytest.raises(SinkError, match="HTTP 403"):
            push_url(pack, "https://example.com/upload")

    def test_network_error_raises_sink_error(
        self, pack: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _fake_urlopen(req: urllib.request.Request, data: object, timeout: int) -> _FakeResponse:
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

        with pytest.raises(SinkError, match="network error"):
            push_url(pack, "https://example.com/upload")

    def test_progress_called_at_start_and_end(
        self, pack: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _fake_urlopen(req: urllib.request.Request, data: object, timeout: int) -> _FakeResponse:
            # Drain the stream so the 100% progress call fires.
            if hasattr(data, "read"):
                data.read()
            return _FakeResponse(status=200)

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        monkeypatch.delenv("DLM_SHARE_AUTH", raising=False)

        seen: list[tuple[int, int]] = []
        push_url(pack, "https://example.com/u", progress=lambda d, t: seen.append((d, t)))

        total = pack.stat().st_size
        # Start + end bracket (exact count not pinned — iterator may be
        # materialized once; we check the first and last samples).
        assert seen[0] == (0, total)
        assert seen[-1] == (total, total)

    def test_http_scheme_warns(
        self,
        pack: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def _fake_urlopen(req: urllib.request.Request, data: object, timeout: int) -> _FakeResponse:
            return _FakeResponse(status=200)

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        with caplog.at_level("WARNING", logger="dlm.share.url_sink"):
            push_url(pack, "http://example.com/u")

        assert any("plaintext HTTP" in rec.message for rec in caplog.records)


class TestPullUrl:
    def test_happy_path_writes_body_to_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = b"hello world" * 500  # > 1 chunk boundary
        out = tmp_path / "fetched.pack"

        def _fake_urlopen(req: urllib.request.Request, timeout: int) -> _FakeResponse:
            return _FakeResponse(
                status=200,
                body=body,
                headers={"Content-Length": str(len(body))},
            )

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

        written = pull_url("https://example.com/p", out)
        assert written == len(body)
        assert out.read_bytes() == body

    def test_missing_content_length_still_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "fetched.pack"
        body = b"no length header"

        def _fake_urlopen(req: urllib.request.Request, timeout: int) -> _FakeResponse:
            return _FakeResponse(status=200, body=body, headers={})

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        seen: list[tuple[int, int]] = []
        written = pull_url(
            "https://example.com/p",
            out,
            progress=lambda d, t: seen.append((d, t)),
        )
        assert written == len(body)
        assert out.read_bytes() == body
        # total is 0 because server didn't advertise Content-Length
        assert all(t == 0 for _, t in seen)

    def test_non_2xx_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fake_urlopen(req: urllib.request.Request, timeout: int) -> _FakeResponse:
            raise urllib.error.HTTPError(
                url=req.full_url,
                code=404,
                msg="Not Found",
                hdrs=None,  # type: ignore[arg-type]
                fp=None,
            )

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        with pytest.raises(SinkError, match="HTTP 404"):
            pull_url("https://example.com/p", tmp_path / "out.pack")

    def test_creates_parent_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        out = tmp_path / "nested" / "dir" / "fetched.pack"

        def _fake_urlopen(req: urllib.request.Request, timeout: int) -> _FakeResponse:
            return _FakeResponse(status=200, body=b"x", headers={"Content-Length": "1"})

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        pull_url("https://example.com/p", out)
        assert out.is_file()
