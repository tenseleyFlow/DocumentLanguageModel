"""Source-string parsing — hf: / https: / peer: / local path routing."""

from __future__ import annotations

import pytest

from dlm.share import SinkKind, UnknownSinkError, parse_source


class TestHfScheme:
    def test_org_repo(self) -> None:
        s = parse_source("hf:myuser/my-adapter")
        assert s.kind == SinkKind.HF
        assert s.target == "myuser/my-adapter"

    def test_nested_repo_path(self) -> None:
        # HF allows multi-segment namespaces via subpaths (team/project/name).
        # We preserve the suffix verbatim; the HF client decides legality.
        s = parse_source("hf:some-org/my-project-adapter-v2")
        assert s.kind == SinkKind.HF
        assert s.target == "some-org/my-project-adapter-v2"

    def test_rejects_no_slash(self) -> None:
        with pytest.raises(UnknownSinkError, match="hf:<org>/<repo>"):
            parse_source("hf:justusername")

    def test_rejects_empty(self) -> None:
        with pytest.raises(UnknownSinkError, match="hf:<org>/<repo>"):
            parse_source("hf:")


class TestUrlScheme:
    def test_https(self) -> None:
        s = parse_source("https://example.com/doc.pack")
        assert s.kind == SinkKind.URL
        assert s.target == "https://example.com/doc.pack"

    def test_http(self) -> None:
        s = parse_source("http://example.com/doc.pack")
        assert s.kind == SinkKind.URL


class TestPeerScheme:
    def test_host_port_path_query(self) -> None:
        s = parse_source("peer://alice-laptop:7337/01HZABC?token=xyz")
        assert s.kind == SinkKind.PEER
        assert s.target == "alice-laptop:7337/01HZABC?token=xyz"

    def test_rejects_empty_target(self) -> None:
        with pytest.raises(UnknownSinkError, match="host:port"):
            parse_source("peer://")


class TestLocalPath:
    @pytest.mark.parametrize(
        "path",
        [
            "./mydoc.pack",
            "/tmp/foo.pack",
            "~/docs/foo.pack",
            "../relative/foo.pack",
            "file-with.dots.pack",
        ],
    )
    def test_path_looking(self, path: str) -> None:
        s = parse_source(path)
        assert s.kind == SinkKind.LOCAL
        assert s.target == path


class TestRejection:
    def test_empty_string(self) -> None:
        with pytest.raises(UnknownSinkError, match="empty"):
            parse_source("")

    def test_unknown_scheme(self) -> None:
        # No path-looking suffix, no known scheme.
        with pytest.raises(UnknownSinkError):
            parse_source("ftp-ish-garbage")
