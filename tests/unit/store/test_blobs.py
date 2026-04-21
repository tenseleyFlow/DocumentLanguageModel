"""Content-addressed blob store — `BlobStore.put/get/gc`.

Covers:

- `put(path)` returns deterministic `(sha, ext, size)`.
- Re-ingesting identical bytes with a different name is a noop on
  disk but returns the same sha + a fresh handle.
- `get(sha)` raises `BlobMissingError` on miss, resolves on hit.
- `gc(live_shas)` deletes exactly the complement and yields the
  removed handles.
- Extension normalization: unknown suffixes become `.bin`, known
  ones lowercase.
- Content-hash sha-prefix fan-out at `<sha[:2]>/`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dlm.store.blobs import (
    BlobCorruptError,
    BlobHandle,
    BlobMissingError,
    BlobStore,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "blobs"


@pytest.fixture
def store(store_root: Path) -> BlobStore:
    return BlobStore(store_root)


class TestBlobStorePut:
    def test_put_returns_handle(self, store: BlobStore, tmp_path: Path) -> None:
        src = tmp_path / "hero.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\n...")
        handle = store.put(src)
        assert handle.sha == _sha(b"\x89PNG\r\n\x1a\n...")
        assert handle.ext == ".png"
        assert handle.size == len(b"\x89PNG\r\n\x1a\n...")

    def test_put_creates_prefix_bucket(
        self, store: BlobStore, store_root: Path, tmp_path: Path
    ) -> None:
        src = tmp_path / "a.png"
        data = b"one"
        src.write_bytes(data)
        handle = store.put(src)
        bucket = store_root / handle.sha[:2]
        assert bucket.is_dir()
        blob_file = bucket / f"{handle.sha}.png"
        assert blob_file.exists()
        assert blob_file.read_bytes() == data

    def test_put_idempotent(self, store: BlobStore, tmp_path: Path) -> None:
        src1 = tmp_path / "a.png"
        src2 = tmp_path / "b.png"
        src1.write_bytes(b"same bytes")
        src2.write_bytes(b"same bytes")
        h1 = store.put(src1)
        h2 = store.put(src2)
        assert h1 == h2

    def test_put_different_bytes_different_shas(self, store: BlobStore, tmp_path: Path) -> None:
        a = tmp_path / "a.png"
        b = tmp_path / "b.png"
        a.write_bytes(b"first")
        b.write_bytes(b"second")
        ha = store.put(a)
        hb = store.put(b)
        assert ha.sha != hb.sha

    def test_put_bytes_same_as_put(self, store: BlobStore, tmp_path: Path) -> None:
        src = tmp_path / "x.jpg"
        data = b"jpeg body"
        src.write_bytes(data)
        from_path = store.put(src)
        from_bytes = store.put_bytes(data, ext=".jpg")
        assert from_path == from_bytes


class TestBlobStoreGet:
    def test_get_returns_stored_path(self, store: BlobStore, tmp_path: Path) -> None:
        src = tmp_path / "a.png"
        data = b"some pixels"
        src.write_bytes(data)
        handle = store.put(src)
        resolved = store.get(handle.sha)
        assert resolved.read_bytes() == data

    def test_get_missing_raises(self, store: BlobStore) -> None:
        with pytest.raises(BlobMissingError):
            store.get("a" * 64)

    def test_get_invalid_sha_raises(self, store: BlobStore) -> None:
        with pytest.raises(ValueError):
            store.get("not-a-sha")

    def test_exists(self, store: BlobStore, tmp_path: Path) -> None:
        src = tmp_path / "a.png"
        src.write_bytes(b"abc")
        handle = store.put(src)
        assert store.exists(handle.sha) is True
        assert store.exists("a" * 64) is False


class TestBlobStoreGC:
    def test_gc_deletes_nonlive(self, store: BlobStore, tmp_path: Path) -> None:
        a = tmp_path / "a.png"
        b = tmp_path / "b.png"
        c = tmp_path / "c.png"
        a.write_bytes(b"alpha")
        b.write_bytes(b"bravo")
        c.write_bytes(b"charlie")
        ha = store.put(a)
        hb = store.put(b)
        hc = store.put(c)
        live = {ha.sha, hc.sha}
        removed = list(store.gc(live))
        assert [h.sha for h in removed] == [hb.sha]
        assert store.exists(ha.sha)
        assert not store.exists(hb.sha)
        assert store.exists(hc.sha)

    def test_gc_empty_live_removes_everything(self, store: BlobStore, tmp_path: Path) -> None:
        src = tmp_path / "x.png"
        src.write_bytes(b"xxx")
        store.put(src)
        removed = list(store.gc(set()))
        assert len(removed) == 1
        assert list(store.iter_all()) == []

    def test_gc_noop_on_empty_store(self, store: BlobStore) -> None:
        assert list(store.gc({"a" * 64})) == []


class TestBlobStoreExtensions:
    @pytest.mark.parametrize(
        ("suffix", "expected"),
        [
            # Images.
            (".png", ".png"),
            (".PNG", ".png"),
            (".jpg", ".jpg"),
            (".jpeg", ".jpeg"),
            (".webp", ".webp"),
            # Audio (Sprint 35.2).
            (".wav", ".wav"),
            (".WAV", ".wav"),
            (".flac", ".flac"),
            (".ogg", ".ogg"),
            # Unknown / deferred. mp3 lands with libsndfile MP3 support.
            (".mp3", ".bin"),
            (".m4a", ".bin"),
            (".unknown", ".bin"),
            ("", ".bin"),
        ],
    )
    def test_extension_normalization(
        self,
        store: BlobStore,
        tmp_path: Path,
        suffix: str,
        expected: str,
    ) -> None:
        src = tmp_path / f"input{suffix}"
        src.write_bytes(b"data")
        handle = store.put(src)
        assert handle.ext == expected


class TestBlobStoreVerify:
    def test_verify_ok(self, store: BlobStore, tmp_path: Path) -> None:
        src = tmp_path / "a.png"
        src.write_bytes(b"ok bytes")
        handle = store.put(src)
        store.verify(handle.sha)  # does not raise

    def test_verify_detects_tamper(self, store: BlobStore, tmp_path: Path) -> None:
        src = tmp_path / "a.png"
        src.write_bytes(b"ok bytes")
        handle = store.put(src)
        # Tamper on disk.
        on_disk = store.get(handle.sha)
        on_disk.write_bytes(b"tampered!")
        with pytest.raises(BlobCorruptError):
            store.verify(handle.sha)


class TestBlobStoreIteration:
    def test_iter_all_deterministic(self, store: BlobStore, tmp_path: Path) -> None:
        files = [tmp_path / f"{i}.png" for i in range(3)]
        for i, f in enumerate(files):
            f.write_bytes(f"bytes {i}".encode())
        handles = [store.put(f) for f in files]
        iterated = list(store.iter_all())
        assert sorted(h.sha for h in handles) == sorted(h.sha for h in iterated)


class TestBlobStoreClear:
    def test_clear_removes_tree(self, store: BlobStore, store_root: Path, tmp_path: Path) -> None:
        src = tmp_path / "a.png"
        src.write_bytes(b"x")
        store.put(src)
        assert store_root.is_dir()
        store.clear()
        assert not store_root.exists()


class TestBlobHandleValue:
    def test_blob_handle_is_frozen(self) -> None:
        h = BlobHandle(sha="a" * 64, ext=".png", size=10)
        with pytest.raises(AttributeError):
            h.sha = "b" * 64  # type: ignore[misc]
