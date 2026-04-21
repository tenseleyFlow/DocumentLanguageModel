"""Vendoring path resolution — missing/uninitialized submodule handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm.export.errors import VendoringError
from dlm.export.vendoring import (
    convert_hf_to_gguf_py,
    convert_lora_to_gguf_py,
    llama_cpp_root,
    llama_quantize_bin,
    pinned_tag,
)


def _populate_vendor(root: Path, *, with_binary: bool = True) -> Path:
    """Create a fake `vendor/llama.cpp/` layout the resolver accepts."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "convert_hf_to_gguf.py").write_text("# mock")
    (root / "convert_lora_to_gguf.py").write_text("# mock")
    if with_binary:
        bin_dir = root / "build" / "bin"
        bin_dir.mkdir(parents=True)
        binary = bin_dir / "llama-quantize"
        binary.write_text("# mock binary")
        binary.chmod(0o755)
    (root / "VERSION").write_text("b1234\n")
    return root


class TestLlamaCppRoot:
    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(VendoringError, match="missing"):
            llama_cpp_root(override=tmp_path / "absent")

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(VendoringError, match="empty"):
            llama_cpp_root(override=empty)

    def test_populated_directory_resolves(self, tmp_path: Path) -> None:
        root = _populate_vendor(tmp_path / "llama.cpp")
        assert llama_cpp_root(override=root) == root


class TestScriptResolvers:
    def test_convert_hf_resolves(self, tmp_path: Path) -> None:
        root = _populate_vendor(tmp_path / "llama.cpp")
        path = convert_hf_to_gguf_py(override=root)
        assert path.name == "convert_hf_to_gguf.py"
        assert path.is_file()

    def test_convert_lora_resolves(self, tmp_path: Path) -> None:
        root = _populate_vendor(tmp_path / "llama.cpp")
        path = convert_lora_to_gguf_py(override=root)
        assert path.name == "convert_lora_to_gguf.py"

    def test_missing_script_raises(self, tmp_path: Path) -> None:
        root = _populate_vendor(tmp_path / "llama.cpp")
        (root / "convert_hf_to_gguf.py").unlink()
        with pytest.raises(VendoringError, match="convert_hf_to_gguf"):
            convert_hf_to_gguf_py(override=root)


class TestLlamaQuantizeBin:
    def test_resolves_build_bin_layout(self, tmp_path: Path) -> None:
        root = _populate_vendor(tmp_path / "llama.cpp")
        path = llama_quantize_bin(override=root)
        assert path.is_file()
        assert path.name == "llama-quantize"

    def test_missing_binary_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear PATH so the `shutil.which` fallback can't find a
        # brew-installed llama-quantize on the developer's machine.
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        root = _populate_vendor(tmp_path / "llama.cpp", with_binary=False)
        with pytest.raises(VendoringError, match="llama-quantize"):
            llama_quantize_bin(override=root)

    def test_dlm_llama_cpp_build_env_preferred(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Audit-08 M6: `DLM_LLAMA_CPP_BUILD` overrides the default vendor dir.

        The env var points at a build-only dir (e.g. the ROCm
        `vendor/llama.cpp/build-rocm`) that contains only binaries.
        `_resolve_binary` must find `bin/llama-quantize` there before
        falling through to the vendor tree.

        The production path has `override=None`; we mirror that here
        by driving vendor resolution through `DLM_LLAMA_CPP_ROOT` so
        both env vars coexist (ROCm users set both).
        """
        rocm_build = tmp_path / "build-rocm"
        (rocm_build / "bin").mkdir(parents=True)
        rocm_bin = rocm_build / "bin" / "llama-quantize"
        rocm_bin.write_text("#!/bin/sh\necho rocm\n")
        rocm_bin.chmod(0o755)

        vendor_root = _populate_vendor(tmp_path / "llama.cpp")

        monkeypatch.setenv("DLM_LLAMA_CPP_BUILD", str(rocm_build))
        monkeypatch.setenv("DLM_LLAMA_CPP_ROOT", str(vendor_root))
        path = llama_quantize_bin()
        # The ROCm build binary wins over the vendored CPU build.
        assert path == rocm_bin

    def test_dlm_llama_cpp_build_env_missing_binary_falls_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env var pointing at an incomplete dir falls through to vendor."""
        empty_build = tmp_path / "build-rocm"
        empty_build.mkdir()
        vendor_root = _populate_vendor(tmp_path / "llama.cpp")
        monkeypatch.setenv("DLM_LLAMA_CPP_BUILD", str(empty_build))
        monkeypatch.setenv("DLM_LLAMA_CPP_ROOT", str(vendor_root))
        path = llama_quantize_bin()
        assert path.is_file()
        assert str(vendor_root) in str(path)

    def test_legacy_quantize_name_found(self, tmp_path: Path) -> None:
        """Pre-rename builds shipped `quantize` rather than `llama-quantize`."""
        root = _populate_vendor(tmp_path / "llama.cpp", with_binary=False)
        legacy = root / "build" / "bin"
        legacy.mkdir(parents=True, exist_ok=True)
        legacy_bin = legacy / "quantize"
        legacy_bin.write_text("# legacy")
        legacy_bin.chmod(0o755)
        path = llama_quantize_bin(override=root)
        assert path.name == "quantize"


class TestPinnedTag:
    def test_reads_version_file(self, tmp_path: Path) -> None:
        root = _populate_vendor(tmp_path / "llama.cpp")
        assert pinned_tag(override=root) == "b1234"

    def test_missing_version_file_returns_none(self, tmp_path: Path) -> None:
        root = _populate_vendor(tmp_path / "llama.cpp")
        (root / "VERSION").unlink()
        assert pinned_tag(override=root) is None

    def test_missing_root_returns_none(self, tmp_path: Path) -> None:
        assert pinned_tag(override=tmp_path / "absent") is None

    def test_empty_version_file_returns_none(self, tmp_path: Path) -> None:
        root = _populate_vendor(tmp_path / "llama.cpp")
        (root / "VERSION").write_text("\n")
        assert pinned_tag(override=root) is None
