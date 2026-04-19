"""Snapshot tests for the vendored-tool command lines (audit F09).

These tests lock down the *exact* argv shape we hand to `subprocess.run`.
A drift in the upstream CLI between llama.cpp tags will break the
snapshot and the bump script's pre-flight check — that's the audit
F09 gate.
"""

from __future__ import annotations

from pathlib import Path

from dlm.export.adapter_gguf import build_convert_lora_args
from dlm.export.base_gguf import build_convert_hf_args, build_quantize_args


def _populate_vendor(root: Path, *, with_bin: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "convert_hf_to_gguf.py").write_text("# mock")
    (root / "convert_lora_to_gguf.py").write_text("# mock")
    if with_bin:
        bin_dir = root / "build" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "llama-quantize").write_text("# mock")
    return root


class TestConvertHfArgs:
    def test_positional_base_dir_then_outfile_outtype(self, tmp_path: Path) -> None:
        vendor = _populate_vendor(tmp_path / "llama.cpp")
        base = tmp_path / "base"
        base.mkdir()
        out = tmp_path / "base.fp16.gguf"
        args = build_convert_hf_args(
            base,
            out_fp16=out,
            script_override=vendor,
            python_exe="/usr/bin/python3",
        )
        # Shape: [python, script, positional_base, --outfile, path, --outtype, f16]
        assert args[0] == "/usr/bin/python3"
        assert args[1].endswith("convert_hf_to_gguf.py")
        assert args[2] == str(base)
        assert args[3] == "--outfile"
        assert args[4] == str(out)
        assert args[5] == "--outtype"
        assert args[6] == "f16"


class TestConvertLoraArgs:
    def test_positional_adapter_dir_then_outfile_outtype(self, tmp_path: Path) -> None:
        """Audit F09: NO `--base` flag — adapter_dir is positional only."""
        vendor = _populate_vendor(tmp_path / "llama.cpp")
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        out = tmp_path / "adapter.gguf"
        args = build_convert_lora_args(
            adapter,
            out_gguf=out,
            script_override=vendor,
            python_exe="/usr/bin/python3",
        )
        assert args[2] == str(adapter)
        # Assert NO `--base` flag exists anywhere in the args.
        assert "--base" not in args
        assert "--base-model" not in args


class TestQuantizeArgs:
    def test_positional_in_out_quant(self, tmp_path: Path) -> None:
        """`llama-quantize <in> <out> <QUANT>` — quant string is positional."""
        vendor = _populate_vendor(tmp_path / "llama.cpp")
        fp16 = tmp_path / "base.fp16.gguf"
        out = tmp_path / "base.Q4_K_M.gguf"
        args = build_quantize_args(
            fp16, out_quant=out, quant="Q4_K_M", bin_override=vendor
        )
        # binary, in, out, quant
        assert len(args) == 4
        assert args[-1] == "Q4_K_M"
        assert args[1] == str(fp16)
        assert args[2] == str(out)
        # No flag prefixes.
        assert "--quant" not in args
