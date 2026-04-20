"""`stage_mlx_adapter_dir` — audit-08 B3 fix.

Covers the file-layout piece of the MLX backend load path without
importing mlx / mlx_lm. The heavy `MlxBackend.load` path stays
`# pragma: no cover` — the correctness claim is that given a
well-shaped PEFT adapter dir, staging produces the pair mlx_lm
requires (`adapters.npz` + `adapter_config.json`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from dlm.inference.backends.mlx_backend import stage_mlx_adapter_dir
from dlm.inference.mlx_adapter import MlxConversionError


def _write_peft_adapter(dst: Path) -> None:
    """Write a minimal PEFT-shaped adapter dir."""
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 8, "lora_alpha": 16})
    )
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(8, 16),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(16, 8),
    }
    save_file(tensors, str(dst / "adapter_model.safetensors"))


class TestStageSuccess:
    def test_writes_npz_and_config(self, tmp_path: Path) -> None:
        src = tmp_path / "peft"
        dst = tmp_path / "mlx"
        _write_peft_adapter(src)
        staged = stage_mlx_adapter_dir(src, dst)

        assert staged == dst
        assert (dst / "adapters.npz").exists()
        assert (dst / "adapter_config.json").exists()
        # Config is copied verbatim.
        original = (src / "adapter_config.json").read_text()
        assert (dst / "adapter_config.json").read_text() == original


class TestStagePreflight:
    def test_missing_adapter_config_refused(self, tmp_path: Path) -> None:
        """N10: reject a dir that doesn't look like a PEFT adapter."""
        src = tmp_path / "broken"
        src.mkdir()
        # tensor only, no config
        save_file(
            {"base_model.model.a.lora_A.weight": torch.zeros(2, 2)},
            str(src / "adapter_model.safetensors"),
        )
        with pytest.raises(MlxConversionError, match="not a PEFT adapter dir"):
            stage_mlx_adapter_dir(src, tmp_path / "out")

    def test_missing_safetensors_refused(self, tmp_path: Path) -> None:
        """Merged-model dir (has config, no adapter_model.safetensors)."""
        src = tmp_path / "merged"
        src.mkdir()
        (src / "adapter_config.json").write_text("{}")
        with pytest.raises(MlxConversionError, match="no adapter_model.safetensors"):
            stage_mlx_adapter_dir(src, tmp_path / "out")

    def test_dst_dir_created_if_missing(self, tmp_path: Path) -> None:
        src = tmp_path / "peft"
        _write_peft_adapter(src)
        # nested non-existent dst
        dst = tmp_path / "nested" / "dst"
        stage_mlx_adapter_dir(src, dst)
        assert dst.is_dir()
