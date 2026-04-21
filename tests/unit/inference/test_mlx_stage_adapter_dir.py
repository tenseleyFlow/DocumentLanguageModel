"""`stage_mlx_adapter_dir` — covers the pure staging path.

After the audit-11 B1 fix, staging no longer copies PEFT's config
verbatim — it translates it into mlx-lm's schema (`num_layers` +
`lora_parameters`) and writes `adapters.safetensors` rather than
`.npz`. The heavy `MlxBackend.load` path stays `# pragma: no cover`;
the correctness claim here is that given a well-shaped PEFT adapter
dir + a base HF id whose config is cached locally, staging produces
the pair mlx_lm requires.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from dlm.inference.backends import mlx_backend
from dlm.inference.backends.mlx_backend import stage_mlx_adapter_dir
from dlm.inference.mlx_adapter import MlxConversionError


def _write_peft_adapter(dst: Path, *, target_modules: list[str] | None = None) -> None:
    """Write a minimal PEFT-shaped adapter dir."""
    dst.mkdir(parents=True, exist_ok=True)
    resolved_targets = ["q_proj", "v_proj"] if target_modules is None else target_modules
    (dst / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": resolved_targets,
                "use_dora": False,
            }
        )
    )
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(8, 16),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(16, 8),
    }
    save_file(tensors, str(dst / "adapter_model.safetensors"))


@pytest.fixture
def stub_num_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the HF cache lookup; pretend every base has 30 layers."""

    def _stub(_hf_id: str) -> int:
        return 30

    monkeypatch.setattr(mlx_backend, "_resolve_base_num_hidden_layers", _stub)


class TestStageSuccess:
    def test_writes_safetensors_and_translated_config(
        self, tmp_path: Path, stub_num_layers: None
    ) -> None:
        src = tmp_path / "peft"
        dst = tmp_path / "mlx"
        _write_peft_adapter(src)
        staged = stage_mlx_adapter_dir(src, dst, base_hf_id="org/fake")

        assert staged == dst
        assert (dst / "adapters.safetensors").exists()
        assert (dst / "adapter_config.json").exists()

        mlx_cfg: dict[str, Any] = json.loads(
            (dst / "adapter_config.json").read_text(encoding="utf-8")
        )
        assert mlx_cfg["fine_tune_type"] == "lora"
        assert mlx_cfg["num_layers"] == 30
        lp = mlx_cfg["lora_parameters"]
        assert lp["rank"] == 8
        assert lp["scale"] == pytest.approx(16 / 8)
        assert lp["dropout"] == pytest.approx(0.05)
        assert lp["keys"] == ["q_proj", "v_proj"]

    def test_tensor_keys_match_mlx_layout(
        self, tmp_path: Path, stub_num_layers: None
    ) -> None:
        src = tmp_path / "peft"
        dst = tmp_path / "mlx"
        _write_peft_adapter(src)
        stage_mlx_adapter_dir(src, dst, base_hf_id="org/fake")

        # Loaded keys should follow mlx-lm's flattened shape: no
        # `base_model.` prefix, lowercase `lora_a`/`lora_b`, no
        # trailing `.weight`.
        loaded = load_file(str(dst / "adapters.safetensors"))
        for key in loaded:
            assert not key.startswith("base_model.")
            assert ".lora_a" in key or ".lora_b" in key
            assert not key.endswith(".weight")

    def test_dora_emits_dora_fine_tune_type(
        self, tmp_path: Path, stub_num_layers: None
    ) -> None:
        src = tmp_path / "peft"
        dst = tmp_path / "mlx"
        _write_peft_adapter(src)
        cfg = json.loads((src / "adapter_config.json").read_text())
        cfg["use_dora"] = True
        (src / "adapter_config.json").write_text(json.dumps(cfg))
        stage_mlx_adapter_dir(src, dst, base_hf_id="org/fake")

        mlx_cfg = json.loads((dst / "adapter_config.json").read_text(encoding="utf-8"))
        assert mlx_cfg["fine_tune_type"] == "dora"


class TestStagePreflight:
    def test_missing_adapter_config_refused(self, tmp_path: Path) -> None:
        src = tmp_path / "broken"
        src.mkdir()
        save_file(
            {"base_model.model.a.lora_A.weight": torch.zeros(2, 2)},
            str(src / "adapter_model.safetensors"),
        )
        with pytest.raises(MlxConversionError, match="not a PEFT adapter dir"):
            stage_mlx_adapter_dir(src, tmp_path / "out", base_hf_id="org/fake")

    def test_missing_safetensors_refused(self, tmp_path: Path) -> None:
        src = tmp_path / "merged"
        src.mkdir()
        (src / "adapter_config.json").write_text("{}")
        with pytest.raises(MlxConversionError, match="no adapter_model.safetensors"):
            stage_mlx_adapter_dir(src, tmp_path / "out", base_hf_id="org/fake")

    def test_dst_dir_created_if_missing(
        self, tmp_path: Path, stub_num_layers: None
    ) -> None:
        src = tmp_path / "peft"
        _write_peft_adapter(src)
        dst = tmp_path / "nested" / "dst"
        stage_mlx_adapter_dir(src, dst, base_hf_id="org/fake")
        assert dst.is_dir()

    def test_peft_config_missing_r_refused(
        self, tmp_path: Path, stub_num_layers: None
    ) -> None:
        src = tmp_path / "peft"
        _write_peft_adapter(src)
        cfg = json.loads((src / "adapter_config.json").read_text())
        cfg.pop("r")
        (src / "adapter_config.json").write_text(json.dumps(cfg))
        with pytest.raises(MlxConversionError, match="'r'"):
            stage_mlx_adapter_dir(src, tmp_path / "out", base_hf_id="org/fake")

    def test_peft_config_empty_target_modules_refused(
        self, tmp_path: Path, stub_num_layers: None
    ) -> None:
        src = tmp_path / "peft"
        _write_peft_adapter(src, target_modules=[])
        with pytest.raises(MlxConversionError, match="target_modules"):
            stage_mlx_adapter_dir(src, tmp_path / "out", base_hf_id="org/fake")
