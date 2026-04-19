"""Unit-test probe behavior via mocked HF + vendored-file fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dlm.base_models import BaseModelSpec, GatedModelError
from dlm.base_models.probes import (
    probe_architecture,
    probe_chat_template,
    probe_gguf_arch_supported,
    probe_pretokenizer_hash,
    run_all,
)


def _spec() -> BaseModelSpec:
    return BaseModelSpec.model_validate(
        {
            "key": "demo-1b",
            "hf_id": "org/demo",
            "revision": "0" * 40,
            "architecture": "DemoForCausalLM",
            "params": 1_000_000_000,
            "target_modules": ["q_proj", "v_proj"],
            "template": "chatml",
            "gguf_arch": "demo",
            "tokenizer_pre": "demo",
            "license_spdx": "MIT",
            "requires_acceptance": False,
            "redistributable": True,
            "size_gb_fp16": 2.0,
            "context_length": 4096,
            "recommended_seq_len": 2048,
        }
    )


class TestProbeArchitecture:
    def test_matching_architectures_pass(self) -> None:
        fake_cfg = SimpleNamespace(architectures=["DemoForCausalLM"])
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_cfg):
            result = probe_architecture(_spec())
        assert result.passed is True
        assert "matched" in result.detail

    def test_mismatch_fails(self) -> None:
        fake_cfg = SimpleNamespace(architectures=["OtherForCausalLM"])
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_cfg):
            result = probe_architecture(_spec())
        assert result.passed is False
        assert "DemoForCausalLM" in result.detail
        assert "OtherForCausalLM" in result.detail

    def test_empty_architectures_fails(self) -> None:
        fake_cfg = SimpleNamespace(architectures=[])
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_cfg):
            result = probe_architecture(_spec())
        assert result.passed is False

    def test_load_error_fails_without_raising(self) -> None:
        with patch("transformers.AutoConfig.from_pretrained", side_effect=OSError("boom")):
            result = probe_architecture(_spec())
        assert result.passed is False
        assert "OSError" in result.detail

    def test_gated_repo_raises_gated_model_error(self) -> None:
        from unittest.mock import Mock as _Mock

        from huggingface_hub.errors import GatedRepoError

        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                side_effect=GatedRepoError("gated", response=_Mock()),
            ),
            pytest.raises(GatedModelError),
        ):
            probe_architecture(_spec())


class TestProbeChatTemplate:
    def test_template_present(self) -> None:
        tokenizer = SimpleNamespace(chat_template="{%- for m in messages %}...")
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ):
            result = probe_chat_template(_spec())
        assert result.passed is True

    def test_template_missing(self) -> None:
        tokenizer = SimpleNamespace(chat_template=None)
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ):
            result = probe_chat_template(_spec())
        assert result.passed is False


class TestProbeGgufArch:
    def test_skips_when_vendor_missing(self, tmp_path: Path) -> None:
        result = probe_gguf_arch_supported(_spec(), vendor_path=tmp_path / "absent")
        assert result.skipped is True
        assert result.passed is True

    def test_matches_registered_arch(self, tmp_path: Path) -> None:
        vendor = tmp_path / "llama.cpp"
        vendor.mkdir()
        (vendor / "convert_hf_to_gguf.py").write_text(
            '@Model.register("qwen2", "qwen3")\n@Model.register("demo")\nclass DemoModel: ...\n',
            encoding="utf-8",
        )
        result = probe_gguf_arch_supported(_spec(), vendor_path=vendor)
        assert result.passed is True
        assert result.detail.count("demo") >= 1

    def test_missing_arch_fails(self, tmp_path: Path) -> None:
        vendor = tmp_path / "llama.cpp"
        vendor.mkdir()
        (vendor / "convert_hf_to_gguf.py").write_text(
            '@Model.register("qwen2")\n@Model.register("llama")\n',
            encoding="utf-8",
        )
        result = probe_gguf_arch_supported(_spec(), vendor_path=vendor)
        assert result.passed is False
        assert "demo" in result.detail


class TestProbePretokenizerHash:
    def test_skips_when_table_missing(self, tmp_path: Path) -> None:
        result = probe_pretokenizer_hash(_spec(), hashes_path=tmp_path / "absent.json")
        assert result.skipped is True
        assert result.passed is True

    def test_known_label_passes(self, tmp_path: Path) -> None:
        hashes = tmp_path / "h.json"
        hashes.write_text(json.dumps(["demo", "qwen2", "llama-bpe"]), encoding="utf-8")
        result = probe_pretokenizer_hash(_spec(), hashes_path=hashes)
        assert result.passed is True

    def test_unknown_label_fails(self, tmp_path: Path) -> None:
        hashes = tmp_path / "h.json"
        hashes.write_text(json.dumps(["qwen2", "llama-bpe"]), encoding="utf-8")
        result = probe_pretokenizer_hash(_spec(), hashes_path=hashes)
        assert result.passed is False
        assert "demo" in result.detail

    def test_unreadable_table_fails(self, tmp_path: Path) -> None:
        hashes = tmp_path / "h.json"
        hashes.write_text("not json", encoding="utf-8")
        result = probe_pretokenizer_hash(_spec(), hashes_path=hashes)
        assert result.passed is False


class TestRunAll:
    def test_aggregates_all_four_probes(self) -> None:
        spec = _spec()
        fake_cfg = SimpleNamespace(architectures=["DemoForCausalLM"])
        tokenizer = SimpleNamespace(chat_template="tmpl")
        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=fake_cfg),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        ):
            report = run_all(spec)
        names = {r.name for r in report.results}
        assert names == {
            "architecture",
            "chat_template",
            "gguf_arch",
            "pretokenizer_hash",
        }
