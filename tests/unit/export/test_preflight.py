"""Preflight checks — adapter config, tokenizer vocab, chat template."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dlm.base_models import BASE_MODELS
from dlm.export.errors import PreflightError
from dlm.export.preflight import (
    check_adapter_config,
    check_chat_template,
    check_pretokenizer_fingerprint,
    check_tokenizer_vocab,
    check_vl_target_modules_lm_only,
    check_was_adapter_qlora,
)

_SPEC = BASE_MODELS["smollm2-135m"]


def _write_adapter_config(dir_: Path, **overrides: object) -> None:
    data = {"base_model_name_or_path": _SPEC.hf_id, "peft_type": "LORA"}
    data.update(overrides)
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "adapter_config.json").write_text(json.dumps(data))


def _write_tokenizer_config(dir_: Path, **overrides: object) -> None:
    data: dict[str, object] = {"vocab_size": 32000, "chat_template": "{{messages}}"}
    data.update(overrides)
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "tokenizer_config.json").write_text(json.dumps(data))


def _write_pinned_versions(dir_: Path, *, bnb: str | None) -> None:
    data = {"torch": "2.4.0", "bitsandbytes": bnb}
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "pinned_versions.json").write_text(json.dumps(data))


class TestAdapterConfig:
    def test_matching_base_ok(self, tmp_path: Path) -> None:
        _write_adapter_config(tmp_path)
        check_adapter_config(tmp_path, _SPEC)

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PreflightError, match="adapter_config"):
            check_adapter_config(tmp_path, _SPEC)

    def test_mismatched_base_raises(self, tmp_path: Path) -> None:
        _write_adapter_config(tmp_path, base_model_name_or_path="other/base")
        with pytest.raises(PreflightError, match="was trained against"):
            check_adapter_config(tmp_path, _SPEC)

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        (tmp_path / "adapter_config.json").write_text("not json {{{")
        with pytest.raises(PreflightError, match="cannot parse"):
            check_adapter_config(tmp_path, _SPEC)


class TestTokenizerVocab:
    def test_vocab_size_from_config(self, tmp_path: Path) -> None:
        _write_tokenizer_config(tmp_path, vocab_size=50257)
        assert check_tokenizer_vocab(tmp_path) == 50257

    def test_fallback_to_tokenizer_json(self, tmp_path: Path) -> None:
        _write_tokenizer_config(tmp_path)
        # Remove vocab_size from config, plant tokenizer.json with model.vocab
        cfg = json.loads((tmp_path / "tokenizer_config.json").read_text())
        del cfg["vocab_size"]
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(cfg))
        (tmp_path / "tokenizer.json").write_text(
            json.dumps({"model": {"vocab": {str(i): i for i in range(5000)}}})
        )
        assert check_tokenizer_vocab(tmp_path) == 5000

    def test_missing_tokenizer_config_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PreflightError, match="tokenizer metadata capture") as exc_info:
            check_tokenizer_vocab(tmp_path)
        assert "Sprint" not in str(exc_info.value)

    def test_malformed_config_raises(self, tmp_path: Path) -> None:
        (tmp_path / "tokenizer_config.json").write_text("not json {{{")
        with pytest.raises(PreflightError, match="cannot parse"):
            check_tokenizer_vocab(tmp_path)

    def test_no_vocab_info_anywhere_raises(self, tmp_path: Path) -> None:
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({}))
        with pytest.raises(PreflightError, match="cannot determine vocab"):
            check_tokenizer_vocab(tmp_path)

    def test_malformed_tokenizer_json_raises(self, tmp_path: Path) -> None:
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{{messages}}"})
        )
        (tmp_path / "tokenizer.json").write_text("not json {{{")
        with pytest.raises(PreflightError, match="cannot parse"):
            check_tokenizer_vocab(tmp_path)


class TestChatTemplate:
    def test_present_ok(self, tmp_path: Path) -> None:
        _write_tokenizer_config(tmp_path, chat_template="{{messages}}")
        check_chat_template(tmp_path)

    def test_missing_raises_by_default(self, tmp_path: Path) -> None:
        _write_tokenizer_config(tmp_path, chat_template="")
        with pytest.raises(PreflightError, match="chat_template"):
            check_chat_template(tmp_path)

    def test_whitespace_only_template_raises(self, tmp_path: Path) -> None:
        _write_tokenizer_config(tmp_path, chat_template="   ")
        with pytest.raises(PreflightError):
            check_chat_template(tmp_path)

    def test_required_false_skips_check(self, tmp_path: Path) -> None:
        # No tokenizer_config.json at all — and the check is skipped.
        check_chat_template(tmp_path, required=False)

    def test_missing_file_raises_when_required(self, tmp_path: Path) -> None:
        with pytest.raises(PreflightError, match="missing"):
            check_chat_template(tmp_path, required=True)

    def test_malformed_config_raises(self, tmp_path: Path) -> None:
        (tmp_path / "tokenizer_config.json").write_text("not json {{{")
        with pytest.raises(PreflightError, match="cannot parse"):
            check_chat_template(tmp_path, required=True)


class TestQloraFlag:
    def test_missing_file_returns_false(self, tmp_path: Path) -> None:
        assert check_was_adapter_qlora(tmp_path) is False

    def test_missing_training_run_falls_back_to_pinned_versions(self, tmp_path: Path) -> None:
        _write_pinned_versions(tmp_path, bnb="0.43.1")
        assert check_was_adapter_qlora(tmp_path) is True

    def test_true_flag_returns_true(self, tmp_path: Path) -> None:
        (tmp_path / "training_run.json").write_text(json.dumps({"use_qlora": True}))
        assert check_was_adapter_qlora(tmp_path) is True

    def test_false_flag_returns_false(self, tmp_path: Path) -> None:
        (tmp_path / "training_run.json").write_text(json.dumps({"use_qlora": False}))
        assert check_was_adapter_qlora(tmp_path) is False

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        """Corrupt `training_run.json` must not silently bypass the pitfall-3 merge gate."""
        _write_pinned_versions(tmp_path, bnb="0.43.1")
        (tmp_path / "training_run.json").write_text("not json")
        with pytest.raises(PreflightError, match="training_run_json"):
            check_was_adapter_qlora(tmp_path)


class TestPretokenizerFingerprint:
    def test_failed_probe_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "dlm.base_models.probes.probe_pretokenizer_hash",
            lambda _spec: SimpleNamespace(skipped=False, passed=False, detail="mismatch"),
        )

        with pytest.raises(PreflightError, match="pre-tokenizer fingerprint mismatch"):
            check_pretokenizer_fingerprint(_SPEC)


class TestVlTargetModulesLmOnly:
    def test_missing_config_is_noop(self, tmp_path: Path) -> None:
        check_vl_target_modules_lm_only(tmp_path)

    def test_malformed_config_is_noop(self, tmp_path: Path) -> None:
        (tmp_path / "adapter_config.json").write_text("not json {{{")
        check_vl_target_modules_lm_only(tmp_path)

    def test_string_pattern_target_modules_is_noop(self, tmp_path: Path) -> None:
        _write_adapter_config(tmp_path, target_modules=".*q_proj.*")
        check_vl_target_modules_lm_only(tmp_path)

    def test_vision_targets_raise(self, tmp_path: Path) -> None:
        _write_adapter_config(tmp_path, target_modules=["q_proj", "vision_tower.block.0"])
        with pytest.raises(PreflightError, match="vision-tower modules"):
            check_vl_target_modules_lm_only(tmp_path)
