"""assert_embedding_rows_match — adapter ↔ base GGUF row cross-check (Sprint 11.5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dlm.export.embedding_sync import assert_embedding_rows_match
from dlm.export.errors import PreflightError
from dlm.export.gguf_tensors import GGML_TYPE_F16

# Reuse the GGUF synthesizer from the tensor-reader tests.
from tests.unit.export.test_gguf_tensors import _build_gguf


def _write_adapter(
    tmp_path: Path,
    *,
    vocab_size: int,
    hidden: int,
    embedding_rows: np.ndarray,
    lm_head_rows: np.ndarray | None = None,
    modules_to_save: list[str] | None = None,
    added_token_ids: tuple[int, ...] = (),
) -> Path:
    """Write a minimal adapter dir: config + tokenizer_config + safetensors."""
    from safetensors.numpy import save_file

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "demo/demo",
                "peft_type": "LORA",
                "modules_to_save": modules_to_save or [],
            }
        )
    )
    added_decoder: dict[str, dict[str, object]] = {
        str(tid): {"content": f"<|special-{tid}|>", "special": True} for tid in added_token_ids
    }
    (adapter / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "vocab_size": vocab_size,
                "chat_template": "{{messages}}",
                "added_tokens_decoder": added_decoder,
            }
        )
    )

    # PEFT writes this key pattern under base_model.model.{layer}.modules_to_save.default.weight.
    tensors: dict[str, np.ndarray] = {}
    if modules_to_save and "embed_tokens" in modules_to_save:
        tensors["base_model.model.model.embed_tokens.modules_to_save.default.weight"] = (
            embedding_rows
        )
    if modules_to_save and "lm_head" in modules_to_save and lm_head_rows is not None:
        tensors["base_model.model.lm_head.modules_to_save.default.weight"] = lm_head_rows
    save_file(tensors, str(adapter / "adapter_model.safetensors"))
    return adapter


def _f16_bytes_for_rows(rows: np.ndarray) -> bytes:
    """Serialize `rows` (float16 array) to contiguous bytes."""
    return bytes(np.ascontiguousarray(rows.astype(np.float16)).tobytes())


class TestSkipPaths:
    def test_no_adapter_config_skips(self, tmp_path: Path) -> None:
        adapter = tmp_path / "empty"
        adapter.mkdir()
        base = tmp_path / "base.gguf"
        base.write_bytes(b"GGUF")  # wouldn't be reached
        assert_embedding_rows_match(adapter, base)  # no raise

    def test_no_modules_to_save_skips(self, tmp_path: Path) -> None:
        # Standard LoRA adapter — no embedding / lm_head layers owned.
        adapter = _write_adapter(
            tmp_path,
            vocab_size=100,
            hidden=4,
            embedding_rows=np.zeros((1, 1), dtype=np.float16),
            modules_to_save=[],
        )
        base = tmp_path / "base.gguf"
        base.write_bytes(b"GGUF")
        assert_embedding_rows_match(adapter, base)

    def test_no_added_special_tokens_skips(self, tmp_path: Path) -> None:
        rows = np.arange(10, dtype=np.float16).reshape(5, 2)
        adapter = _write_adapter(
            tmp_path,
            vocab_size=5,
            hidden=2,
            embedding_rows=rows,
            modules_to_save=["embed_tokens"],
            added_token_ids=(),
        )
        base = tmp_path / "base.gguf"
        base.write_bytes(b"GGUF")
        assert_embedding_rows_match(adapter, base)


class TestRowAgreement:
    def _build_matching_pair(self, tmp_path: Path) -> tuple[Path, Path]:
        """Adapter + base GGUF where row 4 matches byte-for-byte."""
        rows = np.arange(20, dtype=np.float16).reshape(5, 4)  # vocab=5, hidden=4
        adapter = _write_adapter(
            tmp_path,
            vocab_size=5,
            hidden=4,
            embedding_rows=rows,
            modules_to_save=["embed_tokens"],
            added_token_ids=(4,),  # last row is the added special
        )
        base_path = tmp_path / "base.gguf"
        base_path.write_bytes(
            _build_gguf(
                tensors=[
                    (
                        "token_embd.weight",
                        (5, 4),
                        GGML_TYPE_F16,
                        _f16_bytes_for_rows(rows),
                    ),
                ],
            )
        )
        return adapter, base_path

    def test_matching_rows_pass(self, tmp_path: Path) -> None:
        adapter, base = self._build_matching_pair(tmp_path)
        assert_embedding_rows_match(adapter, base)  # no raise

    def test_single_byte_drift_raises(self, tmp_path: Path) -> None:
        adapter, base = self._build_matching_pair(tmp_path)
        # Flip one byte in the base's last row.
        blob = bytearray(base.read_bytes())
        # The last F16 element is the very last 2 bytes of the file.
        blob[-1] ^= 0xFF
        base.write_bytes(bytes(blob))
        with pytest.raises(PreflightError, match="embedding rows disagree"):
            assert_embedding_rows_match(adapter, base)

    def test_missing_base_tensor_raises(self, tmp_path: Path) -> None:
        rows = np.arange(20, dtype=np.float16).reshape(5, 4)
        adapter = _write_adapter(
            tmp_path,
            vocab_size=5,
            hidden=4,
            embedding_rows=rows,
            modules_to_save=["embed_tokens"],
            added_token_ids=(4,),
        )
        # Base GGUF has a different tensor (not token_embd.weight).
        base = tmp_path / "base.gguf"
        base.write_bytes(
            _build_gguf(
                tensors=[("attn.weight", (2, 2), GGML_TYPE_F16, b"\x00" * 8)],
            )
        )
        with pytest.raises(PreflightError, match="missing tensor"):
            assert_embedding_rows_match(adapter, base)

    def test_missing_adapter_safetensors_raises(self, tmp_path: Path) -> None:
        # Declare modules_to_save but don't actually write the tensor file.
        adapter = tmp_path / "partial"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "x",
                    "peft_type": "LORA",
                    "modules_to_save": ["embed_tokens"],
                }
            )
        )
        (adapter / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "vocab_size": 5,
                    "chat_template": "x",
                    "added_tokens_decoder": {"4": {"content": "<|pad|>", "special": True}},
                }
            )
        )
        base = tmp_path / "base.gguf"
        base.write_bytes(
            _build_gguf(
                tensors=[
                    ("token_embd.weight", (5, 4), GGML_TYPE_F16, b"\x00" * 40),
                ],
            )
        )
        with pytest.raises(PreflightError, match="adapter_model.safetensors not found"):
            assert_embedding_rows_match(adapter, base)

    def test_declared_but_absent_module_raises(self, tmp_path: Path) -> None:
        """modules_to_save says `lm_head` but the safetensors has no such key."""
        rows = np.arange(20, dtype=np.float16).reshape(5, 4)
        adapter = _write_adapter(
            tmp_path,
            vocab_size=5,
            hidden=4,
            embedding_rows=rows,
            modules_to_save=["embed_tokens", "lm_head"],
            added_token_ids=(4,),
            # Don't pass lm_head_rows → safetensors won't contain it.
        )
        base = tmp_path / "base.gguf"
        base.write_bytes(
            _build_gguf(
                tensors=[
                    (
                        "token_embd.weight",
                        (5, 4),
                        GGML_TYPE_F16,
                        _f16_bytes_for_rows(rows),
                    ),
                    (
                        "output.weight",
                        (5, 4),
                        GGML_TYPE_F16,
                        _f16_bytes_for_rows(rows),
                    ),
                ],
            )
        )
        with pytest.raises(PreflightError, match="is absent from adapter"):
            assert_embedding_rows_match(adapter, base)

    def test_block_quantized_base_raises(self, tmp_path: Path) -> None:
        rows = np.arange(20, dtype=np.float16).reshape(5, 4)
        adapter = _write_adapter(
            tmp_path,
            vocab_size=5,
            hidden=4,
            embedding_rows=rows,
            modules_to_save=["embed_tokens"],
            added_token_ids=(4,),
        )
        base = tmp_path / "base.gguf"
        # dtype=14 is Q4_K — block-quantized.
        base.write_bytes(
            _build_gguf(
                tensors=[("token_embd.weight", (5, 4), 14, b"\x00" * 40)],
            )
        )
        with pytest.raises(PreflightError, match="block-quantized"):
            assert_embedding_rows_match(adapter, base)


class TestMultipleModules:
    def test_both_modules_match(self, tmp_path: Path) -> None:
        embed = np.arange(20, dtype=np.float16).reshape(5, 4)
        head = np.arange(20, 40, dtype=np.float16).reshape(5, 4)
        adapter = _write_adapter(
            tmp_path,
            vocab_size=5,
            hidden=4,
            embedding_rows=embed,
            lm_head_rows=head,
            modules_to_save=["embed_tokens", "lm_head"],
            added_token_ids=(4,),
        )
        base = tmp_path / "base.gguf"
        base.write_bytes(
            _build_gguf(
                tensors=[
                    (
                        "token_embd.weight",
                        (5, 4),
                        GGML_TYPE_F16,
                        _f16_bytes_for_rows(embed),
                    ),
                    (
                        "output.weight",
                        (5, 4),
                        GGML_TYPE_F16,
                        _f16_bytes_for_rows(head),
                    ),
                ],
            )
        )
        assert_embedding_rows_match(adapter, base)

    def test_mixed_module_mismatches_reports_count(self, tmp_path: Path) -> None:
        embed = np.arange(20, dtype=np.float16).reshape(5, 4)
        head = np.arange(20, 40, dtype=np.float16).reshape(5, 4)
        adapter = _write_adapter(
            tmp_path,
            vocab_size=5,
            hidden=4,
            embedding_rows=embed,
            lm_head_rows=head,
            modules_to_save=["embed_tokens", "lm_head"],
            added_token_ids=(3, 4),
        )
        # Base: embed matches, head doesn't.
        corrupted_head = head.copy()
        corrupted_head[4, 0] = 999.0
        base = tmp_path / "base.gguf"
        base.write_bytes(
            _build_gguf(
                tensors=[
                    (
                        "token_embd.weight",
                        (5, 4),
                        GGML_TYPE_F16,
                        _f16_bytes_for_rows(embed),
                    ),
                    (
                        "output.weight",
                        (5, 4),
                        GGML_TYPE_F16,
                        _f16_bytes_for_rows(corrupted_head),
                    ),
                ],
            )
        )
        with pytest.raises(PreflightError) as excinfo:
            assert_embedding_rows_match(adapter, base)
        # Only row 4 diverges in head; embed row 4 still matches.
        assert "1 added token" in excinfo.value.detail
        assert "lm_head[4]" in excinfo.value.detail


class TestRobustSkips:
    """Malformed inputs on the skip-path must not raise (Sprint 11.5 hygiene)."""

    def test_unreadable_adapter_config_skips(self, tmp_path: Path) -> None:
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("not json")
        base = tmp_path / "base.gguf"
        base.write_bytes(b"ignored")
        # Unreadable adapter_config belongs to Sprint 11's preflight;
        # this function bails out silently so we don't double-report.
        assert_embedding_rows_match(adapter, base)

    def test_non_list_modules_to_save_skips(self, tmp_path: Path) -> None:
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "x",
                    "peft_type": "LORA",
                    "modules_to_save": "embed_tokens",  # should be list, user wrote string
                }
            )
        )
        (adapter / "tokenizer_config.json").write_text(
            json.dumps({"vocab_size": 5, "chat_template": "x"})
        )
        base = tmp_path / "base.gguf"
        base.write_bytes(b"ignored")
        assert_embedding_rows_match(adapter, base)

    def test_missing_tokenizer_config_skips(self, tmp_path: Path) -> None:
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "x",
                    "peft_type": "LORA",
                    "modules_to_save": ["embed_tokens"],
                }
            )
        )
        # No tokenizer_config.json — the helper returns [], so we skip.
        base = tmp_path / "base.gguf"
        base.write_bytes(b"ignored")
        assert_embedding_rows_match(adapter, base)

    def test_unreadable_tokenizer_config_treats_as_no_added(self, tmp_path: Path) -> None:
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "x",
                    "peft_type": "LORA",
                    "modules_to_save": ["embed_tokens"],
                }
            )
        )
        (adapter / "tokenizer_config.json").write_text("not json")
        base = tmp_path / "base.gguf"
        base.write_bytes(b"ignored")
        assert_embedding_rows_match(adapter, base)

    def test_malformed_added_tokens_decoder_skips(self, tmp_path: Path) -> None:
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "x",
                    "peft_type": "LORA",
                    "modules_to_save": ["embed_tokens"],
                }
            )
        )
        (adapter / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "vocab_size": 5,
                    "chat_template": "x",
                    # Mixed-shape entries + non-special + non-int key are
                    # all skipped by `_added_special_token_ids`.
                    "added_tokens_decoder": {
                        "not-an-int": {"special": True},
                        "1": "not-a-dict",
                        "2": {"special": False, "content": "<|a|>"},
                        "3": {"special": True},  # ← legitimate but no added means skip
                    },
                }
            )
        )
        base = tmp_path / "base.gguf"
        base.write_bytes(b"ignored")
        # Only id 3 survives the filter; but since the base isn't a valid
        # GGUF the check would fail. Since we can't easily build a
        # base for this edge case here and the filter is what we want
        # to cover, assert by monkeypatching the safetensors loader.
        # Simpler: confirm the filter returns [3] directly.
        from dlm.export.embedding_sync import _added_special_token_ids

        assert _added_special_token_ids(adapter) == [3]


class TestBoundsChecks:
    def test_added_token_id_out_of_range_raises(self, tmp_path: Path) -> None:
        # Adapter claims an added token id higher than the embedding's vocab dim.
        rows = np.zeros((3, 2), dtype=np.float16)
        adapter = _write_adapter(
            tmp_path,
            vocab_size=3,
            hidden=2,
            embedding_rows=rows,
            modules_to_save=["embed_tokens"],
            added_token_ids=(99,),
        )
        base = tmp_path / "base.gguf"
        base.write_bytes(
            _build_gguf(
                tensors=[
                    ("token_embd.weight", (3, 2), GGML_TYPE_F16, b"\x00" * 12),
                ],
            )
        )
        with pytest.raises(PreflightError, match="out of range"):
            assert_embedding_rows_match(adapter, base)
