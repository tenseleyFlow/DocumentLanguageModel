"""MLX backend helpers and lightweight backend-path coverage."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from dlm.base_models import BASE_MODELS
from dlm.inference.backends.mlx_backend import (
    MlxBackend,
    _apply_chat_template,
    _resolve_base_num_hidden_layers,
    stage_mlx_adapter_dir,
)
from dlm.inference.errors import AdapterNotFoundError
from dlm.inference.mlx_adapter import MlxConversionError


class TestResolveBaseNumHiddenLayers:
    def test_prefers_transformers_auto_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained",
            lambda hf_id, local_files_only=True: SimpleNamespace(num_hidden_layers=24),
        )
        assert _resolve_base_num_hidden_layers("org/demo") == 24

    def test_falls_back_to_cached_config_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snapshot = tmp_path / "snapshots" / ("a" * 40)
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(
            json.dumps({"num_hidden_layers": 18}), encoding="utf-8"
        )
        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained",
            lambda hf_id, local_files_only=True: SimpleNamespace(num_hidden_layers=None),
        )
        monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: str(snapshot))
        assert _resolve_base_num_hidden_layers("org/demo") == 18

    def test_cache_lookup_errors_raise_conversion_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained",
            lambda hf_id, local_files_only=True: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download",
            lambda **kwargs: (_ for _ in ()).throw(OSError("missing")),
        )
        with pytest.raises(MlxConversionError, match="could not resolve num_hidden_layers"):
            _resolve_base_num_hidden_layers("org/demo")

    def test_missing_num_hidden_layers_raises_conversion_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snapshot = tmp_path / "snapshots" / ("a" * 40)
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained",
            lambda hf_id, local_files_only=True: SimpleNamespace(num_hidden_layers=None),
        )
        monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: str(snapshot))
        with pytest.raises(MlxConversionError, match="has no usable num_hidden_layers"):
            _resolve_base_num_hidden_layers("org/demo")


class TestMlxBackend:
    def test_generate_before_load_raises(self) -> None:
        backend = MlxBackend(SimpleNamespace())
        with pytest.raises(RuntimeError, match="before load"):
            backend.generate("hello")

    def test_load_missing_adapter_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = MlxBackend(SimpleNamespace())
        monkeypatch.setattr(
            "dlm.inference.loader.resolve_adapter_path",
            lambda store, adapter_name=None: tmp_path / "missing",
        )
        with pytest.raises(AdapterNotFoundError, match="does not exist"):
            backend.load(BASE_MODELS["smollm2-135m"], SimpleNamespace(root=tmp_path))

    def test_load_generate_and_unload_happy_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        staged_dir = tmp_path / "staged"
        staged_dir.mkdir()
        # Real `stage_mlx_adapter_dir` writes this file; the post-load
        # assertion guard reads it to know which FQNs to verify, so the
        # test stub must mirror the on-disk shape.
        (staged_dir / "adapter_config.json").write_text(
            '{"lora_parameters": {"keys": ["self_attn.q_proj"]}}',
            encoding="utf-8",
        )

        backend = MlxBackend(SimpleNamespace())
        monkeypatch.setattr(
            "dlm.inference.loader.resolve_adapter_path",
            lambda store, adapter_name=None: adapter_dir,
        )
        monkeypatch.setattr(
            "dlm.inference.backends.mlx_backend.stage_mlx_adapter_dir",
            lambda peft_adapter_dir, dst_dir, *, base_hf_id: staged_dir,
        )
        # Stub the post-load assertion: real `mlx_lm.load` produces a
        # model with LoRA-wrapped layers, but here we hand back a
        # placeholder string. Bypassing the assertion keeps the rest
        # of the stubbed happy-path test intact.
        monkeypatch.setattr(
            "dlm.inference.backends.mlx_backend.assert_mlx_adapter_applied",
            lambda model, *, expected_keys: None,
        )

        fake_mlx = ModuleType("mlx_lm")
        fake_mlx.load = lambda hf_id, adapter_path: ("model", "tokenizer")
        fake_mlx.generate = lambda model, tokenizer, *, prompt, max_tokens, sampler, verbose: (
            "mlx output"
        )
        fake_sample_utils = ModuleType("mlx_lm.sample_utils")
        fake_sample_utils.make_sampler = lambda temp, top_p, top_k: {
            "temp": temp,
            "top_p": top_p,
            "top_k": top_k,
        }
        monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx)
        monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)

        backend.load(BASE_MODELS["smollm2-135m"], SimpleNamespace(root=tmp_path))
        assert backend.generate(
            "prompt", max_new_tokens=4, temperature=0.5, top_p=0.9, top_k=12
        ) == ("mlx output")
        backend.unload()
        assert backend._workdir is None
        assert backend._model is None
        assert backend._tokenizer is None


class TestApplyChatTemplate:
    """Mirrors PyTorch backend's `format_chat_prompt` for the MLX path."""

    def test_renders_via_tokenizer_when_template_present(self) -> None:
        captured: dict[str, object] = {}

        def _apply_chat_template_method(
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            captured["messages"] = messages
            captured["tokenize"] = tokenize
            captured["add_generation_prompt"] = add_generation_prompt
            return "<rendered>hello"

        tokenizer = SimpleNamespace(
            chat_template="{{messages}}",
            apply_chat_template=_apply_chat_template_method,
        )
        out = _apply_chat_template(tokenizer, "hello")
        assert out == "<rendered>hello"
        assert captured["messages"] == [{"role": "user", "content": "hello"}]
        assert captured["tokenize"] is False
        assert captured["add_generation_prompt"] is True

    def test_passes_through_when_no_chat_template(self) -> None:
        tokenizer = SimpleNamespace(chat_template=None)
        assert _apply_chat_template(tokenizer, "raw prompt") == "raw prompt"

    def test_passes_through_when_render_returns_non_string(self) -> None:
        """Defensive: HF wrappers occasionally return token lists; fall back to raw."""
        tokenizer = SimpleNamespace(
            chat_template="{{messages}}",
            apply_chat_template=lambda *_a, **_k: [1, 2, 3],
        )
        assert _apply_chat_template(tokenizer, "raw prompt") == "raw prompt"


class TestStageMlxAdapterDir:
    def test_unreadable_adapter_config_raises_conversion_error(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake")
        (adapter_dir / "adapter_config.json").mkdir()

        with pytest.raises(MlxConversionError, match="cannot read .*adapter_config.json"):
            stage_mlx_adapter_dir(
                adapter_dir,
                tmp_path / "staged",
                base_hf_id=BASE_MODELS["smollm2-135m"].hf_id,
            )
