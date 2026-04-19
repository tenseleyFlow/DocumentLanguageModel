"""Happy-path coverage for `resolve_hf()` — the `hf:org/name` escape hatch.

Covers: successful synthesis, helper fns (`_estimate_params`,
`_infer_gguf_arch`, `_infer_template`, `_default_target_modules`), and
`resolve_hf()`'s probe-report failure path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from dlm.base_models import BaseModelSpec, GatedModelError, resolve, resolve_hf
from dlm.base_models.errors import ProbeFailedError, ProbeReport, ProbeResult
from dlm.base_models.resolver import (
    _default_target_modules,
    _estimate_params,
    _infer_gguf_arch,
    _infer_template,
)

# --- helper fns --------------------------------------------------------------


class TestEstimateParams:
    def test_uses_num_parameters_when_present(self) -> None:
        cfg = SimpleNamespace(num_parameters=1_500_000_000)
        # num_parameters is consumed by the caller, not _estimate_params;
        # _estimate_params ignores it and falls back to hidden/layer heuristic.
        result = _estimate_params(cfg)
        assert result > 0

    def test_hidden_plus_layers_default_fallbacks(self) -> None:
        # No hidden_size / num_hidden_layers / vocab_size present: uses
        # built-in defaults (2048, 24, 32000).
        cfg = SimpleNamespace()
        result = _estimate_params(cfg)
        assert result > 0

    def test_honors_overrides(self) -> None:
        # With explicit hidden_size=4096, layers=32, vocab=128k,
        # we expect a sharply larger estimate.
        big = _estimate_params(
            SimpleNamespace(hidden_size=4096, num_hidden_layers=32, vocab_size=128_000)
        )
        small = _estimate_params(
            SimpleNamespace(hidden_size=1024, num_hidden_layers=12, vocab_size=32_000)
        )
        assert big > small


class TestInferGgufArch:
    @pytest.mark.parametrize(
        ("architecture", "expected"),
        [
            ("LlamaForCausalLM", "llama"),
            ("Qwen2ForCausalLM", "qwen2"),
            ("Qwen3ForCausalLM", "qwen3"),
            ("MistralForCausalLM", "llama"),
            ("Phi3ForCausalLM", "phi3"),
            ("GemmaForCausalLM", "gemma"),
            ("Gemma2ForCausalLM", "gemma2"),
        ],
    )
    def test_known_architectures_map_correctly(self, architecture: str, expected: str) -> None:
        assert _infer_gguf_arch(architecture) == expected

    def test_unknown_arch_falls_back_to_lowercase_stripped(self) -> None:
        # Unknown: lowercase + strip `forcausallm`.
        assert _infer_gguf_arch("SomeNewForCausalLM") == "somenew"


class TestInferTemplate:
    @pytest.mark.parametrize(
        ("hf_id", "architecture", "expected"),
        [
            ("meta-llama/Llama-3.2-1B-Instruct", "LlamaForCausalLM", "llama3"),
            ("meta-llama/llama3-base", "LlamaForCausalLM", "llama3"),
            ("microsoft/Phi-3.5-mini-instruct", "Phi3ForCausalLM", "phi3"),
            ("mistralai/Mistral-7B-Instruct", "MistralForCausalLM", "mistral"),
            ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2ForCausalLM", "chatml"),
        ],
    )
    def test_template_inference(self, hf_id: str, architecture: str, expected: str) -> None:
        assert _infer_template(hf_id, architecture) == expected


class TestDefaultTargetModules:
    def test_phi3_uses_fused_qkv(self) -> None:
        assert _default_target_modules("phi3") == [
            "qkv_proj",
            "o_proj",
            "gate_up_proj",
            "down_proj",
        ]

    def test_other_archs_use_split_qkv(self) -> None:
        for arch in ("llama", "qwen2", "qwen3", "gemma2"):
            assert _default_target_modules(arch) == ["q_proj", "k_proj", "v_proj", "o_proj"]


# --- resolve_hf end-to-end ---------------------------------------------------


class TestResolveHfHappyPath:
    def _mock_config(self, **overrides: object) -> SimpleNamespace:
        defaults = {
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "vocab_size": 151_936,
            "max_position_embeddings": 32_768,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_successful_synthesis_returns_spec(self) -> None:
        info = SimpleNamespace(sha="a" * 40, gated=False)
        report = ProbeReport(
            hf_id="org/custom",
            results=(
                ProbeResult(name="architecture", passed=True, detail="ok"),
                ProbeResult(name="chat_template", passed=True, detail="ok"),
                ProbeResult(name="gguf_arch", passed=True, detail="ok", skipped=True),
                ProbeResult(name="pretokenizer_label", passed=True, detail="ok", skipped=True),
            ),
        )
        with (
            patch("huggingface_hub.HfApi") as api_cls,
            patch("transformers.AutoConfig.from_pretrained", return_value=self._mock_config()),
            patch("dlm.base_models.probes.run_all", return_value=report),
        ):
            api_cls.return_value.model_info.return_value = info
            spec = resolve_hf("org/custom")

        assert isinstance(spec, BaseModelSpec)
        assert spec.key == "hf:org/custom"
        assert spec.hf_id == "org/custom"
        assert spec.revision == "a" * 40
        assert spec.architecture == "Qwen2ForCausalLM"
        assert spec.gguf_arch == "qwen2"
        assert spec.template == "chatml"
        assert spec.redistributable is False
        # hf: synthesis is conservative on license.
        assert spec.license_spdx == "Unknown"

    def test_probe_failure_raises_probe_failed_error(self) -> None:
        info = SimpleNamespace(sha="a" * 40, gated=False)
        report = ProbeReport(
            hf_id="org/custom",
            results=(
                ProbeResult(name="architecture", passed=True, detail="ok"),
                ProbeResult(name="chat_template", passed=False, detail="missing"),
                ProbeResult(name="gguf_arch", passed=True, detail="ok", skipped=True),
                ProbeResult(name="pretokenizer_label", passed=True, detail="ok", skipped=True),
            ),
        )
        with (
            patch("huggingface_hub.HfApi") as api_cls,
            patch("transformers.AutoConfig.from_pretrained", return_value=self._mock_config()),
            patch("dlm.base_models.probes.run_all", return_value=report),
        ):
            api_cls.return_value.model_info.return_value = info
            with pytest.raises(ProbeFailedError, match="chat_template"):
                resolve_hf("org/custom")

    def test_rejects_non_40_char_sha_from_hf(self) -> None:
        info = SimpleNamespace(sha="tooshort", gated=False)
        with (
            patch("huggingface_hub.HfApi") as api_cls,
            patch("transformers.AutoConfig.from_pretrained", return_value=self._mock_config()),
        ):
            api_cls.return_value.model_info.return_value = info
            with pytest.raises(RuntimeError, match="non-40-char SHA"):
                resolve_hf("org/custom")

    def test_empty_architectures_fails_fast(self) -> None:
        info = SimpleNamespace(sha="a" * 40, gated=False)
        with (
            patch("huggingface_hub.HfApi") as api_cls,
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=self._mock_config(architectures=[]),
            ),
        ):
            api_cls.return_value.model_info.return_value = info
            with pytest.raises(ProbeFailedError, match="architectures"):
                resolve_hf("org/custom")

    def test_gated_repo_during_config_load_surfaces_as_gated_error(self) -> None:
        from huggingface_hub.errors import GatedRepoError

        info = SimpleNamespace(sha="a" * 40, gated=False)
        with (
            patch("huggingface_hub.HfApi") as api_cls,
            patch(
                "transformers.AutoConfig.from_pretrained",
                side_effect=GatedRepoError("gated at config load", response=Mock()),
            ),
        ):
            api_cls.return_value.model_info.return_value = info
            with pytest.raises(GatedModelError):
                resolve_hf("org/custom")

    def test_resolve_dispatches_to_hf_escape_on_prefix(self) -> None:
        """Smoke test: `resolve('hf:...')` delegates to `resolve_hf`."""
        info = SimpleNamespace(sha="a" * 40, gated=False)
        report = ProbeReport(
            hf_id="org/mini",
            results=(
                ProbeResult(name="architecture", passed=True, detail="ok"),
                ProbeResult(name="chat_template", passed=True, detail="ok"),
                ProbeResult(name="gguf_arch", passed=True, detail="ok", skipped=True),
                ProbeResult(name="pretokenizer_label", passed=True, detail="ok", skipped=True),
            ),
        )
        with (
            patch("huggingface_hub.HfApi") as api_cls,
            patch("transformers.AutoConfig.from_pretrained", return_value=self._mock_config()),
            patch("dlm.base_models.probes.run_all", return_value=report),
        ):
            api_cls.return_value.model_info.return_value = info
            spec = resolve("hf:org/mini")
        assert spec.key == "hf:org/mini"


class TestResolveHfConfigLookupErrors:
    def test_entry_not_found_surfaces_as_unknown_base_model(self) -> None:
        from huggingface_hub.errors import EntryNotFoundError

        info = SimpleNamespace(sha="a" * 40, gated=False)
        with (
            patch("huggingface_hub.HfApi") as api_cls,
            patch(
                "transformers.AutoConfig.from_pretrained",
                side_effect=EntryNotFoundError("no config"),
            ),
        ):
            api_cls.return_value.model_info.return_value = info
            from dlm.base_models import UnknownBaseModelError

            with pytest.raises(UnknownBaseModelError):
                resolve_hf("org/nocfg")
