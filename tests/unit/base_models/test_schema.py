"""BaseModelSpec validation — every field constraint, positive + negative."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlm.base_models import BaseModelSpec


def _minimal(**overrides: object) -> BaseModelSpec:
    defaults: dict[str, object] = {
        "key": "demo-1b",
        "hf_id": "org/demo-1b",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "architecture": "DemoForCausalLM",
        "params": 1_000_000_000,
        "target_modules": ["q_proj", "v_proj"],
        "template": "chatml",
        "gguf_arch": "demo",
        "tokenizer_pre": "demo",
        "license_spdx": "Apache-2.0",
        "license_url": None,
        "requires_acceptance": False,
        "redistributable": True,
        "size_gb_fp16": 2.0,
        "context_length": 4096,
        "recommended_seq_len": 2048,
    }
    defaults.update(overrides)
    return BaseModelSpec.model_validate(defaults)


class TestRevisionValidator:
    @pytest.mark.parametrize(
        "bad_revision",
        [
            "",
            "short",
            "0123456789abcdef",  # 16 chars, too short
            "0" * 39,  # 39 chars
            "0" * 41,  # 41 chars
            "X" * 40,  # non-hex
            "0" * 39 + "G",  # trailing non-hex
            "0123456789abcdef0123456789ABCDEF01234567",  # upper-case rejected (lowercase-only)
        ],
    )
    def test_invalid_revision_rejected(self, bad_revision: str) -> None:
        with pytest.raises(ValidationError, match="40-char"):
            _minimal(revision=bad_revision)

    def test_valid_lowercase_40char_hex_accepted(self) -> None:
        spec = _minimal(revision="a" * 40)
        assert spec.revision == "a" * 40


class TestHfIdValidator:
    @pytest.mark.parametrize("bad_id", ["", "no-slash", "trailing/"])
    def test_invalid_hf_id_rejected(self, bad_id: str) -> None:
        # `trailing/` passes the `/ in value` gate but leading slash doesn't.
        # Pydantic's min_length catches empty. Bad ones without `/` raise.
        if "/" not in bad_id and bad_id:
            with pytest.raises(ValidationError, match="'org/name'"):
                _minimal(hf_id=bad_id)
        else:
            # Still a validation error (min_length or our rule).
            with pytest.raises(ValidationError):
                _minimal(hf_id=bad_id)

    def test_valid_org_slash_name(self) -> None:
        spec = _minimal(hf_id="org/name")
        assert spec.hf_id == "org/name"


class TestTargetModules:
    def test_empty_list_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _minimal(target_modules=[])

    def test_empty_string_in_list_rejected(self) -> None:
        with pytest.raises(ValidationError, match="empty strings"):
            _minimal(target_modules=["q_proj", ""])

    def test_valid_list_accepted(self) -> None:
        spec = _minimal(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        assert len(spec.target_modules) == 4


class TestLiteralConstraints:
    @pytest.mark.parametrize(
        "template", ["chatml", "smollm3", "olmo2", "llama3", "phi3", "mistral"]
    )
    def test_template_literals_accepted(self, template: str) -> None:
        spec = _minimal(template=template)
        assert spec.template == template

    def test_unknown_template_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _minimal(template="alpaca")


class TestNumericConstraints:
    def test_params_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _minimal(params=0)
        with pytest.raises(ValidationError):
            _minimal(params=-1)

    def test_size_gb_fp16_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _minimal(size_gb_fp16=0)

    def test_context_length_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _minimal(context_length=0)

    def test_context_length_effective_defaults_to_none(self) -> None:
        assert _minimal().context_length_effective is None

    def test_context_length_effective_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _minimal(context_length_effective=0)

    def test_context_length_effective_cannot_exceed_nominal(self) -> None:
        with pytest.raises(ValidationError, match="cannot exceed context_length"):
            _minimal(context_length=4096, context_length_effective=8192)


class TestSprint40Substrate:
    def test_reasoning_tuned_defaults_false(self) -> None:
        assert _minimal().reasoning_tuned is False

    def test_reasoning_tuned_accepts_true(self) -> None:
        assert _minimal(reasoning_tuned=True).reasoning_tuned is True

    def test_text_moe_modality_is_accepted(self) -> None:
        spec = _minimal(modality="text-moe")
        assert spec.modality == "text-moe"

    def test_text_moe_rejects_vl_plan(self) -> None:
        with pytest.raises(ValidationError, match="only valid with"):
            _minimal(
                modality="text-moe",
                vl_preprocessor_plan={
                    "target_size": (224, 224),
                    "image_token": "<image>",
                    "num_image_tokens": 256,
                },
            )

    def test_effective_context_length_defaults_to_nominal(self) -> None:
        spec = _minimal(context_length=4096)
        assert spec.effective_context_length == 4096

    def test_effective_context_length_honors_override(self) -> None:
        spec = _minimal(context_length=8192, context_length_effective=4096)
        assert spec.effective_context_length == 4096


class TestImmutability:
    def test_spec_is_frozen(self) -> None:
        spec = _minimal()
        with pytest.raises(ValidationError):
            spec.params = 2_000_000_000  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            BaseModelSpec.model_validate(
                {
                    "key": "x",
                    "hf_id": "org/x",
                    "revision": "0" * 40,
                    "architecture": "X",
                    "params": 1,
                    "target_modules": ["q"],
                    "template": "chatml",
                    "gguf_arch": "x",
                    "tokenizer_pre": "x",
                    "license_spdx": "MIT",
                    "redistributable": True,
                    "size_gb_fp16": 1.0,
                    "context_length": 1,
                    "recommended_seq_len": 1,
                    "surprise": "field",
                }
            )


class TestLicenseFields:
    def test_gated_redistributable_combinations(self) -> None:
        """Llama-style: gated + not redistributable. Qwen-style: permissive."""
        gated = _minimal(
            requires_acceptance=True,
            redistributable=False,
            license_spdx="Other",
            license_url="https://example.com/license",
        )
        assert gated.requires_acceptance is True
        assert gated.redistributable is False

        open_spec = _minimal(
            requires_acceptance=False,
            redistributable=True,
            license_spdx="Apache-2.0",
        )
        assert open_spec.requires_acceptance is False
        assert open_spec.redistributable is True
