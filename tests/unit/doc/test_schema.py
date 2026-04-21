"""Schema validation — positive + negative cases for every constraint."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlm.doc.schema import (
    CURRENT_SCHEMA_VERSION,
    AdapterConfig,
    AudioConfig,
    CptConfig,
    DlmFrontmatter,
    ExportConfig,
    PreferenceConfig,
    PreferenceHyperparams,
    TrainingConfig,
)

VALID_ULID = "01HZ4X7TGZM3J1A2B3C4D5E6F7"


class TestTrainingConfigDefaults:
    def test_default_instance_has_expected_values(self) -> None:
        t = TrainingConfig()
        assert t.adapter == "lora"
        assert t.lora_r == 8
        assert t.lora_alpha == 16
        assert t.lora_dropout == 0.05
        assert t.target_modules == "auto"
        assert t.sequence_len == 2048
        assert t.micro_batch_size == "auto"
        assert t.grad_accum == "auto"
        assert t.learning_rate == pytest.approx(2e-4)
        assert t.num_epochs == 3
        assert t.optimizer == "adamw_torch"
        assert t.lr_scheduler == "cosine"
        assert t.warmup_ratio == pytest.approx(0.1)
        assert t.precision is None
        assert t.seed == 42

    def test_frozen_model_rejects_mutation(self) -> None:
        t = TrainingConfig()
        with pytest.raises(ValidationError):
            t.lora_r = 16  # type: ignore[misc]


class TestTrainingConfigConstraints:
    @pytest.mark.parametrize("bad", [0, -1, 257])
    def test_lora_r_out_of_range(self, bad: int) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(lora_r=bad)

    @pytest.mark.parametrize("bad", [-0.01, 0.51])
    def test_lora_dropout_out_of_range(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(lora_dropout=bad)

    @pytest.mark.parametrize("bad", [63, 32769])
    def test_sequence_len_out_of_range(self, bad: int) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(sequence_len=bad)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_num_epochs_must_be_ge_1(self, bad: int) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(num_epochs=bad)

    def test_learning_rate_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0.0)

    @pytest.mark.parametrize("bad", [-0.01, 0.51])
    def test_warmup_ratio_out_of_range(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(warmup_ratio=bad)

    @pytest.mark.parametrize("value", ["bf16", "fp16", "fp32"])
    def test_precision_accepts_valid_values(self, value: str) -> None:
        t = TrainingConfig(precision=value)  # type: ignore[arg-type]
        assert t.precision == value

    def test_precision_defaults_to_none(self) -> None:
        assert TrainingConfig().precision is None

    @pytest.mark.parametrize("bad", ["float32", "FP16", "int8", ""])
    def test_precision_rejects_invalid(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(precision=bad)  # type: ignore[arg-type]

    def test_adapter_literal_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(adapter="full")  # type: ignore[arg-type]

    def test_optimizer_literal_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(optimizer="lion")  # type: ignore[arg-type]

    def test_lr_scheduler_literal_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(lr_scheduler="exponential")  # type: ignore[arg-type]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig.model_validate({"adapter": "lora", "rubbish": 1})

    def test_micro_batch_size_auto_or_positive_int(self) -> None:
        TrainingConfig(micro_batch_size="auto")
        TrainingConfig(micro_batch_size=4)
        with pytest.raises(ValidationError):
            TrainingConfig(micro_batch_size=0)
        with pytest.raises(ValidationError):
            TrainingConfig(micro_batch_size="high")  # type: ignore[arg-type]

    def test_grad_accum_auto_or_positive_int(self) -> None:
        TrainingConfig(grad_accum="auto")
        TrainingConfig(grad_accum=2)
        with pytest.raises(ValidationError):
            TrainingConfig(grad_accum=-1)

    def test_target_modules_accepts_auto_or_list(self) -> None:
        TrainingConfig(target_modules="auto")
        TrainingConfig(target_modules=["q_proj", "v_proj"])
        with pytest.raises(ValidationError):
            TrainingConfig(target_modules="all-linear")  # type: ignore[arg-type]


class TestPreferenceConfig:
    def test_default_instance_is_disabled(self) -> None:
        p = PreferenceConfig()
        assert p.enabled is False
        assert p.method == "dpo"
        assert p.hyperparams.beta == pytest.approx(0.1)
        assert p.hyperparams.alpha == pytest.approx(0.1)
        assert p.hyperparams.learning_rate == pytest.approx(5e-6)
        assert p.hyperparams.num_epochs == 1
        assert p.loss_type == "sigmoid"
        assert p.reference == "pre_adapter"

    def test_frozen_model_rejects_mutation(self) -> None:
        p = PreferenceConfig()
        with pytest.raises(ValidationError):
            p.enabled = True  # type: ignore[misc]

    @pytest.mark.parametrize("bad", [-0.01, 1.01])
    def test_hyperparams_beta_out_of_range(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            PreferenceHyperparams(beta=bad)

    @pytest.mark.parametrize("bad", [-0.01, 1.01])
    def test_hyperparams_alpha_out_of_range(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            PreferenceHyperparams(alpha=bad)

    def test_hyperparams_learning_rate_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            PreferenceHyperparams(learning_rate=0.0)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_hyperparams_num_epochs_must_be_ge_1(self, bad: int) -> None:
        with pytest.raises(ValidationError):
            PreferenceHyperparams(num_epochs=bad)

    def test_method_literal_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            PreferenceConfig(method="kto")  # type: ignore[arg-type]

    def test_loss_type_literal_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            PreferenceConfig(loss_type="kto")  # type: ignore[arg-type]

    def test_reference_literal_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            PreferenceConfig(reference="sft_adapter")  # type: ignore[arg-type]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            PreferenceConfig.model_validate({"enabled": True, "rubbish": 1})


class TestAudioConfig:
    """v12 training.audio.auto_resample — default False preserves v11 behavior."""

    def test_default_instance_refuses_resample(self) -> None:
        a = AudioConfig()
        assert a.auto_resample is False

    def test_accepts_true(self) -> None:
        a = AudioConfig(auto_resample=True)
        assert a.auto_resample is True

    def test_frozen(self) -> None:
        a = AudioConfig()
        with pytest.raises(ValidationError):
            a.auto_resample = True  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            AudioConfig.model_validate({"auto_resample": False, "sr": 16_000})


class TestTrainingConfigAudioSubfield:
    def test_default_training_has_audio_disabled(self) -> None:
        t = TrainingConfig()
        assert isinstance(t.audio, AudioConfig)
        assert t.audio.auto_resample is False

    def test_accepts_nested_dict(self) -> None:
        t = TrainingConfig.model_validate({"audio": {"auto_resample": True}})
        assert t.audio.auto_resample is True

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig.model_validate(
                {"audio": {"auto_resample": True, "unknown": 1}}
            )


class TestTrainingConfigPreferenceSubfield:
    def test_default_training_has_disabled_preference(self) -> None:
        t = TrainingConfig()
        assert isinstance(t.preference, PreferenceConfig)
        assert t.preference.enabled is False

    def test_accepts_nested_dict_for_preference(self) -> None:
        t = TrainingConfig.model_validate(
            {"preference": {"enabled": True, "hyperparams": {"beta": 0.2}}}
        )
        assert t.preference.enabled is True
        assert t.preference.hyperparams.beta == pytest.approx(0.2)

    def test_rejects_unknown_field_inside_preference(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig.model_validate({"preference": {"enabled": True, "rubbish": 1}})


class TestCptConfig:
    def test_default_instance(self) -> None:
        c = CptConfig()
        assert c.schedule == "auto"
        assert c.embed_warmup_steps == 0

    def test_frozen_model_rejects_mutation(self) -> None:
        c = CptConfig()
        with pytest.raises(ValidationError):
            c.embed_warmup_steps = 10  # type: ignore[misc]

    @pytest.mark.parametrize("value", ["auto", "dapt", "sft"])
    def test_schedule_accepts_known_values(self, value: str) -> None:
        CptConfig(schedule=value)  # type: ignore[arg-type]

    def test_schedule_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            CptConfig(schedule="warmup")  # type: ignore[arg-type]

    def test_embed_warmup_steps_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            CptConfig(embed_warmup_steps=-1)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            CptConfig.model_validate({"schedule": "auto", "rubbish": 1})


class TestTrainingConfigCptSubfield:
    def test_default_training_has_auto_cpt(self) -> None:
        t = TrainingConfig()
        assert isinstance(t.cpt, CptConfig)
        assert t.cpt.schedule == "auto"
        assert t.cpt.embed_warmup_steps == 0

    def test_accepts_nested_dict_for_cpt(self) -> None:
        t = TrainingConfig.model_validate({"cpt": {"schedule": "dapt", "embed_warmup_steps": 200}})
        assert t.cpt.schedule == "dapt"
        assert t.cpt.embed_warmup_steps == 200

    def test_rejects_unknown_field_inside_cpt(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig.model_validate({"cpt": {"schedule": "dapt", "rubbish": 1}})


class TestAdapterConfig:
    def test_default_instance(self) -> None:
        a = AdapterConfig()
        assert a.adapter == "lora"
        assert a.lora_r == 8
        assert a.lora_alpha == 16
        assert a.lora_dropout == pytest.approx(0.05)
        assert a.target_modules == "auto"
        assert a.learning_rate == pytest.approx(2e-4)

    def test_frozen(self) -> None:
        a = AdapterConfig()
        with pytest.raises(ValidationError):
            a.lora_r = 32  # type: ignore[misc]

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            AdapterConfig.model_validate({"rubbish": 1})

    def test_accepts_explicit_target_modules(self) -> None:
        a = AdapterConfig(target_modules=["q_proj", "v_proj"])
        assert a.target_modules == ["q_proj", "v_proj"]


class TestNamedAdapters:
    def test_default_adapters_is_none(self) -> None:
        assert TrainingConfig().adapters is None

    def test_accepts_named_adapters(self) -> None:
        t = TrainingConfig.model_validate(
            {
                "adapters": {
                    "knowledge": {"lora_r": 8},
                    "tone": {"lora_r": 4, "target_modules": ["q_proj", "v_proj"]},
                }
            }
        )
        assert t.adapters is not None
        assert set(t.adapters) == {"knowledge", "tone"}
        assert t.adapters["tone"].lora_r == 4
        assert t.adapters["tone"].target_modules == ["q_proj", "v_proj"]

    def test_empty_adapters_block_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one adapter"):
            TrainingConfig.model_validate({"adapters": {}})

    @pytest.mark.parametrize(
        "name",
        ["", "A", "1bad", "bad-name", "bad name", "bad.name", "_leading", "way_too_long_name_" * 3],
    )
    def test_invalid_adapter_names_rejected(self, name: str) -> None:
        with pytest.raises(ValidationError, match="not a valid adapter name"):
            TrainingConfig.model_validate({"adapters": {name: {}}})

    @pytest.mark.parametrize("name", ["default", "knowledge", "tone_v2", "a", "x9"])
    def test_valid_adapter_names_accepted(self, name: str) -> None:
        t = TrainingConfig.model_validate({"adapters": {name: {}}})
        assert t.adapters is not None
        assert name in t.adapters

    def test_flat_adapter_field_with_block_rejected(self) -> None:
        with pytest.raises(ValidationError, match="flat per-adapter fields"):
            TrainingConfig.model_validate(
                {
                    "adapter": "qlora",
                    "adapters": {"knowledge": {}},
                }
            )

    def test_flat_lora_r_with_block_rejected(self) -> None:
        with pytest.raises(ValidationError, match="flat per-adapter fields"):
            TrainingConfig.model_validate({"lora_r": 32, "adapters": {"knowledge": {}}})

    def test_flat_learning_rate_with_block_rejected(self) -> None:
        with pytest.raises(ValidationError, match="flat per-adapter fields"):
            TrainingConfig.model_validate({"learning_rate": 1e-3, "adapters": {"tone": {}}})

    def test_top_level_shared_knobs_allowed_alongside_block(self) -> None:
        # seed, num_epochs, sequence_len, etc. are explicitly shared
        # across adapters — setting them next to the block is fine.
        t = TrainingConfig.model_validate(
            {
                "seed": 7,
                "num_epochs": 2,
                "sequence_len": 1024,
                "adapters": {"knowledge": {}},
            }
        )
        assert t.seed == 7
        assert t.num_epochs == 2
        assert t.sequence_len == 1024


class TestExportConfig:
    def test_default_quant(self) -> None:
        assert ExportConfig().default_quant == "Q4_K_M"

    @pytest.mark.parametrize("level", ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"])
    def test_accepts_known_quant_levels(self, level: str) -> None:
        ExportConfig(default_quant=level)  # type: ignore[arg-type]

    def test_rejects_unknown_quant_level(self) -> None:
        with pytest.raises(ValidationError):
            ExportConfig(default_quant="Q3_K_L")  # type: ignore[arg-type]

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            ExportConfig.model_validate({"default_quant": "Q4_K_M", "legacy": True})


class TestDlmFrontmatterForwardVersion:
    """Audit-07 M6: `model_validate` rejects forward-dated docs directly,
    not just through the `versioned` dispatcher."""

    def test_version_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="dlm_version must be"):
            DlmFrontmatter.model_validate(
                {
                    "dlm_id": VALID_ULID,
                    "base_model": "smollm2-135m",
                    "dlm_version": 0,
                }
            )

    def test_forward_version_rejected(self) -> None:
        from dlm.doc.schema import CURRENT_SCHEMA_VERSION

        with pytest.raises(ValidationError, match="newer than this CLI"):
            DlmFrontmatter.model_validate(
                {
                    "dlm_id": VALID_ULID,
                    "base_model": "smollm2-135m",
                    "dlm_version": CURRENT_SCHEMA_VERSION + 1,
                }
            )

    def test_current_version_accepted(self) -> None:
        from dlm.doc.schema import CURRENT_SCHEMA_VERSION

        fm = DlmFrontmatter.model_validate(
            {
                "dlm_id": VALID_ULID,
                "base_model": "smollm2-135m",
                "dlm_version": CURRENT_SCHEMA_VERSION,
            }
        )
        assert fm.dlm_version == CURRENT_SCHEMA_VERSION


class TestDlmFrontmatter:
    def test_minimal_valid(self) -> None:
        fm = DlmFrontmatter(dlm_id=VALID_ULID, base_model="smollm2-135m")
        assert fm.dlm_version == CURRENT_SCHEMA_VERSION
        assert fm.training == TrainingConfig()
        assert fm.export == ExportConfig()
        assert fm.system_prompt is None

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            DlmFrontmatter.model_validate({})

    def test_base_model_must_not_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            DlmFrontmatter(dlm_id=VALID_ULID, base_model="")

    @pytest.mark.parametrize(
        "bad_id",
        [
            "",
            "not-a-ulid",
            "01HZ4X7TGZM3J1A2B3C4D5E6F",  # 25 chars
            "01HZ4X7TGZM3J1A2B3C4D5E6FGG",  # 27 chars
            "01HZ4X7TGZM3J1A2B3C4D5E6FI",  # contains I (forbidden)
            "01HZ4X7TGZM3J1A2B3C4D5E6FL",  # contains L
            "01HZ4X7TGZM3J1A2B3C4D5E6FO",  # contains O
            "01HZ4X7TGZM3J1A2B3C4D5E6FU",  # contains U
        ],
    )
    def test_invalid_ulid_rejected(self, bad_id: str) -> None:
        with pytest.raises(ValidationError):
            DlmFrontmatter(dlm_id=bad_id, base_model="smollm2-135m")

    def test_unknown_top_level_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DlmFrontmatter.model_validate(
                {
                    "dlm_id": VALID_ULID,
                    "base_model": "smollm2-135m",
                    "surprise": "field",
                }
            )

    def test_dlm_version_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DlmFrontmatter(dlm_id=VALID_ULID, base_model="x", dlm_version=0)

    def test_frozen_rejects_mutation(self) -> None:
        fm = DlmFrontmatter(dlm_id=VALID_ULID, base_model="x")
        with pytest.raises(ValidationError):
            fm.base_model = "other"  # type: ignore[misc]
