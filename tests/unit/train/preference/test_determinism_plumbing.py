"""Verify DPO + ORPO phases call `seed_everything` with the seed.

Audit-08 N2: Sprint 17 claimed "DPO phase respects determinism
contract — seed+RNG plumbing mirrors SFT" but had no test for it.
This white-box test asserts that both phases invoke
`determinism.seed_everything(seed)` before any model instantiation,
so the claim is grounded rather than aspirational.

Byte-identical-adapter verification still belongs in a future slow
test — this unit test covers the plumbing contract.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from dlm.base_models import BASE_MODELS
from dlm.doc.parser import ParsedDlm
from dlm.doc.schema import DlmFrontmatter, PreferenceConfig, TrainingConfig
from dlm.doc.sections import Section, SectionType
from dlm.store.manifest import Manifest, save_manifest
from dlm.store.paths import for_dlm


def _parsed_with_preferences(method: str) -> ParsedDlm:
    pref_body = "### Prompt\nq?\n### Chosen\nc.\n### Rejected\nr.\n"
    return ParsedDlm(
        frontmatter=DlmFrontmatter(
            dlm_id="01KABCD" + "0" * 19,
            base_model="smollm2-135m",
            training=TrainingConfig(
                seed=1337,
                preference=PreferenceConfig(enabled=True, method=method),
            ),
        ),
        sections=(Section(type=SectionType.PREFERENCE, content=pref_body),),
    )


def _plan() -> SimpleNamespace:
    return SimpleNamespace(
        precision="bf16",
        attn_implementation="sdpa",
        use_qlora=False,
        quant_compute_dtype=None,
        micro_batch_size=1,
        grad_accum=1,
        effective_batch_size=1,
        gradient_checkpointing=False,
        est_peak_vram_gb=1.0,
        est_step_seconds=0.1,
        reason="test",
        to_dict=lambda: {"precision": "bf16"},
    )


def _mock_factory(**_: Any) -> MagicMock:
    dpo = MagicMock()
    dpo.state = SimpleNamespace(global_step=5, epoch=1.0, best_metric=None)
    dpo.optimizer = SimpleNamespace(state_dict=lambda: {"lr": 5e-6})
    dpo.lr_scheduler = SimpleNamespace(state_dict=lambda: {"step": 5})
    dpo.scaler = None
    dpo.control = SimpleNamespace(should_training_stop=False)
    dpo.train.return_value = SimpleNamespace(training_loss=0.5)

    def _save_model(path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_config.json").write_text("{}")
        (p / "adapter_model.safetensors").write_bytes(b"\x00" * 32)

    dpo.save_model.side_effect = _save_model
    return dpo


def _seed_store(tmp_path: Path, dlm_id: str) -> Any:
    store = for_dlm(dlm_id, home=tmp_path)
    store.ensure_layout()
    save_manifest(
        store.manifest,
        Manifest(dlm_id=dlm_id, base_model="smollm2-135m", adapter_version=1),
    )
    v0001 = store.adapter_version(1)
    v0001.mkdir(parents=True, exist_ok=True)
    (v0001 / "adapter_config.json").write_text("{}")
    return store


class TestDpoSeedsRngBeforeTraining:
    def test_explicit_seed_flows_through_to_seed_everything(
        self, tmp_path: Path
    ) -> None:
        from dlm.train.preference.dpo_phase import run

        store = _seed_store(tmp_path, "01KDPOSEED" + "0" * 16)
        spec = BASE_MODELS["smollm2-135m"]

        with patch(
            "dlm.train.preference.dpo_phase.seed_everything",
            wraps=__import__(
                "dlm.train.determinism", fromlist=["seed_everything"]
            ).seed_everything,
        ) as spy:
            run(
                store,
                _parsed_with_preferences("dpo"),
                spec,
                _plan(),
                reference_adapter_version=1,
                seed=9001,
                trainer_factory=_mock_factory,
            )
        spy.assert_called_once_with(9001)

    def test_default_seed_falls_back_to_frontmatter(self, tmp_path: Path) -> None:
        from dlm.train.preference.dpo_phase import run

        store = _seed_store(tmp_path, "01KDPOSEED" + "0" * 16)
        spec = BASE_MODELS["smollm2-135m"]

        with patch(
            "dlm.train.preference.dpo_phase.seed_everything",
            wraps=__import__(
                "dlm.train.determinism", fromlist=["seed_everything"]
            ).seed_everything,
        ) as spy:
            run(
                store,
                _parsed_with_preferences("dpo"),
                spec,
                _plan(),
                reference_adapter_version=1,
                trainer_factory=_mock_factory,
            )
        # Frontmatter seed was 1337.
        spy.assert_called_once_with(1337)


class TestOrpoSeedsRngBeforeTraining:
    def test_explicit_seed_flows_through(self, tmp_path: Path) -> None:
        from dlm.train.preference.orpo_phase import run

        store = _seed_store(tmp_path, "01KORPOSEED" + "0" * 15)
        spec = BASE_MODELS["smollm2-135m"]

        with patch(
            "dlm.train.preference.orpo_phase.seed_everything",
            wraps=__import__(
                "dlm.train.determinism", fromlist=["seed_everything"]
            ).seed_everything,
        ) as spy:
            run(
                store,
                _parsed_with_preferences("orpo"),
                spec,
                _plan(),
                reference_adapter_version=1,
                seed=42,
                trainer_factory=_mock_factory,
            )
        spy.assert_called_once_with(42)
