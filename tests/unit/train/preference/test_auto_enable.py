"""Auto-enable unit tests — the boundary between "user chose" and
"content inferred"."""

from __future__ import annotations

from dlm.doc.schema import PreferenceConfig
from dlm.doc.sections import Section, SectionType
from dlm.train.preference.auto_enable import resolve_preference_enabled


def _pref() -> Section:
    return Section(
        type=SectionType.PREFERENCE,
        content="### Prompt\nq\n### Chosen\nc\n### Rejected\nr\n",
        start_line=1,
    )


def _prose() -> Section:
    return Section(type=SectionType.PROSE, content="text", start_line=1)


class TestUserExplicit:
    def test_user_enabled_true_preserved(self) -> None:
        cfg = PreferenceConfig(enabled=True)
        out = resolve_preference_enabled(cfg, [_pref()])
        assert out.enabled is True
        assert out is cfg  # no copy when no change

    def test_user_enabled_false_preserved_even_with_prefs(self) -> None:
        cfg = PreferenceConfig(enabled=False)
        out = resolve_preference_enabled(cfg, [_pref()])
        assert out.enabled is False
        assert out is cfg

    def test_user_enabled_false_preserved_without_prefs(self) -> None:
        cfg = PreferenceConfig(enabled=False)
        out = resolve_preference_enabled(cfg, [_prose()])
        assert out.enabled is False


class TestAutoEnableFires:
    def test_unset_plus_preferences_flips_on(self) -> None:
        cfg = PreferenceConfig()  # no explicit enabled
        out = resolve_preference_enabled(cfg, [_prose(), _pref()])
        assert out.enabled is True
        # non-enabled fields preserved
        assert out.hyperparams == cfg.hyperparams
        assert out.loss_type == cfg.loss_type
        assert out.method == cfg.method


class TestAutoEnableNoOp:
    def test_unset_plus_no_preferences_stays_off(self) -> None:
        cfg = PreferenceConfig()
        out = resolve_preference_enabled(cfg, [_prose()])
        assert out.enabled is False
        assert out is cfg

    def test_empty_sections_stays_off(self) -> None:
        cfg = PreferenceConfig()
        out = resolve_preference_enabled(cfg, [])
        assert out.enabled is False


class TestImmutability:
    def test_result_is_not_original_instance_when_changed(self) -> None:
        cfg = PreferenceConfig()
        out = resolve_preference_enabled(cfg, [_pref()])
        assert out is not cfg
        # original stayed default
        assert cfg.enabled is False
