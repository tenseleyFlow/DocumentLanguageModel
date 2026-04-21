"""Preference-method registry dispatcher."""

from __future__ import annotations

import pytest

from dlm.train.preference.method_registry import (
    METHODS,
    UnknownMethodError,
    register,
    resolve,
)


class TestBuiltins:
    def test_dpo_registered(self) -> None:
        runner = resolve("dpo")
        assert callable(runner)

    def test_orpo_registered(self) -> None:
        runner = resolve("orpo")
        assert callable(runner)

    def test_methods_set_matches_registry(self) -> None:
        # METHODS is the advertised set; every entry must be resolvable.
        for name in METHODS:
            assert callable(resolve(name))


class TestUnknownMethod:
    def test_raises_unknown_method_error(self) -> None:
        with pytest.raises(UnknownMethodError) as exc_info:
            resolve("kto")
        assert exc_info.value.name == "kto"

    def test_error_message_lists_known_methods(self) -> None:
        with pytest.raises(UnknownMethodError) as exc_info:
            resolve("simpo")
        msg = str(exc_info.value)
        assert "simpo" in msg
        # The known-list mention is part of the contract.
        assert "dpo" in msg


class TestRegisterCanReplace:
    def test_register_overrides_existing(self) -> None:
        saved = resolve("dpo")
        try:

            def _stub(*args: object, **kwargs: object) -> str:  # type: ignore[return-value]
                return "stub"

            register("dpo", _stub)
            assert resolve("dpo") is _stub
        finally:
            register("dpo", saved)
