"""`policy: safety` refusal at extraction time."""

from __future__ import annotations

import pytest

from dlm.control import ControlPolicyRefusal, refuse_if_policy_safety


def test_no_tags_passes() -> None:
    refuse_if_policy_safety([{}, {}])


def test_non_safety_tags_pass() -> None:
    refuse_if_policy_safety(
        [
            {"lang": "python", "domain": "auth"},
            {"policy": "style"},
        ]
    )


def test_single_safety_tag_refuses() -> None:
    with pytest.raises(ControlPolicyRefusal, match="safety"):
        refuse_if_policy_safety([{"policy": "safety"}])


def test_mixed_corpus_refuses_if_any_safety() -> None:
    with pytest.raises(ControlPolicyRefusal):
        refuse_if_policy_safety(
            [
                {"lang": "python"},
                {"policy": "safety"},
                {"domain": "auth"},
            ]
        )


def test_policy_key_with_non_safety_value_passes() -> None:
    """`policy: style` or `policy: persona` — only 'safety' gates."""
    refuse_if_policy_safety(
        [
            {"policy": "style"},
            {"policy": "persona"},
            {"policy": "tone"},
        ]
    )


def test_empty_iterable_passes() -> None:
    refuse_if_policy_safety([])


def test_error_message_explains_why() -> None:
    with pytest.raises(ControlPolicyRefusal) as exc:
        refuse_if_policy_safety([{"policy": "safety"}])
    msg = str(exc.value)
    # The refusal message must explain the attack vector so the user
    # doesn't just retag to bypass and then hit the same footgun.
    assert "negative strength" in msg
