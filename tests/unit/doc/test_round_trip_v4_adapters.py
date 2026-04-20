"""Round-trip regression: v4 doc with `training.adapters` parses and
re-serializes byte-identically (audit-07 N1)."""

from __future__ import annotations

from textwrap import dedent

from dlm.doc.parser import parse_text
from dlm.doc.serializer import serialize


def test_round_trip_v4_multi_adapter_doc_is_idempotent() -> None:
    """A v4 `.dlm` with `training.adapters` + routed sections must
    `parse → serialize → parse → serialize` identically."""
    original = dedent(
        """\
        ---
        dlm_id: 01HZ4X7TGZM3J1A2B3C4D5E6F7
        dlm_version: 4
        base_model: smollm2-135m
        training:
          adapters:
            knowledge:
              adapter: lora
              lora_r: 8
            tone:
              adapter: lora
              lora_r: 4
              target_modules:
              - q_proj
              - v_proj
        ---

        # Shared prose

        Shared prose goes to every adapter.

        ::instruction#knowledge::
        ### Q
        Capital of France?
        ### A
        Paris.

        ::instruction#tone::
        ### Q
        Tone?
        ### A
        Crisp.
        """
    )

    once = serialize(parse_text(original))
    twice = serialize(parse_text(once))
    assert once == twice, (
        "v4 adapters doc not idempotent under serialize round-trip"
    )


def test_round_trip_preserves_fence_suffixes() -> None:
    """`::instruction#name::` suffixes must survive the round-trip."""
    original = dedent(
        """\
        ---
        dlm_id: 01HZ4X7TGZM3J1A2B3C4D5E6F7
        base_model: smollm2-135m
        ---

        ::instruction#tone::
        ### Q
        q
        ### A
        a
        """
    )
    out = serialize(parse_text(original))
    assert "::instruction#tone::" in out
