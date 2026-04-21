"""v8 → v9 identity migrator — the schema bump for `training.cache`."""

from __future__ import annotations

from dlm.doc.parser import parse_text
from dlm.doc.schema import CURRENT_SCHEMA_VERSION


def test_current_schema_is_v9() -> None:
    assert CURRENT_SCHEMA_VERSION == 9


def test_v8_document_parses_under_v9_with_cache_defaults() -> None:
    body = (
        "---\n"
        "dlm_id: 01KPQ9CACHETEST00000000000\n"
        "dlm_version: 8\n"
        "base_model: smollm2-135m\n"
        "---\n"
        "::instruction::\n### Q\nhi?\n### A\nhello.\n"
    )
    parsed = parse_text(body)
    # Migrator upgrades the version; cache picks up factory defaults.
    assert parsed.frontmatter.dlm_version == CURRENT_SCHEMA_VERSION
    cache = parsed.frontmatter.training.cache
    assert cache.enabled is True
    assert cache.max_bytes == 10 * 1024 * 1024 * 1024
    assert cache.prune_older_than_days == 90


def test_v9_document_with_explicit_cache_block_parses() -> None:
    body = (
        "---\n"
        "dlm_id: 01KPQ9CACHETEST00000000000\n"
        "dlm_version: 9\n"
        "base_model: smollm2-135m\n"
        "training:\n"
        "  cache:\n"
        "    enabled: false\n"
        "    max_bytes: 2147483648\n"
        "    prune_older_than_days: 30\n"
        "---\n"
        "::instruction::\n### Q\nhi?\n### A\nhello.\n"
    )
    parsed = parse_text(body)
    cache = parsed.frontmatter.training.cache
    assert cache.enabled is False
    assert cache.max_bytes == 2 * 1024**3
    assert cache.prune_older_than_days == 30
