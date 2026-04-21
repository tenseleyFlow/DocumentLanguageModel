"""`dlm cache prune` reads frontmatter default when `--older-than` is absent.

Per-document `training.cache.prune_older_than_days` takes effect when
`--older-than` is omitted. Pre-v9 docs inherit the 90-day default via
the Pydantic factory on re-parse, so the CLI behavior matches the old
hard-coded default without any config churn.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlm.cli.app import app


def _write_doc(path: Path, *, prune_days: int | None = None) -> None:
    cache_block = ""
    if prune_days is not None:
        cache_block = f"training:\n  cache:\n    prune_older_than_days: {prune_days}\n"
    path.write_text(
        "---\n"
        "dlm_id: 01KPQ9PRNE" + "0" * 16 + "\n"
        "base_model: smollm2-135m\n" + cache_block + "---\n"
        "body\n",
        encoding="utf-8",
    )


def test_prune_without_flag_uses_frontmatter_value(tmp_path: Path) -> None:
    """When the doc declares `prune_older_than_days: 7`, `dlm cache prune`
    without `--older-than` honors that."""
    doc = tmp_path / "doc.dlm"
    _write_doc(doc, prune_days=7)

    runner = CliRunner()
    result = runner.invoke(app, ["--home", str(tmp_path / "home"), "cache", "prune", str(doc)])
    assert result.exit_code == 0, result.output
    # The confirmation message echoes the cutoff label we derived.
    assert "older than 7d" in result.output


def test_prune_without_flag_defaults_to_90d_on_pre_v9_doc(tmp_path: Path) -> None:
    """A doc without a `training.cache` block still gets the default
    90-day window via the Pydantic factory."""
    doc = tmp_path / "doc.dlm"
    _write_doc(doc, prune_days=None)

    runner = CliRunner()
    result = runner.invoke(app, ["--home", str(tmp_path / "home"), "cache", "prune", str(doc)])
    assert result.exit_code == 0, result.output
    assert "older than 90d" in result.output


def test_explicit_flag_overrides_frontmatter(tmp_path: Path) -> None:
    """CLI flag wins over the frontmatter default."""
    doc = tmp_path / "doc.dlm"
    _write_doc(doc, prune_days=7)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--home", str(tmp_path / "home"), "cache", "prune", str(doc), "--older-than", "30d"],
    )
    assert result.exit_code == 0, result.output
    # CLI flag label, not frontmatter's 7d.
    assert "older than 30d" in result.output
    assert "older than 7d" not in result.output
