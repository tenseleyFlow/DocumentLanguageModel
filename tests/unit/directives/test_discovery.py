"""`.dlm/` walker — finds both training.yaml and ignore, tolerates malformed."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from dlm.directives.discovery import discover_configs


def test_missing_root_yields_empty(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    assert discover_configs(missing) == ()


def test_file_root_yields_empty(tmp_path: Path) -> None:
    f = tmp_path / "a_file.txt"
    f.write_text("x")
    assert discover_configs(f) == ()


def test_no_dlm_dirs_yields_empty(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x")
    assert discover_configs(tmp_path) == ()


def test_non_directory_dot_dlm_is_ignored(tmp_path: Path) -> None:
    (tmp_path / ".dlm").write_text("not a directory", encoding="utf-8")
    assert discover_configs(tmp_path) == ()


def test_single_dlm_at_root_with_both_files(tmp_path: Path) -> None:
    (tmp_path / ".dlm").mkdir()
    (tmp_path / ".dlm" / "training.yaml").write_text(
        "dlm_training_version: 1\ninclude: ['src/**/*.py']\n"
    )
    (tmp_path / ".dlm" / "ignore").write_text("*.log\n")
    configs = discover_configs(tmp_path)
    assert len(configs) == 1
    (c,) = configs
    assert c.anchor == tmp_path
    assert c.config is not None
    assert c.config.include == ("src/**/*.py",)
    assert len(c.ignore_rules) == 1


def test_bare_dlm_dir_produces_record(tmp_path: Path) -> None:
    """A `.dlm/` with no files inside still produces a record — a
    future sprint (e.g. the auto-scaffold) may drop files there."""
    (tmp_path / ".dlm").mkdir()
    configs = discover_configs(tmp_path)
    assert len(configs) == 1
    assert configs[0].config is None
    assert configs[0].ignore_rules == ()


def test_sorted_by_depth(tmp_path: Path) -> None:
    # Create nested `.dlm/` at three levels
    (tmp_path / ".dlm").mkdir()
    (tmp_path / "a" / ".dlm").mkdir(parents=True)
    (tmp_path / "a" / "b" / ".dlm").mkdir(parents=True)
    configs = discover_configs(tmp_path)
    anchors = [c.anchor for c in configs]
    # Parents appear before descendants
    assert anchors == [tmp_path, tmp_path / "a", tmp_path / "a" / "b"]


def test_malformed_yaml_logs_and_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / ".dlm").mkdir()
    (tmp_path / ".dlm" / "training.yaml").write_text("dlm_training_version: [not a scalar\n")
    caplog.set_level(logging.WARNING, logger="dlm.directives.discovery")
    configs = discover_configs(tmp_path)
    assert len(configs) == 1
    assert configs[0].config is None
    assert any("invalid YAML" in rec.message for rec in caplog.records)


def test_invalid_utf8_training_yaml_logs_and_continues(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    (tmp_path / ".dlm").mkdir()
    (tmp_path / ".dlm" / "training.yaml").write_bytes(b"caf\xe9\n")
    caplog.set_level(logging.WARNING, logger="dlm.directives.discovery")
    configs = discover_configs(tmp_path)
    assert configs[0].config is None
    assert any("not UTF-8" in rec.message for rec in caplog.records)


def test_schema_violation_logs_and_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / ".dlm").mkdir()
    (tmp_path / ".dlm" / "training.yaml").write_text("dlm_training_version: 1\nunknown_key: bad\n")
    caplog.set_level(logging.WARNING, logger="dlm.directives.discovery")
    configs = discover_configs(tmp_path)
    assert configs[0].config is None
    assert any("schema violation" in rec.message for rec in caplog.records)


def test_training_yaml_non_mapping_top_level(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / ".dlm").mkdir()
    (tmp_path / ".dlm" / "training.yaml").write_text("- just a list\n- of items\n")
    caplog.set_level(logging.WARNING, logger="dlm.directives.discovery")
    configs = discover_configs(tmp_path)
    assert configs[0].config is None
    assert any("must be a mapping" in rec.message for rec in caplog.records)


def test_training_yaml_null_top_level_coerces_to_empty_config(tmp_path: Path) -> None:
    (tmp_path / ".dlm").mkdir()
    (tmp_path / ".dlm" / "training.yaml").write_text("null\n", encoding="utf-8")
    configs = discover_configs(tmp_path)
    assert configs[0].config is not None
    assert configs[0].config.dlm_training_version == 1


def test_both_files_coexist(tmp_path: Path) -> None:
    (tmp_path / ".dlm").mkdir()
    (tmp_path / ".dlm" / "training.yaml").write_text("dlm_training_version: 1\nexclude: ['a']\n")
    (tmp_path / ".dlm" / "ignore").write_text("*.tmp\n")
    (c,) = discover_configs(tmp_path)
    assert c.config is not None
    assert c.config.exclude == ("a",)
    assert len(c.ignore_rules) == 1


def test_invalid_utf8_ignore_logs_and_continues(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    (tmp_path / ".dlm").mkdir()
    (tmp_path / ".dlm" / "ignore").write_bytes(b"bad-\xff\n")
    caplog.set_level(logging.WARNING, logger="dlm.directives.discovery")
    configs = discover_configs(tmp_path)
    assert configs[0].ignore_rules == ()
    assert any("not UTF-8" in rec.message for rec in caplog.records)
