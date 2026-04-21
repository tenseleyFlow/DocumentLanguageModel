"""Multi-level `.dlm/training.yaml` + `.dlm/ignore` fixture end-to-end.

Builds a realistic tree with nested configs and exercises the full
`expand_sources` → discovery → merge pipeline via in-memory expansion
(no model training). The slow-marked train cycle lives in
`test_auto_scaffold_cycle.py`.

This test runs in the fast unit suite (no `@slow` marker) because it
only touches the filesystem + directive pipeline — no torch, no
transformers, no HF cache.
"""

from __future__ import annotations

from pathlib import Path

from dlm.directives import expand_sources
from dlm.doc.parser import parse_text

_VALID_ULID = "01HZ4X7TGZM3J1A2B3C4D5E6F7"


def _build_tree(root: Path) -> None:
    """Build a repo fixture:

        root/
          .dlm/
            training.yaml   include: ['src/**/*.py', 'docs/**/*.md']
                            exclude: ['**/test_*.py']
                            metadata: {language: python}
            ignore          *.log
          src/
            main.py
            test_main.py
            vendor/
              .dlm/
                training.yaml  exclude_defaults: false
                               metadata: {vendor: true_yes}
              .git_shim/       (bare dir w/ file to prove defaults off)
                HEAD
              dep.py
          docs/
            guide.md
            .dlm/
              ignore  !draft.md     (re-include what parent excluded? N/A)
            draft.md
          debug.log
          .env.local
          build/
            output.py
    """
    (root / ".dlm").mkdir()
    (root / ".dlm" / "training.yaml").write_text(
        "dlm_training_version: 1\n"
        "include: ['src/**/*.py', 'docs/**/*.md']\n"
        "exclude: ['**/test_*.py']\n"
        "metadata:\n  language: python\n",
        encoding="utf-8",
    )
    (root / ".dlm" / "ignore").write_text("*.log\n", encoding="utf-8")

    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("def main(): pass\n")
    (root / "src" / "test_main.py").write_text("def test_main(): pass\n")

    (root / "src" / "vendor").mkdir()
    (root / "src" / "vendor" / ".dlm").mkdir()
    (root / "src" / "vendor" / ".dlm" / "training.yaml").write_text(
        "dlm_training_version: 1\n"
        "exclude_defaults: false\n"
        "metadata:\n  vendor: true_yes\n",
        encoding="utf-8",
    )
    (root / "src" / "vendor" / "dep.py").write_text("def dep(): pass\n")

    (root / "docs").mkdir()
    (root / "docs" / "guide.md").write_text("# Guide\ncontent\n")
    (root / "docs" / "draft.md").write_text("# Draft\nwork in progress\n")

    (root / "debug.log").write_text("log line\n")
    (root / ".env.local").write_text("SECRET=shh\n")
    (root / "build").mkdir()
    (root / "build" / "output.py").write_text("print('built')\n")


def _make_parsed(root: Path) -> object:
    """Write a sibling .dlm pointing at root, parse it."""
    text = f"""---
dlm_id: {_VALID_ULID}
dlm_version: 6
base_model: smollm2-135m
training:
  sources_policy: permissive
  sources:
    - path: {root}
      include: ['**/*']
---

body
"""
    dlm_path = root.parent / "driver.dlm"
    dlm_path.write_text(text)
    return parse_text(text, path=dlm_path)


def test_full_descent_cycle(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _build_tree(root)
    parsed = _make_parsed(root)

    result = expand_sources(parsed, base_path=root.parent)  # type: ignore[arg-type]

    # Collect relpath → tags for every synthesized section
    ingested: dict[str, dict[str, str]] = {}
    for section in result.sections:
        # Content starts with "# source: <relpath>\n\n"
        first_line = section.content.splitlines()[0]
        assert first_line.startswith("# source: ")
        relpath = first_line.removeprefix("# source: ")
        ingested[relpath] = dict(section.tags)

    # --- What should be ingested ---------------------------------------
    # src/main.py: included (training.yaml include matches)
    assert "src/main.py" in ingested
    assert ingested["src/main.py"] == {"language": "python"}

    # src/vendor/dep.py: included (parent include **/*.py matches),
    # with vendor metadata layered on top of parent
    assert "src/vendor/dep.py" in ingested
    assert ingested["src/vendor/dep.py"] == {
        "language": "python",
        "vendor": "true_yes",
    }

    # docs/guide.md + docs/draft.md: included (docs/**/*.md matches)
    assert "docs/guide.md" in ingested
    assert "docs/draft.md" in ingested

    # --- What should NOT be ingested ----------------------------------
    # test_main.py: training.yaml exclude
    assert "src/test_main.py" not in ingested
    # debug.log: .dlm/ignore pattern
    assert "debug.log" not in ingested
    # .env.local: default-exclude set
    assert ".env.local" not in ingested
    # build/output.py: default-exclude set (build/**)
    assert "build/output.py" not in ingested

    # --- Provenance ----------------------------------------------------
    assert len(result.provenance) == 1
    prov = result.provenance[0]
    assert prov.file_count >= 4
    assert prov.skipped_by_descent >= 3  # at least log, env, build, test
