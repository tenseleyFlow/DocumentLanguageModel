"""End-to-end: `dlm train <dir>` → scaffold → train → resume.

Slow-marked because it runs the real SFTTrainer path. Verifies:

1. First invocation on a bare directory + `--base` + `--include`
   scaffolds `<dir>/.dlm/corpus.dlm` and produces an adapter.
2. The scaffolded `.dlm` has the expected ULID + base_model + sources.
3. Second invocation on the same directory (no flags) reuses the
   scaffolded anchor — same dlm_id, no new scaffold file, trains to
   a new adapter version.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.slow
def test_auto_scaffold_train_resume_cycle(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    offline_vars = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    saved_env = {k: os.environ.pop(k, None) for k in offline_vars}
    try:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"torch/transformers unavailable: {exc}")

        try:
            from tests.fixtures.tiny_model import tiny_model_path

            tiny_model_path()
        except Exception as exc:
            pytest.skip(f"tiny-model fixture unavailable: {exc}")

        from dlm.base_models import resolve as resolve_base_model
        from dlm.cli.scaffold import scaffold_train_target
        from dlm.doc.parser import parse_file
        from dlm.hardware import doctor
        from dlm.store.manifest import Manifest, load_manifest, save_manifest
        from dlm.store.paths import for_dlm
        from dlm.train import run as run_training

        plan = doctor().plan
        if plan is None:
            pytest.skip("doctor() returned no viable training plan on this host")

        home = tmp_path_factory.mktemp("dlm-scaffold-home")
        os.environ["DLM_HOME"] = str(home)

        # Build a minimal source tree: two .md files.
        corpus_dir = home / "demo-corpus"
        corpus_dir.mkdir()
        (corpus_dir / "note1.md").write_text(
            "# Note one\nContent about widgets.\n", encoding="utf-8"
        )
        (corpus_dir / "note2.md").write_text(
            "# Note two\nContent about gadgets.\n", encoding="utf-8"
        )

        # --- First invocation: scaffold + train ---------------------------
        result = scaffold_train_target(
            corpus_dir,
            base="smollm2-135m",
            include=("**/*.md",),
            exclude=(),
            recursive=True,
            name="corpus",
            policy="strict",
            rescaffold=False,
        )
        assert result.scaffolded is True
        assert result.dlm_path == corpus_dir / ".dlm" / "corpus.dlm"
        assert result.dlm_path.is_file()

        parsed = parse_file(result.dlm_path)
        assert parsed.frontmatter.dlm_id == result.dlm_id
        assert parsed.frontmatter.base_model == "smollm2-135m"
        assert parsed.frontmatter.training.sources is not None
        assert parsed.frontmatter.training.sources[0].include == ("**/*.md",)

        spec = resolve_base_model(parsed.frontmatter.base_model)
        store = for_dlm(parsed.frontmatter.dlm_id)
        store.ensure_layout()
        save_manifest(
            store.manifest,
            Manifest(
                dlm_id=parsed.frontmatter.dlm_id,
                base_model=parsed.frontmatter.base_model,
            ),
        )

        run1 = run_training(store, parsed, spec, plan, mode="fresh", seed=42, max_steps=6)
        assert run1.adapter_version == 1

        # --- Second invocation: reuse scaffolded .dlm ---------------------
        result2 = scaffold_train_target(
            corpus_dir,
            base=None,
            include=(),
            exclude=(),
            recursive=True,
            name="corpus",
            policy="strict",
            rescaffold=False,
        )
        assert result2.scaffolded is False
        assert result2.dlm_path == result.dlm_path
        assert result2.dlm_id == result.dlm_id

        # Train again — should produce adapter v0002 in the same store.
        parsed2 = parse_file(result2.dlm_path)
        run2 = run_training(store, parsed2, spec, plan, mode="fresh", seed=42, max_steps=6)
        assert run2.adapter_version == 2

        manifest = load_manifest(store.manifest)
        assert len(manifest.training_runs) == 2

    finally:
        for key, value in saved_env.items():
            if value is not None:
                os.environ[key] = value
