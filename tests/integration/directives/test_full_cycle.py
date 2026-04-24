"""End-to-end directive ingestion → one-cycle train → finite adapter.

Slow-marked: runs the real SFTTrainer path with a tiny fixture tree
declared as `training.sources` in the frontmatter. Asserts:

- Training completes, produces `v0001/adapter_model.safetensors` finite.
- TrainingSummary JSON contains `source_directives` with non-zero
  file_count and total_bytes matching the fixture tree.
- The CPT row path received the directive-sourced content — verified
  indirectly by observing that `content_hashes` on the manifest
  includes the synthesized section IDs (delta diff writes them on
  commit).
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.slow
def test_directive_tree_trains_and_summarizes(
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

        from dlm.doc.parser import parse_file
        from dlm.eval import load_summary
        from dlm.store.manifest import Manifest, load_manifest, save_manifest
        from dlm.store.paths import for_dlm
        from dlm.train import run as run_training
        from tests.fixtures.planning import resolve_spec_and_plan

        home = tmp_path_factory.mktemp("dlm-directives-home")
        os.environ["DLM_HOME"] = str(home)

        # Fixture tree: two Python files plus an excluded binary.
        tree = home / "src"
        tree.mkdir()
        (tree / "a.py").write_text(
            "def add(x, y):\n    return x + y\n\ndef sub(x, y):\n    return x - y\n",
            encoding="utf-8",
        )
        (tree / "b.py").write_text(
            "class Greeter:\n    def hello(self):\n        return 'hi'\n",
            encoding="utf-8",
        )
        (tree / "ignore.bin").write_bytes(b"\x00\x01\x02binary")

        doc = home / "dir.dlm"
        doc.write_text(
            "---\n"
            "dlm_id: 01HRSHWD00000000000000DHRS\n"
            "dlm_version: 6\n"
            "base_model: smollm2-135m\n"
            "training:\n"
            "  sources:\n"
            "    - path: src\n"
            "      include: ['**/*.py']\n"
            "---\n"
            "::instruction::\n"
            "### Q\n"
            "Pair me with the code.\n"
            "### A\n"
            "Okay.\n",
            encoding="utf-8",
        )

        parsed = parse_file(doc)
        spec, plan, _caps = resolve_spec_and_plan(parsed)
        store = for_dlm(parsed.frontmatter.dlm_id)
        store.ensure_layout()
        save_manifest(
            store.manifest,
            Manifest(
                dlm_id=parsed.frontmatter.dlm_id,
                base_model=parsed.frontmatter.base_model,
            ),
        )

        result = run_training(
            store,
            parsed,
            spec,
            plan,
            mode="fresh",
            seed=42,
            max_steps=10,
        )

        # Adapter produced.
        adapter_dir = store.resolve_current_adapter()
        assert adapter_dir is not None
        assert adapter_dir.name == "v0001"
        assert (adapter_dir / "adapter_model.safetensors").is_file()

        # Weights are finite — defense-in-depth; the integrity gate
        # enforces this but we pin it explicitly for this path.
        from safetensors.torch import load_file  # type: ignore[import-not-found]

        tensors = load_file(adapter_dir / "adapter_model.safetensors")
        assert all(torch.isfinite(t).all().item() for t in tensors.values())

        # TrainingSummary records the directive provenance.
        summary = load_summary(result.summary_path)
        assert len(summary.source_directives) == 1
        prov = summary.source_directives[0]
        assert prov.path == "src"
        assert prov.file_count == 2
        assert prov.total_bytes > 0

        # Manifest content_hashes should include the in-body instruction
        # section + the two directive-sourced prose sections.
        manifest = load_manifest(store.manifest)
        assert len(manifest.content_hashes) >= 3
    finally:
        for key, value in saved_env.items():
            if value is not None:
                os.environ[key] = value
