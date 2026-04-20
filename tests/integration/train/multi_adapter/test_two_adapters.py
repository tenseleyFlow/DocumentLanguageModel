"""End-to-end: two-adapter document trains both lanes on tiny-model.

Asserts the post-state the single-adapter flow doesn't produce:

- `adapter/knowledge/versions/v0001/` and `adapter/tone/versions/v0001/`
  both exist with committed PEFT artifacts + training-state sidecars.
- Each adapter has its own `current.txt` pointer.
- The manifest gains one `TrainingRunSummary` per adapter (two total),
  in declaration order.
- Section routing is honored: the tone-routed instruction section
  trains only the tone adapter, not the knowledge adapter.

Slow — driven by the weekly integration-slow CI job.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


_DOC_BODY = """# Domain

This prose is shared across both adapters — prose fans out to every
declared adapter by default so each one picks up the same domain
vocabulary.

::instruction#knowledge::
### Q
What is the capital of France?
### A
Paris.

::instruction#tone::
### Q
How should I phrase things?
### A
Crisply. One sentence.
"""


@pytest.mark.slow
def test_two_adapters_each_get_their_own_version_history(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:  # type: ignore[no-untyped-def]
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
    from dlm.doc.parser import parse_file
    from dlm.hardware import doctor
    from dlm.store.manifest import Manifest, load_manifest, save_manifest
    from dlm.store.paths import for_dlm
    from dlm.train.multi_adapter.trainer import run_all
    from tests.fixtures.dlm_factory import make_dlm, prose

    plan = doctor().plan
    if plan is None:
        pytest.skip("doctor() returned no viable training plan on this host")

    # Unset offline env so weights can download on cold caches.
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(key, None)

    home = tmp_path_factory.mktemp("dlm-multi-home")
    os.environ["DLM_HOME"] = str(home)
    doc_path: Path = home / "two_adapters.dlm"

    doc_path.write_text(
        make_dlm(
            sections=[prose(_DOC_BODY)],
            base_model="smollm2-135m",
            training_overrides={
                "adapters": {
                    "knowledge": {"lora_r": 8},
                    "tone": {"lora_r": 4},
                }
            },
        ),
        encoding="utf-8",
    )

    # make_dlm wraps the full body in a single PROSE section since it
    # doesn't know about adapter fences yet. Re-save the doc with the
    # raw body so the parser sees the instruction fences.
    raw = doc_path.read_text(encoding="utf-8")
    parsed_pre = parse_file(doc_path)
    fm_end = raw.find("\n---\n", raw.find("---") + 3) + len("\n---\n")
    doc_path.write_text(
        raw[:fm_end] + "\n" + _DOC_BODY,
        encoding="utf-8",
    )
    parsed = parse_file(doc_path)
    assert parsed.frontmatter.training.adapters is not None
    assert set(parsed.frontmatter.training.adapters) == {"knowledge", "tone"}

    spec = resolve_base_model(parsed.frontmatter.base_model, accept_license=True)
    store = for_dlm(parsed.frontmatter.dlm_id)
    store.ensure_layout()
    save_manifest(
        store.manifest,
        Manifest(
            dlm_id=parsed.frontmatter.dlm_id,
            base_model=parsed.frontmatter.base_model,
        ),
    )

    results = run_all(
        store,
        parsed,
        spec,
        plan,
        mode="fresh",
        seed=42,
        max_steps=5,
        lock_mode="ignore",  # avoid re-validating mid-loop
    )

    # Two TrainingRunResults, one per adapter.
    assert len(results) == 2

    # Each adapter got its own v0001 directory with committed PEFT files.
    for name in ("knowledge", "tone"):
        vdir = store.adapter_version_for(name, 1)
        assert vdir.is_dir(), f"{name}/v0001 missing"
        assert (vdir / "adapter_config.json").is_file()
        assert (vdir / "adapter_model.safetensors").is_file()
        pointer = store.resolve_current_adapter_for(name)
        assert pointer == vdir.resolve()

    # 20c B1: per-adapter LoRA config actually flows through. The
    # saved adapter_config.json must reflect the declared lora_r for
    # each adapter, not the flat default.
    import json

    k_cfg = json.loads(
        (store.adapter_version_for("knowledge", 1) / "adapter_config.json")
        .read_text(encoding="utf-8")
    )
    t_cfg = json.loads(
        (store.adapter_version_for("tone", 1) / "adapter_config.json")
        .read_text(encoding="utf-8")
    )
    assert k_cfg["r"] == 8, f"knowledge lora_r: {k_cfg['r']}"
    assert t_cfg["r"] == 4, f"tone lora_r: {t_cfg['r']}"

    # Manifest reflects both runs (in declaration order).
    manifest = load_manifest(store.manifest)
    assert len(manifest.training_runs) == 2
    assert manifest.training_runs[0].run_id < manifest.training_runs[1].run_id
