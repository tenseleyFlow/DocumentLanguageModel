"""End-to-end: build_weighted_merged + save_merged_to_tmp on tiny model.

Audit-07 M5: the weighted-merge heavy path was entirely `# pragma: no
cover` with no integration coverage, which is how B2 (missing tokenizer
files in the merged dir) landed undetected. This test drives the real
PEFT `add_weighted_adapter` call on two freshly-trained tiny adapters
and asserts the saved tmp dir has the files `run_export`'s preflight
requires.

The full `dlm export --adapter-mix` path additionally needs llama.cpp
vendored; that's gated via `resolve_llama_cpp_paths()` and this test
skips gracefully when the submodule isn't built.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.slow


_PROSE = """# Domain

Shared across both adapters.

::instruction#knowledge::
### Q
Capital of France?
### A
Paris.

::instruction#tone::
### Q
Tone guidance?
### A
Crisp.
"""


def _train_two_adapters(
    tmp_path_factory: pytest.TempPathFactory,
):  # type: ignore[no-untyped-def]
    """Train knowledge + tone adapters; return (parsed, store)."""
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
    from dlm.store.manifest import Manifest, save_manifest
    from dlm.store.paths import for_dlm
    from dlm.train.multi_adapter.trainer import run_all
    from tests.fixtures.dlm_factory import make_dlm, prose
    from tests.fixtures.planning import resolve_spec_and_plan

    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(key, None)

    home = tmp_path_factory.mktemp("dlm-wm-home")
    os.environ["DLM_HOME"] = str(home)

    doc = home / "merge.dlm"
    doc.write_text(
        make_dlm(
            sections=[prose(_PROSE)],
            base_model="smollm2-135m",
            training_overrides={"adapters": {"knowledge": {}, "tone": {}}},
        ),
        encoding="utf-8",
    )
    raw = doc.read_text(encoding="utf-8")
    fm_end = raw.find("\n---\n", raw.find("---") + 3) + len("\n---\n")
    doc.write_text(raw[:fm_end] + "\n" + _PROSE, encoding="utf-8")

    parsed = parse_file(doc)
    spec, plan, _caps = resolve_spec_and_plan(parsed, accept_license=True)
    store = for_dlm(parsed.frontmatter.dlm_id)
    store.ensure_layout()
    save_manifest(
        store.manifest,
        Manifest(
            dlm_id=parsed.frontmatter.dlm_id,
            base_model=parsed.frontmatter.base_model,
        ),
    )
    run_all(
        store,
        parsed,
        spec,
        plan,
        mode="fresh",
        seed=42,
        max_steps=3,
        lock_mode="ignore",
    )
    return parsed, store


@pytest.mark.slow
def test_weighted_merge_saves_tokenizer_files(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:  # type: ignore[no-untyped-def]
    """Audit-07 B2 regression: save_merged_to_tmp must copy tokenizer
    files so run_export's preflight (check_tokenizer_vocab) succeeds."""
    from transformers import AutoModelForCausalLM

    from dlm.base_models import download_spec
    from dlm.base_models import resolve as resolve_base_model
    from dlm.export.weighted_merge import (
        MixEntry,
        build_weighted_merged,
        resolve_first_source_path,
        save_merged_to_tmp,
    )

    parsed, store = _train_two_adapters(tmp_path_factory)

    spec = resolve_base_model(parsed.frontmatter.base_model, accept_license=True)
    cached = download_spec(spec, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(str(cached.path), revision=spec.revision)

    entries = [
        MixEntry(name="knowledge", weight=1.0),
        MixEntry(name="tone", weight=0.5),
    ]
    merged = build_weighted_merged(base_model, store, spec, entries)

    merge_dir = store.cache_dir_for("_merge_test")
    first_source = resolve_first_source_path(store, entries)
    out_dir = save_merged_to_tmp(
        merged,
        merge_dir,
        tokenizer_source=first_source,
        training_run_source=first_source,
    )

    # B2's core assertion: the merged dir has the PEFT artifacts
    # AND the tokenizer files run_export preflight demands.
    assert (out_dir / "adapter_config.json").is_file()
    assert (out_dir / "adapter_model.safetensors").is_file()
    assert (out_dir / "tokenizer_config.json").is_file()
    # Either tokenizer.json (fast) or vocab.json + merges.txt (slow)
    # is acceptable — HF tokenizers write different combinations.
    has_tokenizer = (
        (out_dir / "tokenizer.json").is_file()
        or ((out_dir / "vocab.json").is_file() and (out_dir / "merges.txt").is_file())
        or (out_dir / "tokenizer.model").is_file()
    )
    assert has_tokenizer, (
        f"no tokenizer vocabulary files in {out_dir}; "
        f"contents={sorted(p.name for p in out_dir.iterdir())}"
    )


@pytest.mark.slow
def test_weighted_merge_passes_preflight_tokenizer_vocab(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:  # type: ignore[no-untyped-def]
    """Drive the merged dir through `run_export`'s tokenizer preflight
    probe to confirm the path is end-to-end clean (no full export —
    that would need llama.cpp vendored)."""
    from transformers import AutoModelForCausalLM

    from dlm.base_models import download_spec
    from dlm.base_models import resolve as resolve_base_model
    from dlm.export.preflight import check_tokenizer_vocab
    from dlm.export.weighted_merge import (
        MixEntry,
        build_weighted_merged,
        resolve_first_source_path,
        save_merged_to_tmp,
    )

    parsed, store = _train_two_adapters(tmp_path_factory)

    spec = resolve_base_model(parsed.frontmatter.base_model, accept_license=True)
    cached = download_spec(spec, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(str(cached.path), revision=spec.revision)

    entries = [
        MixEntry(name="knowledge", weight=0.7),
        MixEntry(name="tone", weight=0.3),
    ]
    merged = build_weighted_merged(base_model, store, spec, entries)
    merge_dir = store.cache_dir_for("_merge_preflight_test")
    first_source = resolve_first_source_path(store, entries)
    out_dir = save_merged_to_tmp(
        merged,
        merge_dir,
        tokenizer_source=first_source,
        training_run_source=first_source,
    )

    # The preflight probe that B2 broke — now passes.
    check_tokenizer_vocab(out_dir)
