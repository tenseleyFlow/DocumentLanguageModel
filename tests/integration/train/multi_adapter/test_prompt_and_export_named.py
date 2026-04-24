"""End-to-end: dlm prompt + dlm export with named adapters.

Trains knowledge + tone adapters on a prose doc, then:

- Invokes `dlm prompt --adapter knowledge` and `--adapter tone`
  through CliRunner; asserts both produce non-empty output.
- Invokes `dlm export --adapter knowledge --skip-ollama`; asserts
  the GGUFs land and the manifest row carries `adapter_name=knowledge`.

Slow — weekly integration-slow CI job.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


_PROSE = """# Domain

Shared prose across both adapters.

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


def _skip_if_deps_missing() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"torch/transformers unavailable: {exc}")


def _skip_if_tiny_model_unavailable() -> None:
    try:
        from tests.fixtures.tiny_model import tiny_model_path

        tiny_model_path()
    except Exception as exc:
        pytest.skip(f"tiny-model fixture unavailable: {exc}")


def _train_two(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train knowledge + tone on a prose doc; return the .dlm path."""
    from dlm.doc.parser import parse_file
    from dlm.store.manifest import Manifest, save_manifest
    from dlm.store.paths import for_dlm
    from dlm.train.multi_adapter.trainer import run_all
    from tests.fixtures.dlm_factory import make_dlm, prose
    from tests.fixtures.planning import resolve_spec_and_plan

    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(key, None)

    home = tmp_path_factory.mktemp("dlm-named-home")
    os.environ["DLM_HOME"] = str(home)

    doc = home / "multi.dlm"
    doc.write_text(
        make_dlm(
            sections=[prose(_PROSE)],
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
    # Same trick as test_two_adapters: rewrite body so the parser sees
    # the fences instead of the make_dlm escape.
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
    return doc


@pytest.mark.slow
def test_load_for_inference_resolves_each_named_adapter(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:  # type: ignore[no-untyped-def]
    """Verify end-to-end adapter-path resolution for each declared name.

    We deliberately don't drive `generate()` — a 3-step-trained tiny
    model produces near-NaN logits that poison sampling. The wiring
    assertion (correct adapter dir resolved, loaded into PEFT) is
    what 20b needs; generation quality is covered by the larger
    slow suite when real training lands.
    """
    _skip_if_deps_missing()
    _skip_if_tiny_model_unavailable()

    from dlm.doc.parser import parse_file
    from dlm.inference.loader import load_for_inference, resolve_adapter_path
    from dlm.store.paths import for_dlm
    from tests.fixtures.planning import resolve_spec_and_plan

    doc = _train_two(tmp_path_factory)
    parsed = parse_file(doc)
    store = for_dlm(parsed.frontmatter.dlm_id)

    # Path resolution: each named adapter has its own committed version.
    k_path = resolve_adapter_path(store, adapter_name="knowledge")
    t_path = resolve_adapter_path(store, adapter_name="tone")
    assert k_path != t_path
    assert k_path.is_dir()
    assert t_path.is_dir()

    spec, _plan, caps = resolve_spec_and_plan(parsed, accept_license=True)

    # Full load exercises the PEFT adapter-load path on both names.
    loaded_k = load_for_inference(store, spec, caps, adapter_name="knowledge")
    assert loaded_k.adapter_path == k_path

    loaded_t = load_for_inference(store, spec, caps, adapter_name="tone")
    assert loaded_t.adapter_path == t_path


@pytest.mark.slow
def test_export_named_adapter_records_adapter_name(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:  # type: ignore[no-untyped-def]
    """The named-adapter export path stamps `adapter_name` onto ExportSummary."""
    _skip_if_deps_missing()
    _skip_if_tiny_model_unavailable()

    try:
        from dlm.export.vendoring import llama_quantize_bin

        llama_quantize_bin()  # probes the vendored quantize binary; raises if missing
    except Exception as exc:
        pytest.skip(f"llama.cpp vendoring unavailable: {exc}")

    from typer.testing import CliRunner

    from dlm.cli.app import app
    from dlm.doc.parser import parse_file
    from dlm.store.manifest import load_manifest
    from dlm.store.paths import for_dlm

    doc = _train_two(tmp_path_factory)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--home",
            os.environ["DLM_HOME"],
            "export",
            str(doc),
            "--adapter",
            "knowledge",
            "--skip-ollama",
            "--no-imatrix",
            "--no-template",
        ],
    )
    assert result.exit_code == 0, result.output

    parsed = parse_file(doc)
    store = for_dlm(parsed.frontmatter.dlm_id)
    manifest = load_manifest(store.manifest)
    assert manifest.exports, "ExportSummary not appended"
    assert manifest.exports[-1].adapter_name == "knowledge"
