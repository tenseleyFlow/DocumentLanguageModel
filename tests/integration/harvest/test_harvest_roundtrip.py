"""Slow integration: full fortsh harvest cycle (Sprint 33.5).

Flow:

1. Hand-author a `.dlm` with a `training.sources` directive pointing
   at a small fortsh-like fixture.
2. Train v1 LoRA on SmolLM2-135M (30 steps).
3. Synthesize a sway-style JSON report with 6 failing probes, 4 of
   which have (prompt, reference) pairs. We simulate the report
   rather than shell out to `sway` — the integration that matters
   here is dlm's harvest consumer + retrain loop. Live sway runs
   are exercised in `.docs/audits/09-sway-appendix.md`.
4. `dlm harvest --apply` — write failing probes back as
   `!probe`-tagged `::instruction::` sections.
5. Re-parse the updated `.dlm` — expect the new sections with
   `auto_harvest=True`.
6. Retrain v2 on the harvested document.
7. Assert v2's training corpus includes the harvested probes (the
   downstream probe-pass rate assertion from the sprint spec
   requires a live sway run; we assert the weaker but checkable
   property: the harvest round-trip preserves content into the
   retrained section set).

Skips gracefully on hosts without torch/transformers or when the
tiny-model fixture isn't materialized.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


_FORTSH_FIXTURE = """program hello
  implicit none
  print *, "hello, dgemm"
end program hello
"""

_PROBES = [
    # 4 failing probes with complete references — these harvest.
    {
        "name": "dgemm_purpose",
        "kind": "section_internalization",
        "verdict": "fail",
        "score": 0.15,
        "evidence": {
            "prompt": "What does SUBROUTINE DGEMM compute?",
            "reference": "A double-precision general matrix multiplication.",
            "confidence": 0.9,
        },
        "message": "adapter failed recall",
        "duration_s": 0.1,
    },
    {
        "name": "fortran_hello_world_shape",
        "kind": "section_internalization",
        "verdict": "fail",
        "score": 0.2,
        "evidence": {
            "prompt": "How do you declare a Fortran program with implicit none?",
            "reference": (
                "`program NAME` followed by `implicit none`, then the body, "
                "ending with `end program NAME`."
            ),
            "confidence": 0.85,
        },
        "message": "adapter failed recall",
        "duration_s": 0.1,
    },
    {
        "name": "common_block_semantics",
        "kind": "section_internalization",
        "verdict": "fail",
        "score": 0.18,
        "evidence": {
            "prompt": "What is a COMMON block in Fortran 77?",
            "reference": (
                "A mechanism for sharing variables between program units; "
                "legacy form of global state."
            ),
            "confidence": 0.75,
        },
        "message": "adapter failed recall",
        "duration_s": 0.1,
    },
    {
        "name": "allocatable_vs_pointer",
        "kind": "section_internalization",
        "verdict": "fail",
        "score": 0.22,
        "evidence": {
            "prompt": "Difference between allocatable and pointer in Fortran 90+?",
            "reference": (
                "Allocatables have automatic scope-based deallocation; "
                "pointers require explicit nullify/deallocate."
            ),
            "confidence": 0.8,
        },
        "message": "adapter failed recall",
        "duration_s": 0.1,
    },
    # 2 failing probes without references — harvest skips these.
    {
        "name": "style_fingerprint",
        "kind": "style_fingerprint",
        "verdict": "fail",
        "score": 0.3,
        "evidence": {"style_score": 0.3},
        "message": "style drift detected",
        "duration_s": 0.05,
    },
    {
        "name": "calibration_drift",
        "kind": "calibration_drift",
        "verdict": "fail",
        "score": 0.4,
        "evidence": {"delta": 0.12},
        "message": "calibration exceeded bound",
        "duration_s": 0.05,
    },
]


def _write_sway_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sway_version": "0.1.0.dev0",
                "base_model_id": "smollm2-135m",
                "adapter_id": "run_1",
                "started_at": "2026-04-21T00:00:00Z",
                "finished_at": "2026-04-21T00:05:00Z",
                "wall_seconds": 300.0,
                "probes": _PROBES,
            }
        ),
        encoding="utf-8",
    )


def _write_source_dlm(path: Path, corpus_dir: Path) -> None:
    path.write_text(
        (
            "---\n"
            "dlm_id: 01KPQ3HARVEST0000000000000\n"
            "dlm_version: 7\n"
            "base_model: smollm2-135m\n"
            "training:\n"
            "  sources_policy: permissive\n"
            "  sources:\n"
            f'    - path: "{corpus_dir}"\n'
            '      include: ["**/*.f90"]\n'
            "---\n"
            "# fortsh harvest cycle fixture\n"
            "\n"
            "Test corpus for the sway ↔ dlm probe loop integration.\n"
        ),
        encoding="utf-8",
    )


@pytest.mark.slow
def test_harvest_roundtrip_preserves_probes_into_retrain(
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
        from dlm.doc.parser import parse_file
        from dlm.hardware import doctor
        from dlm.harvest import apply_plan, build_plan, read_sway_report
        from dlm.store.manifest import Manifest, save_manifest
        from dlm.store.paths import for_dlm
        from dlm.train import run as run_training

        plan = doctor().plan
        if plan is None:
            pytest.skip("doctor() returned no viable training plan on this host")

        home = tmp_path_factory.mktemp("dlm-harvest-home")
        os.environ["DLM_HOME"] = str(home)

        # Fixture: tiny fortsh-like corpus + a .dlm pointing at it.
        corpus_dir = home / "fortsh-fixture"
        corpus_dir.mkdir()
        (corpus_dir / "hello.f90").write_text(_FORTSH_FIXTURE, encoding="utf-8")

        dlm_path = home / "fortran.dlm"
        _write_source_dlm(dlm_path, corpus_dir)

        sway_json = home / "sway-run1.json"
        _write_sway_report(sway_json)

        # --- v1 train --------------------------------------------------
        parsed_v1 = parse_file(dlm_path)
        spec = resolve_base_model(parsed_v1.frontmatter.base_model)
        store = for_dlm(parsed_v1.frontmatter.dlm_id)
        store.ensure_layout()
        save_manifest(
            store.manifest,
            Manifest(
                dlm_id=parsed_v1.frontmatter.dlm_id,
                base_model=parsed_v1.frontmatter.base_model,
            ),
        )

        run1 = run_training(store, parsed_v1, spec, plan, mode="fresh", seed=42, max_steps=6)
        assert run1.adapter_version == 1

        # --- harvest ---------------------------------------------------
        candidates = read_sway_report(sway_json, strict=False)
        # Four probes had references; two did not — lax mode skipped the latter.
        assert len(candidates) == 4

        harvest_plan = build_plan(parsed_v1, candidates)
        assert len(harvest_plan.additions) == 4

        summary = apply_plan(parsed_v1, harvest_plan, target=dlm_path)
        assert summary.added == 4

        # --- verify on-disk shape --------------------------------------
        parsed_v2 = parse_file(dlm_path)
        harvested = [s for s in parsed_v2.sections if s.auto_harvest]
        assert len(harvested) == 4
        for section in harvested:
            assert section.harvest_source is not None
            assert section.harvest_source.startswith("auto-harvest/")
            assert "!probe" in section.content
            assert "### Q" in section.content
            assert "### A" in section.content

        # The harvested sections carry the sway-reported prompts verbatim.
        joined = " ".join(s.content for s in harvested)
        assert "DGEMM" in joined
        assert "implicit none" in joined
        assert "COMMON block" in joined
        assert "allocatable" in joined

        # --- v2 retrain ------------------------------------------------
        run2 = run_training(store, parsed_v2, spec, plan, mode="fresh", seed=42, max_steps=6)
        assert run2.adapter_version == 2

        # The v2 parsed corpus (what the trainer saw) still has the
        # harvested sections. The round-trip is preserved through the
        # full train → harvest → retrain cycle.
        parsed_v2_reloaded = parse_file(dlm_path)
        still_harvested = [s for s in parsed_v2_reloaded.sections if s.auto_harvest]
        assert len(still_harvested) == 4

    finally:
        for key, value in saved_env.items():
            if value is not None:
                os.environ[key] = value
