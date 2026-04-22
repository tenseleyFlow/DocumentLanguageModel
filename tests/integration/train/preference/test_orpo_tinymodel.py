"""End-to-end: ORPO phase on a tiny model with preference sections.

Parallel to `test_dpo_tinymodel` but routes the preference phase
through the ORPO method. Uses its own `orpo_trained_store` flow
(fresh SFT-trained store, then ORPO on top) so the DPO slow test's
post-state doesn't contaminate ORPO's v0002 assertions.

Slow — driven by the weekly integration-slow CI job.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.slow
def test_orpo_phase_writes_second_adapter_version(trained_store) -> None:  # type: ignore[no-untyped-def]
    from dlm.base_models import resolve as resolve_base_model
    from dlm.doc.parser import parse_file
    from dlm.doc.serializer import serialize
    from dlm.hardware import doctor
    from dlm.store.manifest import load_manifest
    from dlm.train.preference.phase_orchestrator import run_phases

    store = trained_store.store
    dlm_path = trained_store.doc

    terse_preferences = _five_terse_preference_triples()
    _append_preference_section(dlm_path, terse_preferences)

    # Flip frontmatter method to ORPO on the now-appended doc.
    parsed = parse_file(dlm_path)
    new_pref = parsed.frontmatter.training.preference.model_copy(
        update={"method": "orpo", "enabled": True}
    )
    new_training = parsed.frontmatter.training.model_copy(update={"preference": new_pref})
    new_fm = parsed.frontmatter.model_copy(update={"training": new_training})
    from dlm.doc.parser import ParsedDlm

    rewritten = ParsedDlm(frontmatter=new_fm, sections=parsed.sections)
    dlm_path.write_text(serialize(rewritten), encoding="utf-8")

    parsed = parse_file(dlm_path)
    assert parsed.frontmatter.training.preference.method == "orpo"

    spec = resolve_base_model(parsed.frontmatter.base_model, accept_license=True)
    plan = doctor().plan
    if plan is None:
        pytest.skip("no viable plan on this host — ORPO body needs a real trainer")

    prior_manifest = load_manifest(store.manifest)
    prior_runs = len(prior_manifest.training_runs)
    prior_version = prior_manifest.adapter_version

    # Skip lock validation — we're reusing the DPO fixture's store
    # and the frontmatter change invalidates the recorded dlm_sha256.
    results = run_phases(
        store,
        parsed,
        spec,
        plan,
        phase="preference",
        capabilities=doctor().capabilities,
        lock_mode="ignore",
    )
    assert [r.phase for r in results] == ["preference"]
    orpo_result = results[0].result
    assert orpo_result.adapter_version == prior_version + 1
    assert store.adapter_version(orpo_result.adapter_version).is_dir()

    post = load_manifest(store.manifest)
    assert post.adapter_version == prior_version + 1
    assert len(post.training_runs) == prior_runs + 1


def _five_terse_preference_triples() -> str:
    pairs = [
        ("What is 2 + 2?", "4.", "The sum of two and two is four, a basic arithmetic fact."),
        (
            "What color is grass?",
            "Green.",
            "Grass is typically a vibrant shade of green most of the year.",
        ),
        ("Is water wet?", "Yes.", "Water is generally considered wet in most everyday contexts."),
        (
            "Do birds fly?",
            "Most do.",
            "The majority of bird species can indeed fly, though a few cannot.",
        ),
        ("What's 10 - 3?", "7.", "Ten minus three equals seven in standard arithmetic."),
    ]
    parts: list[str] = []
    for q, chosen, rejected in pairs:
        parts.append(f"### Prompt\n{q}\n### Chosen\n{chosen}\n### Rejected\n{rejected}\n")
    return "\n".join(parts)


def _append_preference_section(dlm_path, body: str) -> None:  # type: ignore[no-untyped-def]
    existing = dlm_path.read_text(encoding="utf-8")
    # Append only if a preference section isn't already present (the
    # DPO slow test in the same session appends one).
    if "::preference::" in existing:
        return
    dlm_path.write_text(
        existing.rstrip() + "\n\n::preference::\n" + body,
        encoding="utf-8",
    )
