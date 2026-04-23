"""Shared export-record helpers used by GGUF and non-GGUF targets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dlm.export.manifest import utc_now

if TYPE_CHECKING:
    from dlm.store.paths import StorePath


def append_export_summary(
    *,
    store: StorePath,
    quant: str,
    merged: bool,
    target: str,
    llama_cpp_tag: str | None,
    artifacts: list[Any],
    ollama_name: str | None,
    ollama_version_str: str | None,
    smoke_first_line: str | None,
    adapter_name: str | None = None,
    adapter_mix: list[tuple[str, float]] | None = None,
    timeout: float = 60.0,
) -> None:
    """Update `manifest.exports` with one new export row."""
    from dlm.store.lock import exclusive
    from dlm.store.manifest import ExportSummary, load_manifest, save_manifest

    base_sha = next((a.sha256 for a in artifacts if a.path.startswith("base.")), None)
    adapter_sha = next((a.sha256 for a in artifacts if a.path.startswith("adapter.")), None)

    summary = ExportSummary(
        exported_at=utc_now(),
        target=target,
        quant=quant,
        merged=merged,
        ollama_name=ollama_name,
        ollama_version=ollama_version_str,
        llama_cpp_tag=llama_cpp_tag,
        base_gguf_sha256=base_sha,
        adapter_gguf_sha256=adapter_sha,
        smoke_output_first_line=smoke_first_line,
        adapter_name=adapter_name,
        adapter_mix=adapter_mix,
    )

    with exclusive(store.lock, timeout=timeout):
        manifest = load_manifest(store.manifest)
        updated = manifest.model_copy(
            update={
                "exports": [*manifest.exports, summary],
                "updated_at": utc_now(),
            }
        )
        save_manifest(store.manifest, updated)
