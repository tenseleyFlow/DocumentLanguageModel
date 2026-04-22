"""Audio-language modality dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dlm.modality.registry import ModalityDispatch

if TYPE_CHECKING:
    from dlm.base_models import BaseModelSpec
    from dlm.export.dispatch import DispatchResult


class AudioLanguageModality(ModalityDispatch):
    """Audio-language base — audio accepted, processor required, HF-snapshot export."""

    modality = "audio-language"
    requires_processor = True
    accepts_audio = True

    def load_processor(self, spec: BaseModelSpec) -> Any:
        from dlm.train.loader import load_processor as _load

        return _load(spec)

    def dispatch_export(
        self,
        *,
        store: Any,
        spec: BaseModelSpec,
        adapter_name: str | None,
        quant: str | None,
        merged: bool,
        adapter_mix_raw: str | None,
        gguf_emission_context: dict[str, Any] | None = None,
    ) -> DispatchResult:
        from dlm.export.dispatch import dispatch_audio_export

        return dispatch_audio_export(
            store=store,
            spec=spec,
            adapter_name=adapter_name,
            quant=quant,
            merged=merged,
            adapter_mix_raw=adapter_mix_raw,
        )
