"""`dlm.export.arch_probe` — SUPPORTED / PARTIAL / UNSUPPORTED verdicts.

Covers:

- Fixture `convert_hf_to_gguf.py` with a TextModel registration → SUPPORTED.
- Fixture with MmprojModel-only registration → PARTIAL.
- Fixture without the arch at all → UNSUPPORTED.
- Live probe against the vendored tree: Qwen2-VL is SUPPORTED, PaliGemma
  and InternVL2 are UNSUPPORTED at the pinned tag (as of Sprint 35.4).
  The test pins those expectations; a llama.cpp bump that flips them
  fails the test, forcing an explicit docs/cookbook refresh.
- Cache memoization across repeat calls.
- `VendoringError` surfaces when the script doesn't exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm.export.arch_probe import (
    ArchProbeResult,
    SupportLevel,
    clear_cache,
    probe_gguf_arch,
)
from dlm.export.errors import VendoringError
from dlm.export.vendoring import llama_cpp_root


@pytest.fixture(autouse=True)
def _isolate_cache() -> None:
    """Each test gets a clean memoization table so fixtures don't bleed."""
    clear_cache()
    yield
    clear_cache()


def _fixture_llama_cpp(tmp_path: Path, script_body: str) -> Path:
    """Write a minimal llama.cpp layout with the given convert script body."""
    root = tmp_path / "llama.cpp"
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_text(script_body, encoding="utf-8")
    # Pinned-tag is optional — omit VERSION so the probe reports None.
    return root


class TestUnsupportedVerdict:
    def test_arch_absent_from_script(self, tmp_path: Path) -> None:
        root = _fixture_llama_cpp(
            tmp_path,
            "# Some other model only.\n"
            '@ModelBase.register("LlamaForCausalLM")\n'
            "class LlamaModel(TextModel):\n"
            "    pass\n",
        )
        result = probe_gguf_arch(
            "PaliGemmaForConditionalGeneration",
            llama_cpp_root=root,
        )
        assert isinstance(result, ArchProbeResult)
        assert result.arch_class == "PaliGemmaForConditionalGeneration"
        assert result.support is SupportLevel.UNSUPPORTED
        assert "not found" in result.reason.lower()

    def test_unsupported_reason_names_arch_and_tag(self, tmp_path: Path) -> None:
        root = _fixture_llama_cpp(tmp_path, "# empty script\n")
        (root / "VERSION").write_text("b4321\n", encoding="utf-8")
        result = probe_gguf_arch(
            "InternVLChatModel",
            llama_cpp_root=root,
        )
        assert "InternVLChatModel" in result.reason
        assert "b4321" in result.reason
        assert result.llama_cpp_tag == "b4321"


class TestSupportedVerdict:
    def test_textmodel_binding_is_supported(self, tmp_path: Path) -> None:
        root = _fixture_llama_cpp(
            tmp_path,
            '@ModelBase.register("Qwen2VLModel", "Qwen2VLForConditionalGeneration")\n'
            "class Qwen2VLModel(TextModel):\n"
            "    pass\n",
        )
        result = probe_gguf_arch(
            "Qwen2VLForConditionalGeneration",
            llama_cpp_root=root,
        )
        assert result.support is SupportLevel.SUPPORTED
        assert "Qwen2VLModel" in result.reason

    def test_dual_registration_prefers_textmodel(self, tmp_path: Path) -> None:
        """When an arch is registered on both a TextModel and an MmprojModel,
        the TextModel binding wins → SUPPORTED. This mirrors the actual
        Qwen2-VL layout in upstream llama.cpp."""
        root = _fixture_llama_cpp(
            tmp_path,
            '@ModelBase.register("Qwen2VLForConditionalGeneration")\n'
            "class Qwen2VLTextModel(TextModel):\n"
            "    pass\n"
            "\n"
            '@ModelBase.register("Qwen2VLForConditionalGeneration")\n'
            "class Qwen2VLVisionModel(MmprojModel):\n"
            "    pass\n",
        )
        result = probe_gguf_arch(
            "Qwen2VLForConditionalGeneration",
            llama_cpp_root=root,
        )
        assert result.support is SupportLevel.SUPPORTED


class TestPartialVerdict:
    def test_mmproj_only_binding_is_partial(self, tmp_path: Path) -> None:
        root = _fixture_llama_cpp(
            tmp_path,
            '@ModelBase.register("SomeVLArch")\nclass SomeVisionTower(MmprojModel):\n    pass\n',
        )
        result = probe_gguf_arch("SomeVLArch", llama_cpp_root=root)
        assert result.support is SupportLevel.PARTIAL
        assert "MmprojModel" in result.reason

    def test_partial_reason_names_mmproj_class(self, tmp_path: Path) -> None:
        root = _fixture_llama_cpp(
            tmp_path,
            '@ModelBase.register("FooForConditionalGeneration")\n'
            "class FooVisionTower(MmprojModel):\n"
            "    pass\n",
        )
        result = probe_gguf_arch(
            "FooForConditionalGeneration",
            llama_cpp_root=root,
        )
        assert "FooVisionTower" in result.reason


class TestGrammarEdgeCases:
    def test_single_quoted_arch_name(self, tmp_path: Path) -> None:
        """Register decorators sometimes use single quotes; still match."""
        root = _fixture_llama_cpp(
            tmp_path,
            "@ModelBase.register('FooForCausalLM')\nclass FooModel(TextModel):\n    pass\n",
        )
        result = probe_gguf_arch("FooForCausalLM", llama_cpp_root=root)
        assert result.support is SupportLevel.SUPPORTED

    def test_multiline_decorator_args(self, tmp_path: Path) -> None:
        """Decorators with arg lists wrapped across lines still parse."""
        root = _fixture_llama_cpp(
            tmp_path,
            "@ModelBase.register(\n"
            '    "Qwen2VLModel",\n'
            '    "Qwen2VLForConditionalGeneration",\n'
            ")\n"
            "class Qwen2VLModel(TextModel):\n"
            "    pass\n",
        )
        result = probe_gguf_arch(
            "Qwen2VLForConditionalGeneration",
            llama_cpp_root=root,
        )
        assert result.support is SupportLevel.SUPPORTED

    def test_substring_match_does_not_fire(self, tmp_path: Path) -> None:
        """`"Gemma3..."` should not match `"Gemma"` — use full quoted name."""
        root = _fixture_llama_cpp(
            tmp_path,
            '@ModelBase.register("Gemma3ForCausalLM")\nclass Gemma3Model(TextModel):\n    pass\n',
        )
        result = probe_gguf_arch("GemmaForCausalLM", llama_cpp_root=root)
        # "GemmaForCausalLM" (without the 3) isn't registered.
        assert result.support is SupportLevel.UNSUPPORTED


class TestMemoization:
    def test_repeat_calls_hit_cache(self, tmp_path: Path) -> None:
        """The second call must not re-read the script — proven by
        swapping the file contents and confirming the cached verdict
        persists."""
        root = _fixture_llama_cpp(
            tmp_path,
            '@ModelBase.register("Arch1")\nclass Arch1Model(TextModel):\n    pass\n',
        )
        (root / "VERSION").write_text("tag-v1\n", encoding="utf-8")
        first = probe_gguf_arch("Arch1", llama_cpp_root=root)
        assert first.support is SupportLevel.SUPPORTED

        # Rewrite the script so a re-read would flip the verdict to
        # UNSUPPORTED — the cache must defeat this.
        (root / "convert_hf_to_gguf.py").write_text("# No registrations now.\n", encoding="utf-8")
        second = probe_gguf_arch("Arch1", llama_cpp_root=root)
        assert second is first

    def test_tag_bump_invalidates_cache(self, tmp_path: Path) -> None:
        """Changing the VERSION file (a llama.cpp bump) produces a
        distinct cache key, so the probe re-reads and may return a
        different verdict."""
        root = _fixture_llama_cpp(
            tmp_path,
            "# No registrations.\n",
        )
        (root / "VERSION").write_text("tag-v1\n", encoding="utf-8")
        first = probe_gguf_arch("Arch1", llama_cpp_root=root)
        assert first.support is SupportLevel.UNSUPPORTED

        # Bump the tag AND add the registration.
        (root / "VERSION").write_text("tag-v2\n", encoding="utf-8")
        (root / "convert_hf_to_gguf.py").write_text(
            '@ModelBase.register("Arch1")\nclass Arch1Model(TextModel):\n    pass\n',
            encoding="utf-8",
        )
        second = probe_gguf_arch("Arch1", llama_cpp_root=root)
        assert second.support is SupportLevel.SUPPORTED
        assert second.llama_cpp_tag == "tag-v2"


class TestMissingScript:
    def test_missing_convert_script_raises(self, tmp_path: Path) -> None:
        """Directory exists + has other files but not convert_hf_to_gguf.py.

        (An empty directory hits the earlier "source tree is empty" guard
        in `llama_cpp_root`; we want the convert-script-missing path here.)
        """
        root = tmp_path / "partial-llama.cpp"
        root.mkdir()
        (root / "README.md").write_text("", encoding="utf-8")
        with pytest.raises(VendoringError, match="convert_hf_to_gguf.py"):
            probe_gguf_arch("AnyArch", llama_cpp_root=root)


# --- Live-tree assertions -----------------------------------------------
# These test the actual pinned vendored llama.cpp — flag vendor bumps
# that change support levels for the three registered VL bases.


class TestLiveVendoredTree:
    """Verdicts against the current vendored llama.cpp.

    A llama.cpp bump that changes these expectations is a meaningful
    event — it flips users from the HF-snapshot fallback to the GGUF
    path (or vice versa). Failing here forces the cookbook + vl-memory
    docs to be refreshed in the same commit.
    """

    def test_paligemma_unsupported(self) -> None:
        _require_live_vendored_tree()
        result = probe_gguf_arch("PaliGemmaForConditionalGeneration")
        assert result.support is SupportLevel.UNSUPPORTED

    def test_qwen2vl_supported(self) -> None:
        _require_live_vendored_tree()
        result = probe_gguf_arch("Qwen2VLForConditionalGeneration")
        assert result.support is SupportLevel.SUPPORTED

    def test_internvl2_unsupported(self) -> None:
        _require_live_vendored_tree()
        result = probe_gguf_arch("InternVLChatModel")
        assert result.support is SupportLevel.UNSUPPORTED


def _require_live_vendored_tree() -> None:
    try:
        llama_cpp_root()
    except VendoringError as exc:
        pytest.skip(f"live vendored llama.cpp tree unavailable: {exc}")
