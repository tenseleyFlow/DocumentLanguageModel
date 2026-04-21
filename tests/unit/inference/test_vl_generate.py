"""VL generate — prompt formatting + image loading (Sprint 35 v1).

The real `generate_vl` + `load_for_vl_inference` paths are pragma'd
(they need a real VL HF model); these tests cover the pure helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from dlm.inference.vl_generate import format_vl_prompt, load_images


class TestFormatVlPrompt:
    def test_prepends_token_before_text(self) -> None:
        assert (
            format_vl_prompt("describe this", image_token="<image>", num_images=1)
            == "<image>\ndescribe this"
        )

    def test_multiple_images_repeat_token(self) -> None:
        out = format_vl_prompt("compare", image_token="<image>", num_images=3)
        assert out == "<image><image><image>\ncompare"

    def test_empty_prompt_emits_tokens_only(self) -> None:
        # Passing no text is valid — the user wants a caption for the
        # image and has nothing else to say. Output is the placeholders
        # alone; no trailing separator.
        out = format_vl_prompt("", image_token="<image>", num_images=2)
        assert out == "<image><image>"

    def test_user_placed_token_respected(self) -> None:
        # When the prompt already mentions the image token, we don't
        # prepend anything — the user has placed it deliberately.
        prompt = "Compare the before <image> and after <image> shots."
        assert (
            format_vl_prompt(prompt, image_token="<image>", num_images=2) == prompt
        )

    def test_custom_image_token(self) -> None:
        out = format_vl_prompt(
            "describe",
            image_token="<|vision|>",
            num_images=1,
        )
        assert out == "<|vision|>\ndescribe"


class TestLoadImages:
    def _write_png(self, path: Path, color: tuple[int, int, int]) -> None:
        Image.new("RGB", (2, 2), color=color).save(path, format="PNG")

    def test_loads_single_image(self, tmp_path: Path) -> None:
        p = tmp_path / "a.png"
        self._write_png(p, (255, 0, 0))
        images = load_images([p])
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
        assert images[0].mode == "RGB"

    def test_loads_multiple_images_preserves_order(self, tmp_path: Path) -> None:
        a = tmp_path / "a.png"
        b = tmp_path / "b.png"
        self._write_png(a, (255, 0, 0))
        self._write_png(b, (0, 255, 0))
        [first, second] = load_images([a, b])
        assert first.getpixel((0, 0)) == (255, 0, 0)
        assert second.getpixel((0, 0)) == (0, 255, 0)

    def test_missing_file_raises_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="image not found"):
            load_images([tmp_path / "nope.png"])

    def test_non_image_raises_pil_error(self, tmp_path: Path) -> None:
        bogus = tmp_path / "x.png"
        bogus.write_text("not an image", encoding="utf-8")
        with pytest.raises(Exception):  # noqa: B017 — PIL's UnidentifiedImageError
            load_images([bogus])

    def test_converts_to_rgb_from_other_modes(self, tmp_path: Path) -> None:
        # Save a 4-channel RGBA image; loader must still emit RGB.
        p = tmp_path / "rgba.png"
        Image.new("RGBA", (2, 2), color=(255, 0, 0, 128)).save(p, format="PNG")
        [image] = load_images([p])
        assert image.mode == "RGB"
