"""`dlm.data.audio_resample` — backend pick + identity-pass + error shape.

Covers:

- Same-rate input is a no-op (no copy, no backend call).
- Negative or zero rates reject with `ValueError`.
- Neither soxr nor scipy importable → `AudioResampleUnavailable`.
- scipy fallback resamples to the expected length (integer ratio).
"""

from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from dlm.data import audio_resample
from dlm.data.audio_resample import AudioResampleUnavailable, resample


class TestIdentityPass:
    def test_same_rate_returns_input(self) -> None:
        wave = np.arange(16, dtype=np.float32)
        out = resample(wave, src_sr=16_000, dst_sr=16_000)
        # Same object — no backend ever invoked, no copy.
        assert out is wave

    def test_zero_src_rejected(self) -> None:
        wave = np.zeros(8, dtype=np.float32)
        with pytest.raises(ValueError, match="must be positive"):
            resample(wave, src_sr=0, dst_sr=16_000)

    def test_zero_dst_rejected(self) -> None:
        wave = np.zeros(8, dtype=np.float32)
        with pytest.raises(ValueError, match="must be positive"):
            resample(wave, src_sr=16_000, dst_sr=0)


class TestBackendPickFailure:
    def test_no_backend_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force both imports to fail and confirm the error names both paths."""

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name in ("soxr", "scipy", "scipy.signal"):
                raise ImportError(f"forced: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(AudioResampleUnavailable, match="soxr or scipy"):
            audio_resample._pick_backend()


def _no_backend() -> None:
    raise AudioResampleUnavailable(
        "training.audio.auto_resample=True requires either soxr or scipy; "
        "install one of: `pip install soxr` (recommended) or "
        "`pip install scipy`."
    )


class TestScipyBackend:
    def test_resample_routes_through_selected_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: dict[str, object] = {}

        def fake_backend(waveform: np.ndarray, *, src_sr: int, dst_sr: int) -> np.ndarray:
            called["waveform"] = waveform
            called["src_sr"] = src_sr
            called["dst_sr"] = dst_sr
            return np.ones(4, dtype=np.float32)

        wave = np.zeros(8, dtype=np.float32)
        monkeypatch.setattr(audio_resample, "_pick_backend", lambda: fake_backend)
        out = resample(wave, src_sr=8_000, dst_sr=16_000)

        assert out.tolist() == [1.0, 1.0, 1.0, 1.0]
        assert called == {"waveform": wave, "src_sr": 8_000, "dst_sr": 16_000}

    def test_scipy_fallback_uses_fake_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With soxr disabled, _pick_backend falls through to scipy."""
        monkeypatch.setitem(sys.modules, "soxr", None)
        fake_signal = ModuleType("scipy.signal")
        fake_signal.resample_poly = lambda waveform, *, up, down: np.repeat(waveform, up)[
            : len(waveform) * up // down
        ]
        fake_scipy = ModuleType("scipy")
        fake_scipy.signal = fake_signal
        monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
        monkeypatch.setitem(sys.modules, "scipy.signal", fake_signal)

        backend = audio_resample._pick_backend()
        assert backend is audio_resample._scipy_resample

    def test_soxr_backend_coerces_float32_contiguous(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_soxr = ModuleType("soxr")
        fake_soxr.resample = lambda waveform, src_sr, dst_sr, quality="HQ": np.asarray(
            waveform * 2, dtype=np.float64
        )
        monkeypatch.setitem(sys.modules, "soxr", fake_soxr)

        wave = np.arange(6, dtype=np.float32)[::2]
        out = audio_resample._soxr_resample(wave, src_sr=8_000, dst_sr=16_000)

        assert out.dtype == np.float32
        assert out.flags.c_contiguous
        assert out.tolist() == [0.0, 4.0, 8.0]

    def test_scipy_backend_reduces_ratio_before_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: dict[str, object] = {}

        def fake_resample_poly(waveform: np.ndarray, *, up: int, down: int) -> np.ndarray:
            calls["up"] = up
            calls["down"] = down
            return np.asarray(waveform + 1, dtype=np.float64)

        fake_signal = ModuleType("scipy.signal")
        fake_signal.resample_poly = fake_resample_poly
        fake_scipy = ModuleType("scipy")
        fake_scipy.signal = fake_signal
        monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
        monkeypatch.setitem(sys.modules, "scipy.signal", fake_signal)

        wave = np.arange(5, dtype=np.float32)
        out = audio_resample._scipy_resample(wave, src_sr=48_000, dst_sr=16_000)

        assert calls == {"up": 1, "down": 3}
        assert out.dtype == np.float32
        assert out.flags.c_contiguous


class TestPickBackendDirect:
    def test_pick_backend_with_soxr_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If soxr import succeeds, _pick_backend returns the soxr callable."""
        fake_soxr = SimpleNamespace(resample=lambda *a, **k: None)
        monkeypatch.setitem(sys.modules, "soxr", fake_soxr)
        backend = audio_resample._pick_backend()
        assert backend is audio_resample._soxr_resample

    def test_pick_backend_falls_to_scipy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If soxr is absent but scipy.signal imports, fall back to scipy."""
        monkeypatch.setitem(sys.modules, "soxr", None)
        pytest.importorskip("scipy.signal")
        backend = audio_resample._pick_backend()
        assert backend is audio_resample._scipy_resample
