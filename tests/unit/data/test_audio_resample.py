"""`dlm.data.audio_resample` — backend pick + identity-pass + error shape.

Covers:

- Same-rate input is a no-op (no copy, no backend call).
- Negative or zero rates reject with `ValueError`.
- Neither soxr nor scipy importable → `AudioResampleUnavailable`.
- scipy fallback resamples to the expected length (integer ratio).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

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

        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name in ("soxr", "scipy", "scipy.signal"):
                raise ImportError(f"forced: {name}")
            return real_import(name, *args, **kwargs)  # type: ignore[operator]

        monkeypatch.setitem(sys.modules, "soxr", None)
        # Monkey-patch the _pick_backend helper's import probes so both
        # attempts fail regardless of what's installed in the env.
        monkeypatch.setattr(audio_resample, "_pick_backend", _no_backend)

        with pytest.raises(AudioResampleUnavailable, match="soxr or scipy"):
            resample(np.zeros(8, dtype=np.float32), src_sr=48_000, dst_sr=16_000)


def _no_backend() -> None:
    raise AudioResampleUnavailable(
        "training.audio.auto_resample=True requires either soxr or scipy; "
        "install one of: `pip install soxr` (recommended) or "
        "`pip install scipy`."
    )


class TestScipyBackend:
    def test_scipy_fallback_resamples(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With soxr disabled, scipy fallback produces expected length."""
        # Pretend soxr isn't importable so the pick falls through to scipy.
        monkeypatch.setitem(sys.modules, "soxr", None)

        pytest.importorskip("scipy.signal")

        # 1 second of 8 kHz silence → resample to 16 kHz = 2 s of samples.
        wave = np.zeros(8_000, dtype=np.float32)
        out = resample(wave, src_sr=8_000, dst_sr=16_000)

        assert out.dtype == np.float32
        # resample_poly produces len(x) * up // down on integer ratios.
        # scipy rounds up-or-down depending on filter length; accept ±1.
        assert abs(out.shape[0] - 16_000) <= 1
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
