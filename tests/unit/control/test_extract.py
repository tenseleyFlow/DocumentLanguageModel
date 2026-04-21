"""PCA-over-residuals — math, edge cases, orientation."""

from __future__ import annotations

import numpy as np
import pytest

from dlm.control import ControlExtractError, ControlVector, extract_control_vector


class TestShape:
    def test_returns_unit_vector(self) -> None:
        rng = np.random.default_rng(0)
        chosen = rng.normal(size=(20, 16))
        rejected = rng.normal(size=(20, 16))
        out = extract_control_vector(chosen, rejected)
        assert isinstance(out, ControlVector)
        assert out.direction.shape == (16,)
        assert np.isclose(np.linalg.norm(out.direction), 1.0, atol=1e-5)

    def test_explained_variance_in_unit_interval(self) -> None:
        rng = np.random.default_rng(1)
        # Inject a coherent signal so the mean-pull is non-orthogonal to
        # the principal direction — otherwise the sign-alignment floor
        # (which guards against near-orthogonal ambiguity) rejects pure
        # Gaussian noise, which is the correct behavior for callers but
        # the wrong fixture for an explained-variance shape assertion.
        signal = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        chosen = rng.normal(size=(50, 8)) + signal
        rejected = rng.normal(size=(50, 8))
        out = extract_control_vector(chosen, rejected)
        assert 0.0 <= out.explained_variance <= 1.0

    def test_n_pairs_reflects_input_rows(self) -> None:
        chosen = np.zeros((7, 4))
        rejected = np.ones((7, 4))
        out = extract_control_vector(chosen, rejected)
        assert out.n_pairs == 7


class TestMath:
    def test_perfectly_agreeing_pairs_yield_that_direction(self) -> None:
        # Every pair has the same difference — the direction should be
        # the normalized version of that constant difference.
        pull = np.array([3.0, 4.0, 0.0])  # 3-4-5 for easy arithmetic
        chosen = np.tile(pull, (10, 1))
        rejected = np.zeros((10, 3))
        out = extract_control_vector(chosen, rejected)
        expected = pull / np.linalg.norm(pull)
        assert np.allclose(np.abs(out.direction), np.abs(expected), atol=1e-5)

    def test_orientation_points_toward_chosen(self) -> None:
        # Deliberately construct a case where the raw SVD sign could
        # go either way: differences oscillate but mean is positive.
        diffs = np.array([[5.0, 0.0], [-3.0, 0.0], [5.0, 0.0], [-3.0, 0.0]])
        chosen = diffs
        rejected = np.zeros_like(diffs)
        out = extract_control_vector(chosen, rejected)
        # Mean pull is (+1, 0) → direction should align with +x.
        assert out.direction[0] > 0

    def test_single_pair_returns_itself_normalized(self) -> None:
        chosen = np.array([[1.0, 2.0, 2.0]])
        rejected = np.array([[0.0, 0.0, 0.0]])
        out = extract_control_vector(chosen, rejected)
        assert out.n_pairs == 1
        assert out.explained_variance == 1.0
        expected = np.array([1.0, 2.0, 2.0]) / 3.0
        assert np.allclose(out.direction, expected, atol=1e-5)


class TestEdgeCases:
    def test_shape_mismatch_raises(self) -> None:
        chosen = np.zeros((10, 8))
        rejected = np.zeros((10, 16))
        with pytest.raises(ControlExtractError, match="shape mismatch"):
            extract_control_vector(chosen, rejected)

    def test_wrong_ndim_raises(self) -> None:
        chosen = np.zeros((10,))
        rejected = np.zeros((10,))
        with pytest.raises(ControlExtractError, match="2D"):
            extract_control_vector(chosen, rejected)

    def test_empty_raises(self) -> None:
        chosen = np.zeros((0, 8))
        rejected = np.zeros((0, 8))
        with pytest.raises(ControlExtractError, match="at least one"):
            extract_control_vector(chosen, rejected)

    def test_nan_inputs_rejected(self) -> None:
        chosen = np.zeros((5, 4))
        chosen[2, 1] = np.nan
        rejected = np.zeros((5, 4))
        with pytest.raises(ControlExtractError, match="non-finite"):
            extract_control_vector(chosen, rejected)

    def test_identical_chosen_rejected_raises(self) -> None:
        # Every pair agrees exactly — there's no signal to extract.
        hidden = np.ones((6, 4))
        with pytest.raises(ControlExtractError, match="zero chosen/rejected"):
            extract_control_vector(hidden, hidden)

    def test_single_pair_zero_diff_raises(self) -> None:
        hidden = np.zeros((1, 4))
        with pytest.raises(ControlExtractError, match="zero chosen/rejected"):
            extract_control_vector(hidden, hidden)

    def test_near_orthogonal_mean_raises(self) -> None:
        # Contradictory pairs: spread in the +x direction, mean orthogonal
        # to it (zero mean in x, nonzero in y). Principal SVD direction is
        # +x but mean_pull is +y → cos_align ≈ 0, the sign decision is
        # noise. Reject rather than ship a coin-flip vector.
        chosen = np.array(
            [
                [5.0, 0.01, 0.0],
                [-5.0, 0.01, 0.0],
                [5.0, 0.01, 0.0],
                [-5.0, 0.01, 0.0],
            ]
        )
        rejected = np.zeros_like(chosen)
        with pytest.raises(ControlExtractError, match="near-orthogonal"):
            extract_control_vector(chosen, rejected)


class TestReproducibility:
    def test_same_inputs_same_output(self) -> None:
        rng = np.random.default_rng(42)
        chosen = rng.normal(size=(30, 12))
        rejected = rng.normal(size=(30, 12))
        a = extract_control_vector(chosen, rejected)
        b = extract_control_vector(chosen, rejected)
        assert np.array_equal(a.direction, b.direction)
        assert a.explained_variance == b.explained_variance
