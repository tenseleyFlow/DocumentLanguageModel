"""Extract a steering direction from paired hidden states.

The math: given N preference pairs, each mapped through the base
model to `(hidden_chosen_i, hidden_rejected_i)` at some residual-
stream layer, the difference `d_i = hidden_chosen_i - hidden_rejected_i`
is a "pull toward chosen" vector for that example. The first
principal component of the stack of differences is the direction
these pulls agree on — the steering direction that captures the
preference shared across examples.

We compute the raw (uncentered) SVD of the difference stack —
matching the "Steering Llama" literature (Panickssery et al.).
When every pair agrees, the principal component is the common
direction; when pairs disagree, it's the direction maximizing
the sum of squared projections. Sign is fixed by aligning with
the mean pull, so extraction is reproducible across numpy
versions. A single example's direction collapses to itself
normalized — the expected limit case.

The unit-test path takes synthetic NumPy arrays; no HF model
needed. Wiring a real base model's forward hooks to produce the
hidden states is a later-sprint concern.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from dlm.control.errors import ControlExtractError, ControlPolicyRefusal


@dataclass(frozen=True)
class ControlVector:
    """A single extracted steering direction.

    `direction` is a unit vector of length `hidden_dim`.
    `n_pairs` lets callers reconstruct how many examples fed the
    extraction when rendering audit output. `explained_variance`
    is the leading singular value squared over the total — a 1.0
    reading means every pair agreed perfectly, while 0.25 means
    the principal component explains a quarter of the preference
    spread (the rest is noise or contradictory pairs).
    """

    direction: np.ndarray
    n_pairs: int
    explained_variance: float


def extract_control_vector(
    hidden_chosen: np.ndarray,
    hidden_rejected: np.ndarray,
) -> ControlVector:
    """Compute the steering direction from paired hidden states.

    `hidden_chosen` / `hidden_rejected` are `(N, hidden_dim)` float
    arrays of hidden states at one residual-stream layer. The output
    `direction` is a unit vector oriented so that positive strength
    pushes toward `chosen`.

    Raises `ControlExtractError` on:
    - mismatched shapes
    - `N < 1`
    - non-finite entries (NaN hidden states from a bad forward pass)
    - zero-variance differences (every chosen identical to rejected →
      no signal to extract)
    """
    if hidden_chosen.shape != hidden_rejected.shape:
        raise ControlExtractError(
            f"chosen/rejected shape mismatch: {hidden_chosen.shape} vs {hidden_rejected.shape}"
        )
    if hidden_chosen.ndim != 2:
        raise ControlExtractError(f"expected 2D (N, hidden_dim) arrays, got {hidden_chosen.ndim}D")
    if hidden_chosen.shape[0] < 1:
        raise ControlExtractError("need at least one (chosen, rejected) pair")
    if not (np.isfinite(hidden_chosen).all() and np.isfinite(hidden_rejected).all()):
        raise ControlExtractError("hidden states contain non-finite values")

    diffs = hidden_chosen.astype(np.float64) - hidden_rejected.astype(np.float64)
    # Single-pair limit case: the direction is just that pair,
    # normalized. SVD on one row works but this short-circuit keeps
    # the explained-variance denominator well-defined (it's 1.0 by
    # definition when there's only one component to explain).
    if diffs.shape[0] == 1:
        norm = float(np.linalg.norm(diffs[0]))
        if norm == 0.0:
            raise ControlExtractError("single pair has zero chosen/rejected difference")
        return ControlVector(
            direction=(diffs[0] / norm).astype(np.float32),
            n_pairs=1,
            explained_variance=1.0,
        )

    # Raw (uncentered) SVD on the difference stack — matches the
    # control-vector literature (Panickssery et al., "Steering
    # Llama"). Centering would wipe the signal when every pair
    # agrees exactly; uncentered, the principal component is the
    # direction maximizing the sum of squared projections, which
    # coincides with the mean pull when all diffs align and tracks
    # the dominant direction otherwise.
    total_energy = float(np.sum(diffs**2))
    if total_energy == 0.0:
        raise ControlExtractError(
            "zero chosen/rejected differences across all pairs — no signal to extract"
        )

    # Thin SVD: full_matrices=False so we don't allocate an
    # (N, N) left matrix we never use.
    _u, singular_values, vh = np.linalg.svd(diffs, full_matrices=False)
    # Principal direction is the first right-singular vector.
    direction = vh[0]

    # Orient so that the direction points *toward* chosen. Without
    # this the sign of the first singular vector is arbitrary (SVD
    # is unique only up to sign), which would make extraction
    # non-reproducible across numpy versions. Convention: positive
    # strength pushes toward chosen, so align with mean(diffs).
    mean_pull = diffs.mean(axis=0)
    if float(np.dot(direction, mean_pull)) < 0:
        direction = -direction

    explained = float(singular_values[0] ** 2 / total_energy)
    return ControlVector(
        direction=direction.astype(np.float32),
        n_pairs=int(diffs.shape[0]),
        explained_variance=explained,
    )


_SAFETY_POLICY_VALUE = "safety"
_POLICY_TAG_KEY = "policy"


def refuse_if_policy_safety(
    section_tags: Iterable[Mapping[str, str]],
) -> None:
    """Refuse extraction when any source section carries `policy: safety`.

    A control vector over safety-flagged preference pairs would, by
    construction, be a "more safety vs less safety" steering
    direction — applied at negative strength, it erodes the exact
    behavior the document is trying to preserve. We don't offer the
    user that footgun. The check runs at extraction-entry so the
    artifact never reaches disk.

    Takes a flat iterable of per-section tag dicts so callers can
    pass whatever source their sections were collected from
    (preference sections, a mix of types, etc.). Cost is linear
    in the section count — negligible next to the HF forward pass.
    """
    for tags in section_tags:
        if tags.get(_POLICY_TAG_KEY) == _SAFETY_POLICY_VALUE:
            raise ControlPolicyRefusal(
                "refusing to extract a control vector from preference "
                "sections tagged `policy: safety` — the resulting steering "
                "direction could be used at negative strength to undo the "
                "safety training the document is trying to preserve. "
                "Extract separately from non-safety preferences instead."
            )
