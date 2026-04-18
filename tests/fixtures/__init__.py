"""Test fixtures — factories for synthetic .dlm files, hardware mocks,
tiny-model cache, and the golden-output registry.

Importable from tests without pytest's fixture machinery; helpers are
plain functions and context managers.
"""

from __future__ import annotations

from tests.fixtures.dlm_factory import make_dlm
from tests.fixtures.golden import assert_golden, golden_path, load_golden
from tests.fixtures.hardware_mocks import force_cpu, force_cuda, force_mps, force_rocm

__all__ = [
    "assert_golden",
    "force_cpu",
    "force_cuda",
    "force_mps",
    "force_rocm",
    "golden_path",
    "load_golden",
    "make_dlm",
]
