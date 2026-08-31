"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from aodl.params import AODLParams, default_1030

#: Fixed seed so every failure is reproducible.
SEED = 20260831


@pytest.fixture
def params1030() -> AODLParams:
    """Product-default hardware preset (lambda = 1030 nm)."""
    return default_1030()


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator (fresh per test, so tests stay independent)."""
    return np.random.default_rng(SEED)
