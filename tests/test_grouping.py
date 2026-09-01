r"""Interference grouping: the 1 kHz tolerance *and* the cluster-diameter cap (WO-08 §2).

:func:`aodl.field.focal.group_terms` decides which pupil terms are summed coherently.  Two
rules, and a group has to satisfy both (see its docstring): neighbours more than ``tol``
apart start a new group, and no group may *span* more than ``tol``.

The second rule is the one this file exists for.  Single-linkage chaining alone is
transitive, and the pathological input is exactly the object M2 introduces — a tone ladder.
Under the old 10 kHz default a 40-tone array spaced 9 kHz would chain into a single
"coherent" 360 kHz-wide group, i.e. 40 tweezers declared mutually interfering when every
pair of them beats at 9 kHz.  Meanwhile the genuinely coherent case — an IM3 product landing
*on* a fundamental (Eqs. S20-S22), or a shadow-tweezer pair (Fig. S6) — is degenerate to a
few ULP, so nothing legitimate lives between 0 Hz and a tone spacing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from aodl.field.focal import GROUP_TOL, group_terms
from aodl.units import MHz, kHz


@dataclass
class _Terms:
    """Minimal structural stand-in: grouping reads ``df_opt`` and nothing else."""

    df_opt: Any

    c: Any = None
    theta1: Any = None
    theta2: Any = None
    alpha: Any = None
    edge: Any = (None, None)


def _terms(df_opt) -> _Terms:
    return _Terms(df_opt=np.asarray(df_opt, dtype=np.float64))


def _values(groups, df) -> list[np.ndarray]:
    return [np.asarray(df)[idx] for idx in groups]


def _check_partition(groups, df, tol) -> None:
    """Every group is within ``tol`` end to end, and the groups partition the terms."""
    for idx in groups:
        values = np.asarray(df)[idx]
        assert values.max() - values.min() <= tol + 1e-12
        assert np.all(np.diff(idx) > 0)  # ascending inside a group
    union = np.sort(np.concatenate([np.asarray(idx) for idx in groups])) if groups else np.array([])
    np.testing.assert_array_equal(union, np.arange(len(df)))


# ================================================================== the default tolerance


def test_default_tolerance_is_one_kilohertz():
    assert GROUP_TOL == pytest.approx(1.0 * kHz)


def test_a_ladder_does_not_chain(params1030):
    """40 terms spaced 9 kHz: 40 groups, not one (the finding this rule fixes)."""
    df = np.arange(40) * 9.0 * kHz
    groups = group_terms(_terms(df))
    assert len(groups) == 40
    assert [idx.tolist() for idx in groups] == [[i] for i in range(40)]
    _check_partition(groups, df, GROUP_TOL)

    # ... and it is the *chaining* rule that already separates them here: no cut is needed
    # until the spacing drops below the tolerance.
    tight = np.arange(40) * 0.4 * kHz  # 15.6 kHz wide, every gap under tol
    capped = group_terms(_terms(tight))
    assert len(capped) > 1
    _check_partition(capped, tight, GROUP_TOL)


def test_exact_degeneracies_merge():
    """An IM3 product landing on a fundamental agrees to a few ULP: one coherent group."""
    base = 3.0 * MHz
    df = np.array([base, base * (1.0 + 2e-16), -base, base])
    groups = group_terms(_terms(df))
    assert [idx.tolist() for idx in groups] == [[2], [0, 1, 3]]
    _check_partition(groups, df, GROUP_TOL)


def test_tol_kwarg_still_works():
    df = np.array([0.0, 5.0 * kHz, 1.0 * MHz])
    assert [idx.tolist() for idx in group_terms(_terms(df))] == [[0], [1], [2]]
    assert [idx.tolist() for idx in group_terms(_terms(df), tol=10.0 * kHz)] == [[0, 1], [2]]
    assert [idx.tolist() for idx in group_terms(_terms(df), tol=2.0 * MHz)] == [[0, 1, 2]]
    assert [idx.tolist() for idx in group_terms(_terms(df), tol=0.0)] == [[0], [1], [2]]


def test_empty_and_single_term():
    assert group_terms(_terms([])) == []
    assert [idx.tolist() for idx in group_terms(_terms([7.0]))] == [[0]]


# ====================================================================== the diameter cap


def test_oversized_cluster_splits_at_its_largest_gaps():
    """Three clusters; the middle one chains but is too wide, so it is cut at its gaps.

    Values [kHz]: an isolated 0; then 10, 10.6, 11.2, 12.0, 12.8 — every neighbour gap is
    under the 1 kHz tolerance, so chaining glues all five into a 2.8 kHz-wide cluster; then
    an isolated 30.  The cap cuts the middle cluster at its largest internal gap (0.8 kHz,
    between 11.2 and 12.0 — the first of the two 0.8 kHz gaps, so the tie is broken at the
    lower frequency), leaving [10, 10.6, 11.2] (1.2 kHz) still oversized, which is cut again
    at its own largest gap (0.6 kHz, again the first).
    """
    df = np.array([0.0, 10.0, 10.6, 11.2, 12.0, 12.8, 30.0]) * kHz
    groups = group_terms(_terms(df))
    assert [idx.tolist() for idx in groups] == [[0], [1], [2, 3], [4, 5], [6]]
    _check_partition(groups, df, GROUP_TOL)
    np.testing.assert_allclose(
        [values.max() - values.min() for values in _values(groups, df)],
        [0.0, 0.0, 0.6 * kHz, 0.8 * kHz, 0.0],
        rtol=1e-9,
        atol=1e-6,
    )


def test_uniform_chain_is_cut_into_fitting_pieces():
    """A long chain at half the tolerance: no group wider than ``tol``, order preserved."""
    df = np.arange(64) * 0.5 * kHz
    groups = group_terms(_terms(df))
    _check_partition(groups, df, GROUP_TOL)
    starts = [float(df[idx[0]]) for idx in groups]
    assert starts == sorted(starts)  # groups come out in frequency order
    span = float(df[-1] - df[0])
    assert len(groups) >= span / GROUP_TOL  # a width bound no partition can beat


def test_the_cap_never_splits_what_fits():
    """A cluster already inside the tolerance is left alone, however dense it is."""
    df = np.array([0.0, 1e-3, 0.4 * kHz, 0.9 * kHz])
    assert [idx.tolist() for idx in group_terms(_terms(df))] == [[0, 1, 2, 3]]


# ============================================================================ fuzz + order


def test_random_scenes_are_partitioned_deterministically(rng):
    """1000 random frequency sets: diameters ≤ tol, union preserved, repeatable."""
    tol = GROUP_TOL
    for _ in range(1000):
        n = int(rng.integers(1, 40))
        scale = float(10.0 ** rng.uniform(1.0, 6.0))  # 10 Hz .. 1 MHz spreads
        df = rng.normal(0.0, scale, size=n)
        if n > 3:  # sprinkle exact duplicates, the coherent case
            df[: n // 4] = df[0]
        terms = _terms(df)
        groups = group_terms(terms, tol)
        _check_partition(groups, df, tol)
        again = group_terms(terms, tol)
        assert [idx.tolist() for idx in groups] == [idx.tolist() for idx in again]
        # groups are ordered by frequency, and no two groups interleave
        edges = [(float(df[idx].min()), float(df[idx].max())) for idx in groups]
        assert edges == sorted(edges)
        for (_, hi), (lo, _) in zip(edges[:-1], edges[1:], strict=True):
            assert lo >= hi


def test_grouping_is_independent_of_input_order(rng):
    """Shuffling the terms permutes the group members, never the partition itself."""
    df = np.concatenate([rng.normal(0.0, 5.0 * kHz, 30), np.full(4, 1.0 * MHz)])
    reference = [np.sort(df[idx]) for idx in group_terms(_terms(df))]
    for _ in range(20):
        perm = rng.permutation(df.size)
        shuffled = [np.sort(df[perm][idx]) for idx in group_terms(_terms(df[perm]))]
        assert len(shuffled) == len(reference)
        for a, b in zip(reference, shuffled, strict=True):
            np.testing.assert_allclose(a, b)
