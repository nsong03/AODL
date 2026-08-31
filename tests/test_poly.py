"""``PiecewisePoly`` closure tests: exact evaluation, derivative/antiderivative, algebra."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial import polynomial as npoly
from scipy.integrate import quad

from aodl.poly import MAX_DEGREE, PiecewisePoly
from aodl.units import um, us

RTOL = 1e-12


def _min_jerk(x_i: float, x_f: float, t0: float, t1: float) -> PiecewisePoly:
    """Eq. S14 min-jerk quintic in normalized time: x_i + d * (10 t^3 - 15 t^4 + 6 t^5)."""
    d = x_f - x_i
    coeffs = np.array([[x_i, 0.0, 0.0, 10.0 * d, -15.0 * d, 6.0 * d]])
    return PiecewisePoly.from_segment_coeffs([t0, t1], coeffs)


def test_min_jerk_endpoints_and_calculus_closure():
    t_end = 12.0 * us
    x_i, x_f = -3.0 * um, 5.0 * um
    scale = abs(x_f - x_i)
    p = _min_jerk(x_i, x_f, 0.0, t_end)

    assert p(0.0) == pytest.approx(x_i, rel=RTOL)
    assert p(t_end) == pytest.approx(x_f, rel=RTOL)

    dp = p.derivative()
    v_scale = scale / t_end
    assert abs(dp(0.0)) < RTOL * v_scale
    assert abs(dp(t_end)) < RTOL * v_scale
    # Peak min-jerk speed is 15/8 * d / T (analytic).
    t = np.linspace(0.0, t_end, 4001)
    assert np.max(np.abs(dp(t))) == pytest.approx(1.875 * scale / t_end, rel=1e-6)

    # antiderivative(derivative(p)) - p must be a constant.
    round_trip = dp.antiderivative()
    diff = round_trip(t) - p(t)
    assert np.ptp(diff) < RTOL * scale


def test_random_piecewise_matches_numpy_polynomial(rng):
    breaks = np.cumsum(np.concatenate([[0.0], rng.uniform(0.5, 3.0, size=5)])) * us
    degree = 6
    coeffs = rng.normal(size=(5, degree + 1))
    p = PiecewisePoly.from_segment_coeffs(breaks, coeffs)
    assert p.degree == degree
    assert p.n_segments == 5
    assert p.domain == (breaks[0], breaks[-1])

    t = np.sort(rng.uniform(breaks[0], breaks[-1], size=1000))
    k = np.clip(np.searchsorted(breaks, t, side="right") - 1, 0, 4)
    tau = (t - breaks[k]) / (breaks[k + 1] - breaks[k])
    expected = np.array([npoly.polyval(tau[i], coeffs[k[i]]) for i in range(t.size)])
    scale = np.max(np.abs(expected))
    assert np.max(np.abs(p(t) - expected)) < RTOL * scale


def test_clamp_hold_outside_domain():
    p = _min_jerk(0.0, 4.0 * um, 2.0 * us, 9.0 * us)
    t0, t1 = p.domain
    assert p(t0 - 1.0) == p(t0)
    assert p(-1e9) == p(t0)
    assert p(t1 + 1.0) == p(t1)
    outside = p(np.array([t0 - 5.0 * us, t0, t1, t1 + 5.0 * us]))
    assert outside[0] == outside[1]
    assert outside[2] == outside[3]
    # Scalar in -> float out; array in -> array out shaped like the input.
    assert isinstance(p(0.0), float)
    assert p(np.zeros((3, 2))).shape == (3, 2)


def test_add_refines_to_break_union(rng):
    t_end = 6.0 * us
    p = PiecewisePoly.from_segment_coeffs([0.0, 1.0 * us, t_end], rng.normal(size=(2, 4)))
    q = PiecewisePoly.from_segment_coeffs([0.0, 2.5 * us, 4.0 * us, t_end], rng.normal(size=(3, 3)))
    s = p + q
    np.testing.assert_allclose(
        s.breaks, np.array([0.0, 1.0 * us, 2.5 * us, 4.0 * us, t_end]), rtol=0, atol=0
    )
    t = np.linspace(0.0, t_end, 777)
    expected = p(t) + q(t)
    scale = np.max(np.abs(expected))
    assert np.max(np.abs(s(t) - expected)) < RTOL * scale

    # scalar shorthand and subtraction
    assert np.max(np.abs((p + 3.0)(t) - (p(t) + 3.0))) < RTOL * max(scale, 3.0)
    assert np.max(np.abs((p - q)(t) - (p(t) - q(t)))) < RTOL * scale

    with pytest.raises(ValueError, match="identical domains"):
        _ = p + q.shift(1.0 * us)


def test_concat_and_shift(rng):
    a = PiecewisePoly.from_segment_coeffs([0.0, 2.0 * us], rng.normal(size=(1, 5)))
    b = PiecewisePoly.from_segment_coeffs([2.0 * us, 3.0 * us, 7.0 * us], rng.normal(size=(2, 2)))
    c = PiecewisePoly.concat([a, b])
    assert c.n_segments == 3
    assert c.degree == 4
    assert c.domain == (0.0, 7.0 * us)
    ta = np.linspace(0.0, 2.0 * us, 101)[:-1]
    tb = np.linspace(2.0 * us, 7.0 * us, 101)
    scale = max(np.max(np.abs(a(ta))), np.max(np.abs(b(tb))))
    assert np.max(np.abs(c(ta) - a(ta))) < RTOL * scale
    assert np.max(np.abs(c(tb) - b(tb))) < RTOL * scale

    with pytest.raises(ValueError, match="contiguous"):
        PiecewisePoly.concat([a, b.shift(1.0 * us)])

    dt = 4.0 * us
    shifted = a.shift(dt)
    assert shifted.domain == (dt, 2.0 * us + dt)
    assert np.max(np.abs(shifted(ta + dt) - a(ta))) < RTOL * scale

    # scale / offset / constant
    assert np.max(np.abs(a.scale(-2.5)(ta) + 2.5 * a(ta))) < RTOL * scale
    assert np.max(np.abs(a.offset(7.0)(ta) - a(ta) - 7.0)) < RTOL * max(scale, 7.0)
    k = PiecewisePoly.constant(1.5, 0.0, 1.0)
    assert k(0.3) == 1.5
    assert k.derivative()(0.3) == 0.0


def test_antiderivative_is_continuous_across_breaks(rng):
    """Phase continuity: the phase integral must not jump when the frequency law does."""
    breaks = np.array([0.0, 1.3, 2.0, 5.5, 6.0]) * us
    coeffs = rng.normal(size=(4, 5)) * 1e6  # MHz-scale, deliberately discontinuous
    p = PiecewisePoly.from_segment_coeffs(breaks, coeffs)
    # values genuinely jump at the interior breaks
    eps = 1e-15
    for b in breaks[1:-1]:
        assert abs(p(b) - p(b - eps)) > 1.0

    a = p.antiderivative(c0=0.4)
    assert a(breaks[0]) == pytest.approx(0.4, rel=RTOL)
    scale = np.max(np.abs(a(np.linspace(breaks[0], breaks[-1], 2001))))
    for k in range(1, a.n_segments):
        left = float(a.coeffs[k - 1].sum())  # segment k-1 at tau = 1
        right = float(a.coeffs[k, 0])  # segment k at tau = 0
        assert abs(left - right) < RTOL * scale

    # and it really is the integral of p (quad told where the jumps are)
    interior = list(breaks[1:-1])
    for t_test in (0.7 * us, 1.3 * us, 3.7 * us, 6.0 * us):
        points = [b for b in interior if breaks[0] < b < t_test]
        value, _ = quad(p, breaks[0], t_test, points=points or None, limit=200)
        assert a(t_test) == pytest.approx(0.4 + value, abs=RTOL * scale)


def test_construction_validation():
    with pytest.raises(ValueError, match="strictly increasing"):
        PiecewisePoly(np.array([0.0, 1.0, 1.0]), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="one row per segment"):
        PiecewisePoly(np.array([0.0, 1.0, 2.0]), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="MAX_DEGREE"):
        PiecewisePoly(np.array([0.0, 1.0]), np.zeros((1, MAX_DEGREE + 2)))
    with pytest.raises(ValueError, match="MAX_DEGREE"):
        PiecewisePoly(np.array([0.0, 1.0]), np.zeros((1, MAX_DEGREE + 1))).antiderivative()
