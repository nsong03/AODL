"""Ramp families (Eqs. S14-S17): endpoint conditions, continuity, exact calculus.

The ramps are the time profiles of both positions and frequency laws, so the endpoint
conditions asserted here are physical statements: ``ydot = 0`` at the ends means the
tweezer starts and stops at rest, and ``yddot = 0`` at the ends of a *frequency* ramp
means it starts and stops with no residual chirp lensing (Table I).
"""

from __future__ import annotations

import numpy as np
import pytest

from aodl.trajectory import ramps
from aodl.units import MHz, um, us

RTOL = 1e-12

#: Every ``(t0, T, y_i, y_f) -> PiecewisePoly`` family.
FAMILIES = [
    ramps.min_jerk,
    ramps.constant_jerk,
    ramps.constant_accel,
    ramps.switching_constant_jerk,
    ramps.linear,
]

T0 = 3.0 * us
DURATION = 40.0 * us
Y_I = -2.0 * um
Y_F = 7.0 * um
DELTA = Y_F - Y_I


def _one_sided(p, k: int) -> tuple[float, float]:
    """Exact ``(left, right)`` limits of ``p`` at interior break ``k``, from coefficients."""
    return float(p.coeffs[k - 1].sum()), float(p.coeffs[k, 0])


@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.__name__)
def test_endpoints_and_domain(family):
    p = family(T0, DURATION, Y_I, Y_F)
    assert p.domain == (T0, T0 + DURATION)
    assert p(T0) == pytest.approx(Y_I, rel=RTOL)
    assert p(T0 + DURATION) == pytest.approx(Y_F, rel=RTOL)
    # monotone from y_i to y_f (no overshoot in any family)
    t = np.linspace(T0, T0 + DURATION, 2001)
    y = p(t)
    assert np.min(np.diff(y)) > -RTOL * abs(DELTA)
    assert np.max(y) <= Y_F + RTOL * abs(DELTA)


@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.__name__)
def test_rejects_non_positive_duration(family):
    with pytest.raises(ValueError, match="positive"):
        family(T0, 0.0, Y_I, Y_F)
    with pytest.raises(ValueError, match="positive"):
        family(T0, -1.0 * us, Y_I, Y_F)


def test_min_jerk_starts_and_ends_at_rest_with_zero_acceleration():
    """Eq. S14: ydot = yddot = 0 at both ends; peak |yddot| = 10 |d| / (sqrt(3) T^2)."""
    p = ramps.min_jerk(T0, DURATION, Y_I, Y_F)
    vel, acc = p.derivative(), p.derivative().derivative()
    v_scale = abs(DELTA) / DURATION
    a_scale = abs(DELTA) / DURATION**2
    for t in (T0, T0 + DURATION):
        assert abs(vel(t)) < RTOL * v_scale
        assert abs(acc(t)) < RTOL * a_scale
    t = np.linspace(T0, T0 + DURATION, 20001)
    assert np.max(np.abs(acc(t))) == pytest.approx(10.0 / np.sqrt(3.0) * a_scale, rel=1e-6)
    # 15/8 * d / T peak speed (analytic), reached at mid-move
    assert np.max(np.abs(vel(t))) == pytest.approx(1.875 * v_scale, rel=1e-6)


def test_constant_jerk_endpoint_accelerations_and_constant_jerk():
    """Eq. S15: ydot = 0 at the ends, yddot = +/-6 d / T^2, jerk constant = -12 d / T^3."""
    p = ramps.constant_jerk(T0, DURATION, Y_I, Y_F)
    vel = p.derivative()
    acc = vel.derivative()
    jerk = acc.derivative()
    v_scale = abs(DELTA) / DURATION
    for t in (T0, T0 + DURATION):
        assert abs(vel(t)) < RTOL * v_scale
    assert acc(T0) == pytest.approx(6.0 * DELTA / DURATION**2, rel=1e-12)
    assert acc(T0 + DURATION) == pytest.approx(-6.0 * DELTA / DURATION**2, rel=1e-12)
    t = np.linspace(T0, T0 + DURATION, 501)
    np.testing.assert_allclose(jerk(t), -12.0 * DELTA / DURATION**3, rtol=1e-12)


def test_constant_accel_two_parabolic_halves():
    """Eq. S16: yddot = +4 d/T^2 on the first half, -4 d/T^2 on the second."""
    p = ramps.constant_accel(T0, DURATION, Y_I, Y_F)
    assert p.n_segments == 2
    mid = T0 + 0.5 * DURATION
    np.testing.assert_allclose(p.breaks, [T0, mid, T0 + DURATION], rtol=1e-15)
    assert p(mid) == pytest.approx(Y_I + 0.5 * DELTA, rel=RTOL)

    vel, acc = p.derivative(), p.derivative().derivative()
    a_expected = 4.0 * DELTA / DURATION**2
    first = np.linspace(T0, mid, 101)[:-1]
    second = np.linspace(mid, T0 + DURATION, 101)[1:]
    np.testing.assert_allclose(acc(first), a_expected, rtol=1e-12)
    np.testing.assert_allclose(acc(second), -a_expected, rtol=1e-12)

    # velocity is continuous at the switch and vanishes at both ends
    left, right = _one_sided(vel, 1)
    assert left == pytest.approx(right, rel=1e-12)
    assert left == pytest.approx(2.0 * DELTA / DURATION, rel=1e-12)  # peak speed
    v_scale = abs(DELTA) / DURATION
    assert abs(vel(T0)) < RTOL * v_scale
    assert abs(vel(T0 + DURATION)) < RTOL * v_scale


def test_switching_constant_jerk_is_c2_at_both_switches():
    """Eq. S17: y, ydot, yddot continuous at T/4 and 3T/4; rest-to-rest with zero accel."""
    p = ramps.switching_constant_jerk(T0, DURATION, Y_I, Y_F)
    assert p.n_segments == 3
    np.testing.assert_allclose(
        p.breaks,
        [T0, T0 + 0.25 * DURATION, T0 + 0.75 * DURATION, T0 + DURATION],
        rtol=1e-15,
    )

    vel = p.derivative()
    acc = vel.derivative()
    jerk = acc.derivative()
    for poly, scale in (
        (p, abs(DELTA)),
        (vel, abs(DELTA) / DURATION),
        (acc, abs(DELTA) / DURATION**2),
    ):
        for k in (1, 2):  # the two switches
            left, right = _one_sided(poly, k)
            assert abs(left - right) < 1e-12 * scale, (poly, k)

    v_scale = abs(DELTA) / DURATION
    a_scale = abs(DELTA) / DURATION**2
    for t in (T0, T0 + DURATION):
        assert abs(vel(t)) < RTOL * v_scale
        assert abs(acc(t)) < RTOL * a_scale

    # jerk switches +J, -J, +J with J = 32 d / T^3; peak |acceleration| is 8 d / T^2
    j_expected = 32.0 * DELTA / DURATION**3
    probes = [
        T0 + 0.1 * DURATION,
        T0 + 0.5 * DURATION,
        T0 + 0.9 * DURATION,
    ]
    np.testing.assert_allclose(
        jerk(np.array(probes)), [j_expected, -j_expected, j_expected], rtol=1e-12
    )
    t = np.linspace(T0, T0 + DURATION, 20001)
    assert np.max(np.abs(acc(t))) == pytest.approx(8.0 * a_scale, rel=1e-6)


def test_linear_ramp_antiderivative_is_exact():
    """The phase of a constant-chirp segment: int (y_i + d tau) dt in closed form."""
    p = ramps.linear(T0, DURATION, Y_I, Y_F)
    np.testing.assert_allclose(
        p.derivative()(np.linspace(T0, T0 + DURATION, 51)), DELTA / DURATION, rtol=1e-12
    )

    integral = p.antiderivative()
    t = np.linspace(T0, T0 + DURATION, 1001)
    dt = t - T0
    expected = Y_I * dt + 0.5 * DELTA * dt**2 / DURATION
    scale = abs(Y_I) * DURATION + abs(DELTA) * DURATION
    assert np.max(np.abs(integral(t) - expected)) < RTOL * scale
    # trapezoid identity: the mean value of a linear ramp is (y_i + y_f) / 2
    assert integral(T0 + DURATION) == pytest.approx(0.5 * (Y_I + Y_F) * DURATION, rel=RTOL)


def test_hold_is_constant():
    p = ramps.hold(T0, DURATION, 4.0 * um)
    t = np.linspace(T0, T0 + DURATION, 101)
    np.testing.assert_allclose(p(t), 4.0 * um, rtol=0, atol=0)
    np.testing.assert_allclose(p.derivative()(t), 0.0, rtol=0, atol=0)
    with pytest.raises(ValueError, match="positive"):
        ramps.hold(T0, 0.0, 1.0)


def test_ramps_double_as_frequency_laws():
    """Same profiles, MHz ordinate: the min-jerk mean value is exactly d/2 (Eq. S19)."""
    f = ramps.min_jerk(0.0, 100.0 * us, 0.0, 2.0 * MHz)
    assert f(0.0) == 0.0
    assert f(100.0 * us) == pytest.approx(2.0 * MHz, rel=RTOL)
    cycles = f.antiderivative()(100.0 * us)
    assert cycles == pytest.approx(0.5 * 2.0 * MHz * 100.0 * us, rel=RTOL)  # 100 cycles

    # chirp rate (which sets Zbar and Delta F, Table I) vanishes at both ends
    fdot = f.derivative()
    assert abs(fdot(0.0)) < RTOL * 2.0 * MHz / (100.0 * us)
    assert abs(fdot(100.0 * us)) < RTOL * 2.0 * MHz / (100.0 * us)


def test_ramps_registry_matches_module_functions():
    assert set(ramps.RAMPS) == {f.__name__ for f in FAMILIES}
    for name, fn in ramps.RAMPS.items():
        assert fn is getattr(ramps, name)
