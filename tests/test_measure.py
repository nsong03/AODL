r"""Analytic spot metrics: Table I positions, focus, astigmatic interval, waists, power.

As in ``test_focal.py`` the device layer is not imported — terms are synthetic arrays with the
frozen ``device.aodl.TermArray`` layout (WO-03 §3).  Most assertions are algebra identities
(``measure`` must return exactly the closed-form quantities the term was built from); the power
closed form is checked against an independent numerical integral, both in the image plane (via
``intensity_frame``) and on the pupil side (via ``scipy.integrate.quad``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from scipy.integrate import quad

from aodl.field.focal import Z_LAB_SIGN, FrameGrid, intensity_frame, spot_params
from aodl.field.measure import measure, track_z
from aodl.units import MHz

N_TRIALS = 5


@dataclass
class TermArray:
    """Structural stand-in for ``device.aodl.TermArray`` (WO-03 §3)."""

    c: Any
    theta1: Any
    theta2: Any
    alpha: Any
    df_opt: Any
    edge: Any = (None, None)


def make_terms(
    *,
    c: Any,
    theta1: Any,
    theta2: Any,
    alpha: Any = None,
    df_opt: Any = 0.0,
    edge: Any = (None, None),
) -> TermArray:
    """Build a synthetic term array; ``theta*`` are ``(2, N)`` (or broadcastable)."""
    c_arr = np.atleast_1d(np.asarray(c, dtype=np.complex128))
    n = c_arr.size
    th1 = np.broadcast_to(np.asarray(theta1, dtype=np.float64).reshape(2, -1), (2, n)).copy()
    th2 = np.broadcast_to(np.asarray(theta2, dtype=np.float64).reshape(2, -1), (2, n)).copy()
    if alpha is None:
        alpha_arr = np.zeros((2, 3, n), dtype=np.complex128)
        alpha_arr[:, 0, :] = 1.0
    else:
        alpha_arr = np.broadcast_to(
            np.asarray(alpha, dtype=np.complex128).reshape(2, 3, -1), (2, 3, n)
        ).copy()
    df = np.broadcast_to(np.asarray(df_opt, dtype=np.float64).ravel(), (n,)).copy()
    return TermArray(c=c_arr, theta1=th1, theta2=th2, alpha=alpha_arr, df_opt=df, edge=edge)


def test_single_term_metrics_are_the_closed_form(params1030, rng):
    """Random single terms: ``measure`` returns exactly the quantities they were built from."""
    optics = params1030.optics
    k, focal = optics.k, optics.focal_length
    z_r = optics.rayleigh

    for _ in range(N_TRIALS):
        position = rng.uniform(-20.0, 20.0, 2) * optics.waist0
        focus_lab = rng.uniform(-2.0, 2.0, 2) * z_r
        df_opt = rng.uniform(-5.0, 5.0) * MHz
        theta1 = position * k / focal
        # Z_axis,lab = Z_LAB_SIGN 2 F^2 theta2 / k, inverted.
        theta2 = Z_LAB_SIGN * focus_lab * k / (2.0 * focal**2)
        terms = make_terms(
            c=[rng.uniform(0.5, 1.5) * np.exp(2j * np.pi * rng.uniform())],
            theta1=theta1.reshape(2, 1),
            theta2=theta2.reshape(2, 1),
            df_opt=df_opt,
        )

        (spot,) = measure(terms, optics)
        assert spot.x == pytest.approx(position[0], rel=1e-9)
        assert spot.y == pytest.approx(position[1], rel=1e-9)
        assert spot.z_lab == pytest.approx(0.5 * (focus_lab[0] + focus_lab[1]), rel=1e-9)
        assert spot.delta_f == pytest.approx(focus_lab[0] - focus_lab[1], rel=1e-9)
        assert spot.sigma_astig == pytest.approx(spot.delta_f / z_r, rel=1e-12)
        assert spot.df_opt == pytest.approx(df_opt, rel=1e-12)

        # Waists at the best-focus plane follow the textbook Gaussian law about each axis focus.
        for waist, focus in ((spot.wx, focus_lab[0]), (spot.wy, focus_lab[1])):
            expected = optics.waist0 * np.sqrt(1.0 + ((spot.z_lab - focus) / z_r) ** 2)
            assert waist == pytest.approx(expected, rel=1e-12)


def test_symmetric_astigmatic_term(params1030):
    """theta2x = -theta2y: mean focus at Z = 0, ``|delta_f| = 2 |Z_axis|``, waists ordered.

    The work order also asks for ``wx(z=0) > wy(z=0)``; for this deliberately symmetric term
    that is false by construction — ``z = 0`` is the circle of least confusion and the two
    radii are *equal* there (they depend on ``|a|``, and the two ``a`` are conjugates).  The
    ordering statement is therefore checked where it has content: in the x focal plane, where
    ``wx = waist0`` and ``wy = waist0 sqrt(1 + sigma_astig^2)``.
    """
    optics = params1030.optics
    k, focal = optics.k, optics.focal_length
    z_r = optics.rayleigh

    z_x_lab = 0.5 * z_r
    theta2x = Z_LAB_SIGN * z_x_lab * k / (2.0 * focal**2)
    terms = make_terms(c=[1.0], theta1=np.zeros((2, 1)), theta2=[[theta2x], [-theta2x]], df_opt=0.0)

    (spot,) = measure(terms, optics)
    assert spot.z_lab == pytest.approx(0.0, abs=1e-12 * z_r)
    assert spot.delta_f == pytest.approx(2.0 * z_x_lab, rel=1e-12)
    assert abs(spot.delta_f) == pytest.approx(2.0 * abs(z_x_lab), rel=1e-12)
    assert spot.sigma_astig == pytest.approx(1.0, rel=1e-12)
    # Circle of least confusion: the two radii coincide at the mean focus.
    assert spot.wx == pytest.approx(spot.wy, rel=1e-12)
    assert spot.wx == pytest.approx(optics.waist0 * np.sqrt(1.25), rel=1e-12)

    # In the x focal plane that axis is sharp and the other is sigma_astig away from focus.
    _, _, wx, wy = spot_params(terms, optics, z_x_lab)
    assert wx[0] == pytest.approx(optics.waist0, rel=1e-12)
    assert wy[0] == pytest.approx(optics.waist0 * np.sqrt(1.0 + spot.sigma_astig**2), rel=1e-12)
    assert wy[0] > wx[0]


def test_track_z_is_power_weighted(params1030):
    """Two frequency groups with |c|^2 in a 1:4 ratio pull the tracked plane 4:1."""
    optics = params1030.optics
    k, focal = optics.k, optics.focal_length
    z_r = optics.rayleigh

    z_lab = np.array([1.0, -0.5]) * z_r
    theta2 = Z_LAB_SIGN * z_lab * k / (2.0 * focal**2)
    terms = make_terms(
        c=[1.0, 2.0],
        theta1=np.zeros((2, 2)),
        theta2=np.stack([theta2, theta2]),
        df_opt=[0.0, 3.0 * MHz],
    )

    metrics = measure(terms, optics)
    assert len(metrics) == 2
    assert [m.z_lab for m in metrics] == [
        pytest.approx(z_lab[0], rel=1e-12),
        pytest.approx(z_lab[1], rel=1e-12),
    ]
    assert metrics[1].power == pytest.approx(4.0 * metrics[0].power, rel=1e-12)
    assert track_z(metrics) == pytest.approx((z_lab[0] + 4.0 * z_lab[1]) / 5.0, rel=1e-12)
    assert track_z([]) == 0.0


def test_power_matches_the_rendered_frame(params1030):
    """``power`` is the integral of ``intensity_frame`` — same (prefactor-dropped) scale."""
    optics = params1030.optics
    w0 = optics.waist0
    alpha = np.zeros((2, 3, 1), dtype=np.complex128)
    alpha[:, 0, 0] = 1.0
    alpha[0, 1, 0] = 0.35 / optics.w_in
    alpha[1, 2, 0] = -0.5 / optics.w_in**2
    terms = make_terms(
        c=[0.8 + 0.6j], theta1=np.zeros((2, 1)), theta2=np.zeros((2, 1)), alpha=alpha
    )

    grid = FrameGrid(-10.0 * w0, 10.0 * w0, 501, -10.0 * w0, 10.0 * w0, 501)
    frame = intensity_frame(terms, optics, grid, 0.0)
    (spot,) = measure(terms, optics)
    assert frame.sum() * grid.dx * grid.dy == pytest.approx(spot.power, rel=1e-8)


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_power_closed_form_with_irising_and_fill_edge(params1030, side):
    """The Parseval power is exact for an irising polynomial on a partially filled aperture."""
    optics = params1030.optics
    u_edge = -0.5 * optics.w_in if side == "lower" else 0.5 * optics.w_in
    alpha_axes = np.array(
        [
            [1.0, 0.4 / optics.w_in, -0.6 / optics.w_in**2],
            [1.0, -0.2 / optics.w_in, 0.3 / optics.w_in**2],
        ]
    )
    alpha = np.zeros((2, 3, 1), dtype=np.complex128)
    alpha[:, :, 0] = alpha_axes
    terms = make_terms(
        c=[1.3 - 0.4j],
        theta1=[[optics.k * optics.waist0 / optics.focal_length], [0.0]],
        theta2=[[optics.k * optics.rayleigh / (2.0 * optics.focal_length**2)], [0.0]],
        alpha=alpha,
        edge=((u_edge, side), None),
    )

    def sq_pupil(axis: int) -> Any:
        a0, a1, a2 = alpha_axes[axis]

        def integrand(u: float) -> float:
            poly = a0 + a1 * u + a2 * u * u
            return float(abs(poly) ** 2 * np.exp(-2.0 * u**2 / optics.w_in**2))

        return integrand

    lo = u_edge if side == "lower" else -np.inf
    hi = np.inf if side == "lower" else u_edge
    s_x = quad(sq_pupil(0), lo, hi, limit=200)[0]
    s_y = quad(sq_pupil(1), -np.inf, np.inf, limit=200)[0]
    expected = abs(terms.c[0]) ** 2 * (optics.wavelength * optics.focal_length) ** 2 * s_x * s_y

    (spot,) = measure(terms, optics)
    assert spot.power == pytest.approx(expected, rel=1e-10)
