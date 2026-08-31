r"""Pins the Eq. S11 geometry mapping that the device and field layers build on.

For a separable pupil factor on one axis,

    pupil(u) = (a0 + a1 u + a2 u^2) exp(-u^2/w_in^2 + i th2 u^2 + i th1 u)

Eq. S11 multiplies in ``exp(-i k u X / F) exp(-i k Z u^2 / (2 F^2))``, so the axis
integral is exactly ``\int (a0 + a1 u + a2 u^2) e^{-a u^2 + b u} du`` with

    a = 1/w_in^2 - i (th2 - k Z / (2 F^2))          <- beam radius, chirp lens, defocus
    b = i (th1 - k X / F)                           <- deflection, image coordinate

Downstream code (``device/conventions.py``, ``field/focal.py``) must use exactly this
mapping; the tests here fix it against the brute-force quadrature backend, including the
sign of the defocus term.
"""

from __future__ import annotations

import numpy as np
import pytest

from aodl.field.gaussian import gauss_moments
from aodl.field.reference import reference_field_2d, reference_field_separable
from aodl.units import MHz, ms, um, us

N_TRIALS = 3


def _pupil(optics, alpha, th1, th2):
    """Synthetic separable pupil factor, input-beam Gaussian included."""

    def pupil(u):
        poly = alpha[0] + alpha[1] * u + alpha[2] * u * u
        return poly * np.exp(-(u**2) / optics.w_in**2 + 1j * th2 * u * u + 1j * th1 * u)

    return pupil


def _analytic_axis(optics, alpha, th1, th2, coord, z_s11):
    """Closed-form axis factor: a0 I0 + a1 I1 + a2 I2 with the (a, b) mapping above."""
    k, focal = optics.k, optics.focal_length
    a = 1.0 / optics.w_in**2 - 1j * (th2 - k * np.asarray(z_s11) / (2.0 * focal**2))
    b = 1j * (th1 - k * np.asarray(coord) / focal)
    i0, i1, i2 = gauss_moments(a, b)
    return alpha[0] * i0 + alpha[1] * i1 + alpha[2] * i2


def _draw_term(rng, optics):
    """Modest random (alpha, th1, th2): deflection within +-3 waists, focus within +-1 z_R."""
    k, focal = optics.k, optics.focal_length
    th1 = rng.uniform(-3.0, 3.0) * k * optics.waist0 / focal
    th2 = rng.uniform(-1.0, 1.0) * k * optics.rayleigh / (2.0 * focal**2)
    alpha = np.array(
        [
            1.0,
            rng.uniform(-1.0, 1.0) / optics.w_in,
            rng.uniform(-1.0, 1.0) / optics.w_in**2,
        ]
    )
    return alpha, th1, th2


def _peak_normalize(field, index):
    return field / field[index]


def test_analytic_matches_reference_on_xz_patch(params1030, rng):
    """21x21 (X, Z) patch, relative 1e-8 after peak normalization."""
    optics = params1030.optics
    k, focal = optics.k, optics.focal_length
    for _ in range(N_TRIALS):
        ax, th1x, th2x = _draw_term(rng, optics)
        ay, th1y, th2y = _draw_term(rng, optics)

        x_centre = th1x * focal / k
        z_centre = 2.0 * focal**2 * th2x / k
        x_axis = x_centre + np.linspace(-3.0, 3.0, 21) * optics.waist0
        z_axis = z_centre + np.linspace(-2.0, 2.0, 21) * optics.rayleigh
        xx, zz = np.meshgrid(x_axis, z_axis, indexing="ij")

        reference = reference_field_separable(
            _pupil(optics, ax, th1x, th2x),
            _pupil(optics, ay, th1y, th2y),
            optics,
            xx,
            0.0,
            zz,
        )
        analytic = _analytic_axis(optics, ax, th1x, th2x, xx, zz) * _analytic_axis(
            optics, ay, th1y, th2y, 0.0, zz
        )
        peak = np.unravel_index(np.argmax(np.abs(reference)), reference.shape)
        deviation = np.abs(_peak_normalize(reference, peak) - _peak_normalize(analytic, peak))
        assert deviation.max() < 1e-8


def test_analytic_matches_reference_on_xy_patch(params1030, rng):
    """Both axes at once, off-focus: 21x21 (X, Y) patch at fixed Z."""
    optics = params1030.optics
    k, focal = optics.k, optics.focal_length
    ax, th1x, th2x = _draw_term(rng, optics)
    ay, th1y, th2y = _draw_term(rng, optics)
    z_s11 = 0.7 * optics.rayleigh

    x_axis = th1x * focal / k + np.linspace(-3.0, 3.0, 21) * optics.waist0
    y_axis = th1y * focal / k + np.linspace(-3.0, 3.0, 21) * optics.waist0
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="ij")

    reference = reference_field_separable(
        _pupil(optics, ax, th1x, th2x),
        _pupil(optics, ay, th1y, th2y),
        optics,
        xx,
        yy,
        z_s11,
    )
    analytic = _analytic_axis(optics, ax, th1x, th2x, xx, z_s11) * _analytic_axis(
        optics, ay, th1y, th2y, yy, z_s11
    )
    peak = np.unravel_index(np.argmax(np.abs(reference)), reference.shape)
    deviation = np.abs(_peak_normalize(reference, peak) - _peak_normalize(analytic, peak))
    assert deviation.max() < 1e-8


def test_unaberrated_waist_is_optics_waist0(params1030):
    """th1 = th2 = 0, Z = 0: the focal spot is the textbook lambda F / (pi w_in) Gaussian."""
    optics = params1030.optics
    alpha = np.array([1.0, 0.0, 0.0])
    pupil = _pupil(optics, alpha, 0.0, 0.0)

    x_axis = np.linspace(-1.5, 1.5, 61) * optics.waist0
    field = reference_field_separable(pupil, pupil, optics, x_axis, 0.0, 0.0)
    intensity = np.abs(field) ** 2
    slope = np.polyfit(x_axis**2, np.log(intensity / intensity.max()), 1)[0]
    waist_fit = np.sqrt(-2.0 / slope)
    assert waist_fit == pytest.approx(optics.waist0, rel=1e-6)

    analytic = _analytic_axis(optics, alpha, 0.0, 0.0, x_axis, 0.0) * _analytic_axis(
        optics, alpha, 0.0, 0.0, 0.0, 0.0
    )
    analytic_intensity = np.abs(analytic) ** 2
    slope_a = np.polyfit(x_axis**2, np.log(analytic_intensity / analytic_intensity.max()), 1)[0]
    assert np.sqrt(-2.0 / slope_a) == pytest.approx(optics.waist0, rel=1e-6)

    # Rayleigh range: the 2D on-axis intensity (both axis factors) halves one z_R out.
    axis_factor = _analytic_axis(optics, alpha, 0.0, 0.0, 0.0, np.array([0.0, optics.rayleigh]))
    on_axis = np.abs(axis_factor**2) ** 2
    assert on_axis[1] / on_axis[0] == pytest.approx(0.5, rel=1e-9)


@pytest.mark.parametrize("th2_signed", [+0.6, -0.6])
def test_defocus_sign_is_pinned(params1030, th2_signed):
    """A pupil quadratic phase th2 focuses at Z_S11 = 2 F^2 th2 / k, sign included."""
    optics = params1030.optics
    k, focal = optics.k, optics.focal_length
    alpha = np.array([1.0, 0.0, 0.0])
    th2 = th2_signed * k * optics.rayleigh / (2.0 * focal**2)
    z_star = 2.0 * focal**2 * th2 / k
    assert z_star == pytest.approx(th2_signed * optics.rayleigh, rel=1e-12)

    z_axis = np.linspace(-3.0, 3.0, 601) * optics.rayleigh
    on_axis = _analytic_axis(optics, alpha, 0.0, th2, 0.0, z_axis) ** 2
    intensity = np.abs(on_axis) ** 2
    imax = int(np.argmax(intensity))
    assert 0 < imax < intensity.size - 1
    y0, y1, y2 = intensity[imax - 1 : imax + 2]
    dz = z_axis[1] - z_axis[0]
    z_peak = z_axis[imax] + 0.5 * dz * (y0 - y2) / (y0 - 2.0 * y1 + y2)
    assert abs(z_peak - z_star) < 1e-6 * optics.rayleigh

    # The quadrature backend agrees: brightest at z_star, symmetric about it.
    pupil = _pupil(optics, alpha, 0.0, th2)
    probe = np.array([z_star - optics.rayleigh, z_star, z_star + optics.rayleigh])
    reference = np.abs(reference_field_separable(pupil, pupil, optics, 0.0, 0.0, probe)) ** 2
    assert reference[1] > reference[0]
    assert reference[1] > reference[2]
    assert reference[0] == pytest.approx(reference[2], rel=1e-9)
    # One z_R out from the shifted focus the 2D on-axis intensity is halved.
    assert reference[0] / reference[1] == pytest.approx(0.5, rel=1e-9)


def test_preset_scales_match_plan(params1030):
    """The 1030 nm preset reproduces the handy scales quoted in ``docs/PLAN.md`` §1.5."""
    optics = params1030.optics
    assert optics.waist0 == pytest.approx(1.07 * um, rel=5e-3)
    assert optics.rayleigh == pytest.approx(3.46 * um, rel=5e-3)
    assert params1030.channels["Ax"].transit_time == pytest.approx(11.54 * us, rel=1e-3)
    # 1 MHz of frequency difference <-> 10.3 um of lateral motion (Table I).
    assert params1030.deflection_scale * MHz == pytest.approx(10.3 * um, rel=5e-3)
    # Co-chirping all four channels at beta reaches Zbar = 2 * lens_scale * beta;
    # 10 um needs beta ~ 48.5 MHz/ms.
    beta = 10.0 * um / (2.0 * params1030.lens_scale)
    assert beta == pytest.approx(48.5 * MHz / ms, rel=5e-3)


def test_reference_2d_matches_separable(params1030):
    """The non-separable backend reproduces the separable one on a product pupil."""
    optics = params1030.optics
    alpha = np.array([1.0, 0.0, 0.0])
    th1 = 2.0 * optics.k * optics.waist0 / optics.focal_length
    px = _pupil(optics, alpha, th1, 0.0)
    py = _pupil(optics, alpha, 0.0, 0.0)

    grid = np.linspace(-2.0, 2.0, 9) * optics.waist0
    xx, yy = np.meshgrid(grid + th1 * optics.focal_length / optics.k, grid, indexing="ij")
    z_s11 = 0.4 * optics.rayleigh

    two_d = reference_field_2d(lambda x, y: px(x) * py(y), optics, xx, yy, z_s11, n=801, span=6.0)
    separable = reference_field_separable(px, py, optics, xx, yy, z_s11, n=801, span=6.0)
    assert np.abs(two_d - separable).max() < 1e-12 * np.abs(separable).max()
