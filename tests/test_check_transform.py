r"""M6 §4: the zoom (chirp-z) transform of Eq. S11, and the sub-time schedule.

Three claims are pinned.

**The CZT is the sum it says it is.**  ``zoom_field`` is compared against the literal matrix
DFT it stands for — the same ``du sum_n P(u_n) e^{-i k u_n X / F}``, written out.  Measured
3.5e-14, 7.7e-14 and 4.0e-13 relative on 1024-, 4096- and 24576-cell grids, so the work
order's 1e-12 holds on all of them with room to spare.  The residual is the Bluestein chirp's
own round-off (``scipy.signal.czt`` raises ``w`` to ``N^2/2``, whose phase reaches 2e5 rad at
``N = 24576``), which is why it grows with the grid; the matrix DFT it is measured against is
itself within 1e-15 of an extended-precision evaluation, so this is the CZT's error and not
the comparison's.

**It is Eq. S11.**  Against ``field/reference.py``, the brute-force quadrature backend the
whole simulator is validated with, fed the same literal pupil — 1e-6, measured 1e-8.  The
two share no code: one is a trapezoid sum over its own dense grid, the other a chirp-z
transform of a sampled pupil.

**The axial sign is the package's.**  A pupil carrying the chirp-lens curvature of
``conventions.theta2_contribution(fdot)`` focuses at lab ``Z = +lens_scale fdot`` — the
Table I sign, i.e. an up-chirp puts the tweezer *above* the static focal plane
(``docs/conventions.md`` §6).  Getting ``Z_LAB_SIGN`` backwards here would flip the axial
half of every M6 verdict, so it is pinned the WO-01 way: build the pupil from the sign
authority, scan planes, and fit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aodl.check.metrics import best_focus, fit_gaussian_1d
from aodl.check.pupil import ApertureGrid
from aodl.check.transform import GOLDEN_RATIO, subtimes, zoom_field
from aodl.device import conventions
from aodl.device.conventions import geometry
from aodl.field.reference import reference_field_separable
from aodl.units import MHz, ms, um, us


def _grid(params, n: int) -> ApertureGrid:
    """A grid of ``n`` cells spanning the pinned aperture — for the size study."""
    design = ApertureGrid.design(params, "weak")
    du = 2.0 * design.half_span / n
    return ApertureGrid(u=(np.arange(n, dtype=np.float64) - n // 2) * du, du=du)


def _matrix_dft(pupil, grid, optics, coords, z_lab):
    """The literal sum ``zoom_field`` claims to evaluate, written as a dense matrix product."""
    z_s11 = float(conventions.z_s11_from_lab(z_lab))
    kernel = np.exp(-1j * optics.k * z_s11 * grid.u**2 / (2.0 * optics.focal_length**2)) * np.exp(
        -1j * (optics.k / optics.focal_length) * np.outer(np.asarray(coords), grid.u)
    )
    return grid.du * kernel @ np.asarray(pupil)


def _deflected_pupil(params, grid, detuning=3.0 * MHz, fdot=0.0):
    """A physically shaped pupil: input beam x deflection tilt x chirp-lens curvature."""
    theta1 = float(conventions.theta1_contribution(detuning, geometry("Ay"), params.sound_speed))
    theta2 = float(conventions.theta2_contribution(fdot, params.sound_speed))
    return np.exp(
        -((grid.u / params.optics.w_in) ** 2) + 1j * theta1 * grid.u + 1j * theta2 * grid.u**2
    )


# =============================================================== 1. czt vs direct DFT


def test_czt_matches_the_direct_matrix_dft(params1030) -> None:
    """The zoom transform *is* the Riemann sum, to the chirp's own round-off."""
    optics = params1030.optics
    coords = np.linspace(-30.0 * um, 30.0 * um, 65)
    small = _grid(params1030, 1024)
    pupil = _deflected_pupil(params1030, small)
    got = zoom_field(pupil, small, optics, coords, 0.0)
    want = _matrix_dft(pupil, small, optics, coords, 0.0)
    assert float(np.abs(got - want).max() / np.abs(want).max()) < 1e-12

    # The production grids, with and without defocus.  Round-off grows with the Bluestein
    # chirp's N^2 phase — measured 7.7e-14 at 4096 cells and 4.0e-13 at 24576.
    for mode in ("weak", "bragg_band"):
        grid = ApertureGrid.design(params1030, mode)
        pupil = _deflected_pupil(params1030, grid, fdot=50.0 * MHz / ms)
        for z_lab in (0.0, 4.0 * um):
            got = zoom_field(pupil, grid, optics, coords, z_lab)
            want = _matrix_dft(pupil, grid, optics, coords, z_lab)
            assert float(np.abs(got - want).max() / np.abs(want).max()) < 1e-12

    # A single image point is a degenerate zoom (no ratio), and must still be the same sum.
    grid = ApertureGrid.design(params1030, "weak")
    pupil = _deflected_pupil(params1030, grid)
    one = zoom_field(pupil, grid, optics, np.array([7.0 * um]), 0.0)
    assert one.shape == (1,)
    assert one[0] == pytest.approx(
        complex(_matrix_dft(pupil, grid, optics, [7.0 * um], 0.0)[0]), rel=1e-11
    )


def test_zoom_field_batches_and_refuses_a_non_uniform_grid(params1030) -> None:
    """Leading axes (sub-times, Z planes) pass through; a non-uniform ``coords`` raises."""
    grid = ApertureGrid.design(params1030, "weak")
    coords = np.linspace(-10.0 * um, 10.0 * um, 33)
    batch = np.stack(
        [_deflected_pupil(params1030, grid, detuning=f) for f in (1.0 * MHz, 3.0 * MHz, -2.0 * MHz)]
    ).reshape(3, 1, grid.n)
    out = zoom_field(batch, grid, params1030.optics, coords, 1.0 * um)
    assert out.shape == (3, 1, coords.size)
    for i in range(3):
        single = zoom_field(batch[i, 0], grid, params1030.optics, coords, 1.0 * um)
        np.testing.assert_allclose(out[i, 0], single, rtol=1e-13, atol=0.0)

    with pytest.raises(ValueError, match="uniformly spaced"):
        zoom_field(batch[0, 0], grid, params1030.optics, np.array([0.0, 1e-6, 3e-6]), 0.0)
    with pytest.raises(ValueError, match="last axis must match"):
        zoom_field(np.ones(17), grid, params1030.optics, coords, 0.0)


# =============================================================== 2. czt vs quadrature


def test_zoom_field_matches_the_quadrature_reference(params1030) -> None:
    """Eq. S11 two ways: chirp-z on a sampled pupil vs ``field/reference.py``'s trapezoid.

    A deflected, chirp-lensed pupil on one axis and the bare beam on the other, over a patch
    around the expected spot and through focus.  1e-6 asked, ~1e-8 measured.
    """
    optics = params1030.optics
    grid = ApertureGrid.design(params1030, "weak")
    detuning, fdot = 3.0 * MHz, 50.0 * MHz / ms
    py = _deflected_pupil(params1030, grid, detuning=detuning, fdot=fdot)
    px = np.exp(-((grid.u / optics.w_in) ** 2))

    theta1 = float(
        conventions.theta1_contribution(detuning, geometry("Ay"), params1030.sound_speed)
    )
    theta2 = float(conventions.theta2_contribution(fdot, params1030.sound_speed))
    y_spot = theta1 * optics.focal_length / optics.k
    y_axis = y_spot + np.linspace(-4.0, 4.0, 41) * optics.waist0

    # The reference builds the same pupil in closed form on its own dense grid, so the
    # comparison is of two independent quadratures of one analytic function.
    def beam(u):
        return np.exp(-((u / optics.w_in) ** 2))

    def lensed(u):
        return beam(u) * np.exp(1j * theta1 * u + 1j * theta2 * u**2)

    for z in (0.0, params1030.lens_scale * fdot):
        mine = (
            zoom_field(px, grid, optics, np.array([0.0]), z)[0]
            * zoom_field(py, grid, optics, y_axis, z)
            / (1j * optics.wavelength * optics.focal_length)
        )
        want = reference_field_separable(
            beam, lensed, optics, 0.0, y_axis, float(conventions.z_s11_from_lab(z))
        )
        error = float(np.abs(mine - want).max() / np.abs(want).max())
        assert error < 1e-6, f"z_lab = {z}, error = {error}"


# =============================================================== 3. the axial sign


@pytest.mark.parametrize("fdot", [+50.0 * MHz / ms, -50.0 * MHz / ms])
def test_defocus_sign_follows_z_lab_sign(params1030, fdot) -> None:
    """A chirp-lens pupil focuses at lab ``Z = +lens_scale fdot`` (Table I, ``Z_LAB_SIGN``).

    The pupil's curvature comes from :func:`aodl.device.conventions.theta2_contribution`, the
    defocus from :func:`aodl.device.conventions.z_s11_from_lab` inside ``zoom_field``, and
    nothing in between restates a sign.  An up-chirp on one channel makes a cylindrical lens
    whose focus sits *above* the static plane — the M1 phenomenology of
    ``docs/conventions.md`` §6.
    """
    optics = params1030.optics
    grid = ApertureGrid.design(params1030, "weak")
    pupil = _deflected_pupil(params1030, grid, detuning=0.0, fdot=fdot)
    expected = params1030.lens_scale * fdot
    assert abs(expected) == pytest.approx(5.15 * um, rel=0.02)
    assert abs(expected) > optics.rayleigh  # the two planes are distinguishable

    coords = np.linspace(-6.0, 6.0, 241) * optics.waist0
    planes = expected + np.linspace(-1.0, 1.0, 7) * optics.rayleigh
    widths = []
    for z in planes:
        profile = np.abs(zoom_field(pupil, grid, optics, coords, float(z))) ** 2
        _, radius, _ = fit_gaussian_1d(coords, profile)
        widths.append(radius**2)
    focus = best_focus(planes, np.array(widths))
    assert focus == pytest.approx(expected, rel=2e-3)

    # ... and the waist there is the unaberrated one, so the fit is measuring a real focus.
    at_focus = np.abs(zoom_field(pupil, grid, optics, coords, focus)) ** 2
    _, radius, _ = fit_gaussian_1d(coords, at_focus)
    assert radius == pytest.approx(optics.waist0, rel=1e-3)
    # Flipping the lab sign would put the focus on the wrong side of zero.
    assert math.copysign(1.0, focus) == math.copysign(1.0, fdot)


# =============================================================== 4. sub-times


def test_subtimes_are_deterministic_and_inside_the_window() -> None:
    """``t_j = t + W (frac(j phi) - 1/2)``, in order, reproducible, and never outside."""
    t, window = 20.0 * us, 3.0 * us
    got = subtimes(t, window, 8)
    want = t + window * (np.mod(np.arange(8) * GOLDEN_RATIO, 1.0) - 0.5)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(got, subtimes(t, window, 8))
    assert got[0] == pytest.approx(t - 0.5 * window)
    assert np.all(np.abs(got - t) <= 0.5 * window)
    assert subtimes(t, 0.0, 4).tolist() == [t] * 4
    with pytest.raises(ValueError, match="at least one sub-time"):
        subtimes(t, window, 0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        subtimes(t, -1.0, 4)


def test_golden_subtimes_do_not_alias_a_commensurate_beat() -> None:
    r"""A uniform schedule passes the beats commensurate with it at full amplitude.

    The two schedules fail in different places, and that is the whole argument.  A uniform
    set of ``k`` instants annihilates every beat exactly *except* the multiples of ``k/W``,
    which it lets through untouched — and an array's beats are integer combinations of two
    tone spacings, i.e. a regular comb that can land on them systematically.  The golden
    schedule annihilates nothing exactly but lets nothing through either, except at the
    sparse Fibonacci multiples of ``1/W``, which no comb hits on purpose.
    """
    window, k = 4.0 * us, 64
    golden = subtimes(0.0, window, k)
    uniform = window * (np.arange(k) / k - 0.5)

    def residual(schedule, cycles):
        """|<exp(2 pi i f t)>| for a beat of ``cycles`` periods per window."""
        return np.abs(
            np.mean(np.exp(2j * np.pi * np.asarray(cycles)[:, None] / window * schedule), axis=1)
        )

    cycles = np.arange(1, 8 * k + 1)
    uniform_residual = residual(uniform, cycles)
    golden_residual = residual(golden, cycles)
    commensurate = cycles % k == 0

    # Uniform: exactly zero off its own multiples, exactly one on them.
    assert uniform_residual[~commensurate].max() < 1e-12
    np.testing.assert_allclose(uniform_residual[commensurate], 1.0, atol=1e-12)
    # Golden: nowhere exact, but at those same frequencies it is 20x down at worst.
    assert golden_residual[commensurate].max() < 0.05
    assert golden_residual.min() > 0.0

    # Below k/2 cycles per window — where an array's own beats live once W is chosen from
    # the smallest spacing — the golden schedule keeps every beat well suppressed.
    assert golden_residual[: k // 2].max() < 0.25
    # Its own weak spots are the Fibonacci counts (phi is the worst-approximable irrational).
    fibonacci = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377}
    assert int(cycles[int(np.argmax(golden_residual))]) in fibonacci
